"""RunPod serverless handler — GLM-OCR direct Pipeline (no Flask subprocess).

Architecture vs runpod/glm-ocr:
  OLD: RunPod → handler → HTTP → gunicorn/glmocr.server → Pipeline → HTTP → vLLM
  NEW: RunPod → handler → Pipeline → HTTP → vLLM

Benefits:
  - One fewer subprocess (no gunicorn/flask hop, ~5 ms saved per request)
  - concurrency_modifier=2: RunPod sends 2 concurrent jobs per worker
  - Both Pipeline.process() calls run in parallel threads; vLLM batches their
    OCR regions together via continuous batching — same gain as Modal gthread×2
  - _LAYOUT_GPU_SEMAPHORE serialises layout GPU passes to prevent CUDA OOM
  - No _VLLM_SEMAPHORE: let vLLM batch concurrent recognition streams

Startup (once per worker):
  1. Patch page_loader.py  — data:application/pdf URI support
  2. Patch _workers.py     — _LAYOUT_GPU_SEMAPHORE
  3. Start vLLM subprocess, wait for /health
  4. Load glmocr config, create Pipeline, call .start()

Per-request handler:
  Input:  {"images": ["data:application/pdf;base64,..."]}
  Output: {"json_result": [...], "markdown_result": "..."}
"""

import os
import re
import sys
import subprocess
import time
import uuid
import requests
import runpod

from glmocr.config import load_config
from glmocr.pipeline import Pipeline

MODEL         = os.environ.get("MODEL", "zai-org/GLM-OCR")
VLLM_PORT     = int(os.environ.get("VLLM_PORT", "8000"))
GPU_UTIL      = os.environ.get("GPU_MEM_UTIL", "0.80")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_TOKENS    = int(os.environ.get("MAX_TOKENS", "2048"))
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
CONFIG_PATH   = "/tmp/glmocr_config.yaml"

MTP_JSON = '{"method":"mtp","num_speculative_tokens":3}'

_vllm_proc = None
_pipeline  = None


# ── Config ────────────────────────────────────────────────────────────────────

def _write_config():
    import shutil
    src = "/usr/local/lib/python3.11/dist-packages/glmocr/config.yaml"
    shutil.copy(src, CONFIG_PATH)
    subs = [
        (r"enabled: true",    "enabled: false"),
        (r"api_port: 8080",   f"api_port: {VLLM_PORT}"),
        (r"# device: null",   'device: "cuda:0"'),
        (r"batch_size: 1",    "batch_size: 2"),
        (r"max_tokens: 8192", f"max_tokens: {MAX_TOKENS}"),
    ]
    text = open(CONFIG_PATH).read()
    for pattern, replacement in subs:
        text = re.sub(pattern, replacement, text, count=1)
    open(CONFIG_PATH, "w").write(text)
    print(f"[init] glmocr config written to {CONFIG_PATH}")


# ── Patches ───────────────────────────────────────────────────────────────────

def _patch_page_loader():
    """Patch installed page_loader.py to handle data:application/pdf URIs."""
    import pathlib
    pkg = pathlib.Path(
        "/usr/local/lib/python3.11/dist-packages/glmocr/dataloader/page_loader.py"
    )
    if not pkg.exists():
        import glmocr
        pkg = pathlib.Path(glmocr.__file__).parent / "dataloader" / "page_loader.py"

    src = pkg.read_text()
    if "data:application/pdf" in src:
        print("[init] page_loader already patched")
        return

    OLD_LOAD = '''        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        # Detect PDF
        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            return self._load_pdf(file_path)

        # Otherwise load as a single image page
        return [self._load_image(source)]'''

    NEW_LOAD = '''        if source.startswith("data:application/pdf"):
            _, b64data = source.split(",", 1)
            import base64 as _b64
            return self._load_pdf_bytes(_b64.b64decode(b64data))

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            return self._load_pdf(file_path)

        return [self._load_image(source)]'''

    OLD_ITER = '''        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            yield from self._iter_pdf(file_path)
        else:
            yield self._load_image(source)'''

    NEW_ITER = '''        if source.startswith("data:application/pdf"):
            _, b64data = source.split(",", 1)
            import base64 as _b64
            yield from self._iter_pdf_bytes(_b64.b64decode(b64data))
            return

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            yield from self._iter_pdf(file_path)
        else:
            yield self._load_image(source)'''

    assert OLD_LOAD in src, "patch target _load_source not found in page_loader.py"
    assert OLD_ITER in src, "patch target _iter_source not found in page_loader.py"
    patched = src.replace(OLD_LOAD, NEW_LOAD, 1).replace(OLD_ITER, NEW_ITER, 1)
    pkg.write_text(patched)
    print("[init] page_loader.py patched for data:application/pdf support")


def _patch_layout_semaphore():
    """Patch _workers.py to serialise concurrent layout GPU calls with Semaphore(1).

    Multiple concurrent RunPod jobs each run Pipeline.process() in separate
    threads, all sharing the same LayoutDetector GPU model.  Without a lock
    they run simultaneously, causing CUDA OOM.  The semaphore ensures only one
    GPU layout forward pass runs at a time; page loading and vLLM OCR stay
    fully concurrent so vLLM can batch both streams together.
    """
    import glmocr
    pkg_path = os.path.join(
        os.path.dirname(glmocr.__file__), "pipeline", "_workers.py"
    )
    src = open(pkg_path).read()
    if "_LAYOUT_GPU_SEMAPHORE" in src:
        print("[init] _workers.py: layout semaphore already present, skipping")
        return

    OLD_IMPORT = '''import queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed'''

    NEW_IMPORT = '''import queue
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# One GPU layout forward pass at a time across all concurrent requests.
_LAYOUT_GPU_SEMAPHORE = threading.Semaphore(1)'''

    OLD_PROCESS = '''    try:
        layout_results, vis_images = layout_detector.process(
            batch_images,
            save_visualization=save_visualization,
            global_start_idx=global_start_idx,
            use_polygon=use_polygon,
        )
        if vis_images:
            state.layout_vis_images.update(vis_images)'''

    NEW_PROCESS = '''    try:
        with _LAYOUT_GPU_SEMAPHORE:
            layout_results, vis_images = layout_detector.process(
                batch_images,
                save_visualization=save_visualization,
                global_start_idx=global_start_idx,
                use_polygon=use_polygon,
            )
        if vis_images:
            state.layout_vis_images.update(vis_images)'''

    assert OLD_IMPORT in src, \
        "patch target (imports) not found in _workers.py — glmocr version mismatch"
    assert OLD_PROCESS in src, \
        "patch target (_flush_layout_batch) not found in _workers.py — glmocr version mismatch"
    patched = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    patched = patched.replace(OLD_PROCESS, NEW_PROCESS, 1)
    open(pkg_path, "w").write(patched)
    print("[init] _workers.py: layout GPU semaphore(1) applied")


# ── vLLM subprocess ───────────────────────────────────────────────────────────

def _start_vllm():
    global _vllm_proc
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--served-model-name", "glm-ocr",
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", GPU_UTIL,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--max-num-seqs", "32",
        "--max-num-batched-tokens", "32768",
        "--no-enable-log-requests",
        "--speculative-config", MTP_JSON,
    ]
    _vllm_proc = subprocess.Popen(cmd, env=env)
    print(f"[init] vLLM started (pid {_vllm_proc.pid}), waiting for /health ...")
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2).ok:
                print("[init] vLLM ready")
                return
        except Exception:
            pass
        if _vllm_proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {_vllm_proc.returncode}")
        time.sleep(3)
    raise RuntimeError("vLLM did not become healthy within 300s")


# ── Pipeline startup ──────────────────────────────────────────────────────────

def _start_pipeline():
    global _pipeline
    config = load_config(CONFIG_PATH)
    _pipeline = Pipeline(config.pipeline)
    _pipeline.start()
    print("[init] glmocr Pipeline started")


# ── Response helper ───────────────────────────────────────────────────────────

def _build_response(json_result, markdown_result):
    """Match glmocr server.py response structure for client compatibility."""
    return {
        "json_result":     json_result,
        "markdown_result": markdown_result,
        "layout_details":  json_result,
        "md_results":      markdown_result,
        "data_info":       {"pages": []},
        "usage":           {},
        "model":           "glm-ocr",
        "id":              f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "created":         int(time.time()),
    }


# ── Initialisation (runs once per worker) ────────────────────────────────────

_write_config()
_patch_page_loader()
_patch_layout_semaphore()
_start_vllm()
_start_pipeline()
print("[init] Worker ready — accepting up to 2 concurrent jobs")


# ── Per-request handler ───────────────────────────────────────────────────────

def handler(job):
    job_input = job.get("input", {})
    images = job_input.get("images", [])
    if isinstance(images, str):
        images = [images]
    # MaaS compat: some clients send "file" instead of "images"
    if not images and "file" in job_input:
        file_val = job_input["file"]
        if isinstance(file_val, str) and file_val:
            images = [file_val]
    if not images:
        return {"error": "No images provided"}

    # Build request_data in the format Pipeline.process() expects
    messages = [{"role": "user", "content": []}]
    for image_url in images:
        messages[0]["content"].append(
            {"type": "image_url", "image_url": {"url": image_url}}
        )
    request_data = {"messages": messages}

    try:
        results = list(_pipeline.process(request_data))
        if not results:
            return _build_response(None, "")
        if len(results) == 1:
            r = results[0]
            return _build_response(r.json_result, r.markdown_result or "")
        # Multiple input units (rare — usually a single PDF per call)
        json_result     = [r.json_result for r in results]
        markdown_result = "\n\n---\n\n".join(r.markdown_result or "" for r in results)
        return _build_response(json_result, markdown_result)
    except Exception as e:
        err = str(e)
        if "CUDA" in err or "out of memory" in err.lower():
            print(f"[handler] CUDA error — exiting for worker restart: {err}")
            sys.exit(1)
        return {"error": err}


runpod.serverless.start({
    "handler": handler,
    # Tell RunPod it may send 2 jobs to this worker simultaneously.
    # Both Pipeline.process() calls run in parallel threads; vLLM batches their
    # OCR regions together, matching Modal's max_inputs=2 behaviour.
    "concurrency_modifier": lambda x: 2,
})
