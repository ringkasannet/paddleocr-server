"""RunPod serverless handler for GLM-OCR.

Startup (runs once per worker, before any request is accepted):
  1. Write glmocr config
  2. Start vLLM subprocess, wait for /health
  3. Start glmocr subprocess, wait for /glmocr/parse to accept connections

Per-request handler:
  - Forwards the job input directly to glmocr's /glmocr/parse endpoint
  - Input: {"images": ["data:application/pdf;base64,..."]} (same format as glmocr /parse)
  - Output: glmocr JSON response
"""

import os
import subprocess
import time
import requests
import runpod

MODEL          = os.environ.get("MODEL", "zai-org/GLM-OCR")
VLLM_PORT      = int(os.environ.get("VLLM_PORT", "8000"))
GLMOCR_PORT    = int(os.environ.get("GLMOCR_PORT", "5002"))
GPU_UTIL       = os.environ.get("GPU_MEM_UTIL", "0.80")
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "4096"))
MAX_TOKENS     = int(os.environ.get("MAX_TOKENS", "2048"))
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
CONFIG_PATH    = "/tmp/glmocr_config.yaml"

MTP_JSON = '{"method":"mtp","num_speculative_tokens":3}'

_vllm_proc   = None
_glmocr_proc = None


def _write_config():
    import shutil, re
    src = "/usr/local/lib/python3.11/dist-packages/glmocr/config.yaml"
    shutil.copy(src, CONFIG_PATH)
    subs = [
        (r"port: 5002",      f"port: {GLMOCR_PORT}"),
        (r"enabled: true",   "enabled: false"),
        (r"api_port: 8080",  f"api_port: {VLLM_PORT}"),
        (r"# device: null",  'device: "cuda:0"'),
        (r"batch_size: 1",   "batch_size: 2"),
        (r"max_tokens: 8192", f"max_tokens: {MAX_TOKENS}"),
    ]
    text = open(CONFIG_PATH).read()
    for pattern, replacement in subs:
        text = re.sub(pattern, replacement, text, count=1)
    open(CONFIG_PATH, "w").write(text)
    print(f"[init] glmocr config written to {CONFIG_PATH}")


def _patch_page_loader():
    """Patch installed page_loader.py to handle data:application/pdf URIs."""
    import pathlib
    pkg = pathlib.Path("/usr/local/lib/python3.11/dist-packages/glmocr/dataloader/page_loader.py")
    if not pkg.exists():
        # try site-packages fallback
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

    assert OLD_LOAD in src, "patch target _load_source not found"
    assert OLD_ITER in src, "patch target _iter_source not found"
    patched = src.replace(OLD_LOAD, NEW_LOAD, 1).replace(OLD_ITER, NEW_ITER, 1)
    pkg.write_text(patched)
    print("[init] page_loader.py patched for data:application/pdf support")


def _start_vllm():
    global _vllm_proc
    env = os.environ.copy()
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
        "--disable-log-requests",
        "--speculative-config", MTP_JSON,
    ]
    _vllm_proc = subprocess.Popen(cmd, env=env)
    print(f"[init] vLLM started (pid {_vllm_proc.pid}), waiting for /health...")
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


def _start_glmocr():
    global _glmocr_proc
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    _glmocr_proc = subprocess.Popen(
        ["python", "-m", "glmocr.server", "--config", CONFIG_PATH],
        env=env,
    )
    print(f"[init] glmocr started (pid {_glmocr_proc.pid}), waiting for port {GLMOCR_PORT}...")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            requests.get(f"http://127.0.0.1:{GLMOCR_PORT}/", timeout=2)
            print("[init] glmocr ready")
            return
        except Exception:
            pass
        if _glmocr_proc.poll() is not None:
            raise RuntimeError(f"glmocr exited with code {_glmocr_proc.returncode}")
        time.sleep(2)
    raise RuntimeError("glmocr did not become ready within 60s")


# ── Worker initialisation (runs once per container) ───────────────────────────
_write_config()
_patch_page_loader()
_start_vllm()
_start_glmocr()
print("[init] Worker ready")


# ── Per-request handler ───────────────────────────────────────────────────────
def handler(job):
    job_input = job.get("input", {})
    try:
        r = requests.post(
            f"http://127.0.0.1:{GLMOCR_PORT}/glmocr/parse",
            json=job_input,
            timeout=300,
        )
        if not r.ok:
            return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
        return r.json()
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
