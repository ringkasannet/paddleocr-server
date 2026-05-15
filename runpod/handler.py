"""
RunPod Serverless handler for PaddleOCR-VL.

Initialization (once per worker):
  - Starts vLLM genai server on port 8118
  - Waits until vLLM is healthy
  - Initializes PaddleOCRVL Python pipeline (connects to vLLM)

Handler (once per job):
  - Accepts image as base64 string or URL
  - Calls PaddleOCRVL.predict() directly (no HTTP server needed)
  - Returns structured OCR results as JSON
"""

import os
import base64
import subprocess
import tempfile
import time
import signal
import sys
import requests
import runpod

# ── Configuration from env vars ───────────────────────────────
GPU_MEM_UTIL = os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")
MODEL_NAME   = os.environ.get("MODEL_NAME", "PaddleOCR-VL-1.5-0.9B")
VLLM_PORT    = int(os.environ.get("VLLM_PORT", "8118"))
INIT_TIMEOUT = int(os.environ.get("RUNPOD_INIT_TIMEOUT", "600"))

os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface-cache")
os.environ["PYTHONUNBUFFERED"] = "1"

_vllm_proc = None
_pipeline  = None


def _start_vllm():
    backend_cfg = "/tmp/vllm_backend.yaml"
    with open(backend_cfg, "w") as f:
        f.write(f"gpu-memory-utilization: {GPU_MEM_UTIL}\n")

    cmd = [
        "paddleocr", "genai_server",
        "--model_name", MODEL_NAME,
        "--backend",    "vllm",
        "--host",       "0.0.0.0",
        "--port",       str(VLLM_PORT),
        "--backend_config", backend_cfg,
    ]
    print(f"[init] Starting vLLM: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def _wait_healthy(url, proc, label, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"[init] {label} exited unexpectedly (code {proc.returncode})")
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print(f"[init] {label} is healthy.")
                return
        except requests.RequestException:
            pass
        elapsed = int(time.time() - (deadline - timeout))
        print(f"[init] Waiting for {label}... ({elapsed}s)")
        time.sleep(10)
    raise TimeoutError(f"[init] {label} did not become healthy within {timeout}s")


def initialize():
    global _vllm_proc, _pipeline

    print(f"[init] Worker starting | model={MODEL_NAME} | gpu_mem={GPU_MEM_UTIL}")

    # Start vLLM and wait for it
    _vllm_proc = _start_vllm()
    _wait_healthy(f"http://localhost:{VLLM_PORT}/health", _vllm_proc, "vLLM", INIT_TIMEOUT)

    # Initialize PaddleOCRVL Python pipeline pointing at local vLLM
    print("[init] Initializing PaddleOCRVL pipeline (loading layout models)...")
    from paddleocr import PaddleOCRVL
    _pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=f"http://localhost:{VLLM_PORT}/v1",
    )
    print("[init] PaddleOCRVL pipeline ready — accepting jobs.")


def handler(job):
    """
    Input:
      {
        "image": "<base64 string OR public URL>",
        "output_format": "json" | "markdown"   (optional, default: json)
      }

    Returns structured OCR result.
    """
    job_input = job.get("input", {})
    image = job_input.get("image")
    if not image:
        return {"error": "Missing required field: 'image'"}

    output_format = job_input.get("output_format", "json")

    # Resolve input: URL passes through, base64 gets written to a temp file
    tmp_path = None
    if image.startswith("http://") or image.startswith("https://"):
        input_path = image
    else:
        try:
            data = base64.b64decode(image)
        except Exception as e:
            return {"error": f"Invalid base64 input: {e}"}
        suffix = ".pdf" if data[:4] == b"%PDF" else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(data)
            tmp_path = f.name
        input_path = tmp_path

    try:
        results = []
        for res in _pipeline.predict(input_path):
            if output_format == "markdown":
                results.append({
                    "page_index": res.json.get("page_index"),
                    "markdown": res.markdown.get("markdown_texts", ""),
                })
            else:
                results.append(res.json)
        return results if len(results) > 1 else results[0]
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path:
            os.unlink(tmp_path)


def _cleanup(signum, frame):
    if _vllm_proc and _vllm_proc.poll() is None:
        print("[shutdown] Stopping vLLM...")
        _vllm_proc.terminate()
        try:
            _vllm_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _vllm_proc.kill()
    sys.exit(0)


signal.signal(signal.SIGTERM, _cleanup)
signal.signal(signal.SIGINT, _cleanup)

initialize()
runpod.serverless.start({"handler": handler})
