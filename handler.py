"""
RunPod Serverless handler for PaddleOCR-VL.

Initialization (once per worker):
  - Starts vLLM genai server on port 8118
  - Starts PaddleX pipeline server on port 8080
  - Waits until both are healthy before accepting jobs

Handler (once per job):
  - Accepts image as base64 string or URL
  - Calls PaddleX /ocr-doc-parser endpoint
  - Returns structured OCR results
"""

import os
import subprocess
import time
import signal
import sys
import requests
import runpod

# ── Configuration from env vars ───────────────────────────────
GPU_MEM_UTIL   = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
MODEL_NAME     = os.environ.get("MODEL_NAME", "PaddleOCR-VL-1.5-0.9B")
VLLM_PORT      = int(os.environ.get("VLLM_PORT", "8118"))
PADDLE_PORT    = int(os.environ.get("PADDLE_PORT", "8080"))
PIPELINE_CFG   = os.environ.get("PIPELINE_CONFIG", "/workspace/PaddleOCR-VL.yaml")
INIT_TIMEOUT   = int(os.environ.get("RUNPOD_INIT_TIMEOUT", "600"))

# HuggingFace model cache — RunPod model caching stores here
# Set HF_HOME to /runpod-volume/huggingface-cache for RunPod model caching to work
os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface-cache")
os.environ["PYTHONUNBUFFERED"] = "1"

_vllm_proc   = None
_paddle_proc = None


def _start_vllm():
    """Start the vLLM genai server as a subprocess."""
    # Write backend config to temp file
    backend_cfg = f"/tmp/vllm_backend.yaml"
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


def _start_paddlex():
    """Start the PaddleX pipeline server as a subprocess."""
    cmd = [
        "/workspace/.paddleocr/bin/paddlex",
        "--serve",
        "--pipeline", PIPELINE_CFG,
        "--host",     "0.0.0.0",
        "--port",     str(PADDLE_PORT),
    ]
    print(f"[init] Starting PaddleX: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def _wait_healthy(url: str, proc: subprocess.Popen, label: str, timeout: int):
    """Poll URL until 200 OK or timeout."""
    deadline = time.time() + timeout
    interval = 10
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"[init] {label} process exited unexpectedly (code {proc.returncode})")
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"[init] {label} is healthy.")
                return
        except requests.RequestException:
            pass
        elapsed = int(time.time() - (deadline - timeout))
        print(f"[init] Waiting for {label}... ({elapsed}s)")
        time.sleep(interval)
    raise TimeoutError(f"[init] {label} did not become healthy within {timeout}s")


def initialize():
    """Run once at worker startup before any jobs are accepted."""
    global _vllm_proc, _paddle_proc

    print(f"[init] Worker starting | model={MODEL_NAME} | gpu_mem={GPU_MEM_UTIL}")

    _vllm_proc = _start_vllm()
    _wait_healthy(f"http://localhost:{VLLM_PORT}/health", _vllm_proc, "vLLM", INIT_TIMEOUT)

    _paddle_proc = _start_paddlex()
    _wait_healthy(f"http://localhost:{PADDLE_PORT}/health", _paddle_proc, "PaddleX", 120)

    print("[init] Both services healthy — ready for jobs.")


def handler(job):
    """
    Called by RunPod for each job.

    Expected input:
      {
        "image": "<base64-encoded image OR public URL>",
        "use_layout_detection": true,    # optional
        "use_doc_preprocessor": false    # optional
      }

    Returns PaddleX OCR output as JSON.
    """
    job_input = job.get("input", {})

    image = job_input.get("image")
    if not image:
        return {"error": "Missing required field: 'image'"}

    payload = {"image": image}

    # Pass through optional pipeline flags if provided
    for flag in ("use_layout_detection", "use_doc_preprocessor",
                 "use_chart_recognition", "use_doc_orientation_classify"):
        if flag in job_input:
            payload[flag] = job_input[flag]

    try:
        resp = requests.post(
            f"http://localhost:{PADDLE_PORT}/ocr-doc-parser",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"PaddleX HTTP error: {e}", "status_code": resp.status_code}
    except requests.exceptions.Timeout:
        return {"error": "PaddleX request timed out after 300s"}
    except Exception as e:
        return {"error": str(e)}


def _cleanup(signum, frame):
    """Gracefully stop subprocesses on container shutdown."""
    for proc, name in [(_paddle_proc, "PaddleX"), (_vllm_proc, "vLLM")]:
        if proc and proc.poll() is None:
            print(f"[shutdown] Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    sys.exit(0)


signal.signal(signal.SIGTERM, _cleanup)
signal.signal(signal.SIGINT, _cleanup)

# ── Run initialization then start the RunPod job loop ─────────
initialize()

runpod.serverless.start({"handler": handler})
