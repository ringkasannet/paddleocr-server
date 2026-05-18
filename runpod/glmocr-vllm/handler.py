"""RunPod serverless handler — GLM-OCR text extraction via vLLM.

Designed to work with the layout endpoint output. Receives pre-cropped region
images (already attached to non-text regions by the layout endpoint) and runs
GLM-OCR inference on each crop.

Input:
  {
    "regions": [
      {
        "page_num": 0,
        "bbox": [x0, y0, x1, y1],
        "type": "table",
        "order": 5,
        "image": "<base64_png_crop>"   ← from layout endpoint output
      }
    ]
  }

Output:
  {
    "regions": [
      {"page_num": 0, "bbox": [...], "type": "table", "order": 5, "text": "..."}
    ]
  }
"""

import base64
import io
import os
import subprocess
import time

import runpod
from openai import OpenAI

MODEL         = os.environ.get("MODEL", "zai-org/GLM-OCR")
VLLM_PORT     = int(os.environ.get("VLLM_PORT", "8000"))
GPU_UTIL      = os.environ.get("GPU_MEM_UTIL", "0.90")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
MAX_TOKENS    = int(os.environ.get("MAX_TOKENS", "2048"))
ENABLE_MTP    = os.environ.get("ENABLE_MTP", "0") == "1"

# Task-specific prompts matching glmocr's task_prompt_mapping.
# GLM-OCR is trained on these exact strings — do not change them.
TASK_PROMPTS = {
    "table":           "Table Recognition:",
    "figure":          "Figure Understanding:",
    "chart":           "Figure Understanding:",
    "formula":         "Formula Recognition:",
    "display_formula": "Formula Recognition:",
    "inline_formula":  "Formula Recognition:",
    "seal":            "Text Recognition:",
    "vertical_text":   "Text Recognition:",
    "algorithm":       "Text Recognition:",
}
DEFAULT_PROMPT = "Text Recognition:"

_vllm_proc = None


# ── vLLM startup (once per worker) ───────────────────────────────────────────
def _start_vllm():
    global _vllm_proc
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--served-model-name", "glm-ocr",
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", GPU_UTIL,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", "8",
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if ENABLE_MTP:
        cmd += [
            "--speculative-config.method", "mtp",
            "--speculative-config.num_speculative_tokens", "1",
        ]
    _vllm_proc = subprocess.Popen(cmd)
    print(f"[init] vLLM started (pid {_vllm_proc.pid}), waiting for /health …")

    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            if requests.get(f"{VLLM_URL}/health", timeout=2).ok:
                print("[init] vLLM ready")
                return
        except Exception:
            pass
        if _vllm_proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early (code {_vllm_proc.returncode})")
        time.sleep(3)
    raise RuntimeError("vLLM did not become healthy within 300s")


_start_vllm()
_client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{VLLM_PORT}/v1", timeout=3600)
print("[init] Worker ready")


# ── VLM inference ─────────────────────────────────────────────────────────────
def _extract_text(crop_b64: str, region_type: str) -> str:
    prompt = TASK_PROMPTS.get(region_type, DEFAULT_PROMPT)
    response = _client.chat.completions.create(
        model="glm-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ── Per-request handler ───────────────────────────────────────────────────────
def handler(job):
    job_input = job.get("input", {})
    regions = job_input.get("regions", [])
    if not regions:
        return {"regions": []}

    results = []
    for region in regions:
        crop_b64 = region.get("image", "")
        if not crop_b64:
            results.append({**region, "text": "[no image]"})
            continue
        try:
            text = _extract_text(crop_b64, region.get("type", ""))
        except Exception as e:
            text = f"[error: {e}]"

        out = {k: v for k, v in region.items() if k != "image"}
        out["text"] = text
        results.append(out)

    return {"regions": results}


runpod.serverless.start({"handler": handler})
