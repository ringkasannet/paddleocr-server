"""RunPod serverless handler — VLM (GLM-OCR) endpoint.

Startup (once per worker):
  1. Start vLLM subprocess with zai-org/GLM-OCR
  2. Wait for /health (up to 300s)

Per-request:
  - Accept a batch of pre-cropped region images from the layout endpoint
  - Submit all crops concurrently to vLLM's /v1/chat/completions
  - vLLM continuous batching handles N concurrent requests natively
  - Return content per region_id

Input:
  {"regions": [{"region_id": "p0_r7", "crop": "data:image/png;base64,...", "label": "table"}, ...]}

Output:
  {"results": [{"region_id": "p0_r7", "content": "| Col | ..."}], "meta": {...}}
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
import traceback

import aiohttp
import requests
import runpod

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL         = os.environ.get("MODEL", "zai-org/GLM-OCR")
VLLM_PORT     = int(os.environ.get("VLLM_PORT", "8000"))
GPU_UTIL      = os.environ.get("GPU_MEM_UTIL", "0.93")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
MAX_TOKENS    = int(os.environ.get("MAX_TOKENS", "2048"))
MAX_NUM_SEQS  = int(os.environ.get("MAX_NUM_SEQS", "64"))
HF_TOKEN      = os.environ.get("HF_TOKEN", "")

VLLM_URL  = f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions"
MTP_JSON  = '{"method":"mtp","num_speculative_tokens":3}'

# Exact prompts from glmocr config.yaml task_prompt_mapping
_PROMPTS = {
    "text":    "Text Recognition:",
    "table":   "Table Recognition:",
    "formula": "Formula Recognition:",
}

# ── vLLM process management ────────────────────────────────────────────────────

_vllm_proc: subprocess.Popen | None = None


def _start_vllm() -> None:
    global _vllm_proc
    env = os.environ.copy()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  MODEL,
        "--served-model-name",      "glm-ocr",
        "--port",                   str(VLLM_PORT),
        "--gpu-memory-utilization", GPU_UTIL,
        "--max-model-len",          str(MAX_MODEL_LEN),
        "--tensor-parallel-size",   "1",
        "--max-num-seqs",           str(MAX_NUM_SEQS),
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--speculative-config",     MTP_JSON,
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


_start_vllm()
print("[init] Worker ready")


# ── Inference helpers ──────────────────────────────────────────────────────────

def _build_payload(crop_data_uri: str, label: str) -> dict:
    """Build OpenAI vision chat payload matching glmocr's exact request format."""
    prompt = _PROMPTS.get(label, "")
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": crop_data_uri}},
    ]
    if prompt:
        content.append({"type": "text", "text": prompt})

    return {
        "model":             "glm-ocr",
        "messages":          [{"role": "user", "content": content}],
        "max_tokens":        MAX_TOKENS,
        "temperature":       0.0,
        "top_p":             0.00001,
        "top_k":             1,
        "repetition_penalty": 1.1,
    }


async def _infer_one(
    session: aiohttp.ClientSession,
    region_id: str,
    crop: str,
    label: str,
) -> tuple[str, str | None, str | None]:
    """POST one crop to vLLM. Returns (region_id, content_or_None, error_or_None)."""
    payload = _build_payload(crop, label)
    try:
        async with session.post(VLLM_URL, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return region_id, None, f"HTTP {resp.status}: {body[:200]}"
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            return region_id, (content or "").strip(), None
    except Exception as e:
        return region_id, None, str(e)


async def _infer_batch(regions: list[dict]) -> list[dict]:
    """Submit all regions concurrently to vLLM."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _infer_one(session, r["region_id"], r["crop"], r.get("label", "text"))
            for r in regions
        ]
        raw = await asyncio.gather(*tasks)

    results = []
    errors = []
    for region_id, content, error in raw:
        if error:
            errors.append({"region_id": region_id, "error": error})
            results.append({"region_id": region_id, "content": None})
        else:
            results.append({"region_id": region_id, "content": content})

    return results, errors


# ── Per-request handler ────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    try:
        job_input = job.get("input", {})
        regions = job_input.get("regions", [])
        if not regions:
            return {"error": "input.regions is required"}

        results, errors = asyncio.run(_infer_batch(regions))

        return {
            "results": results,
            "meta": {
                "regions_received":  len(regions),
                "regions_processed": len(results),
                "errors":            errors,
            },
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
