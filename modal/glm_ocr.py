"""Modal serverless deployment — GLM-OCR text recognition via vLLM with memory snapshot.

Memory snapshot strategy (mirrors Modal's Ministral-3 example):
  1. @modal.enter(snap=True): start vllm serve, warm up (compiles CUDA kernels),
     then POST /sleep to offload GPU weights → snapshot captures sleeping server.
  2. @modal.enter(snap=False): POST /wake_up to restore weights to GPU — fast path.
  VLLM_SERVER_DEV_MODE=1 enables the /sleep and /wake_up dev endpoints.
  TORCHINDUCTOR_COMPILE_THREADS=1 improves snapshot stability.

Architecture:
  GLMOCRWorker  GPU class — vllm serve subprocess with sleep/wake snapshot
  OCRFrontend   CPU class — PDF → first page JPEG → GLMOCRWorker.recognize

One-time setup:
  modal run modal/glm_ocr.py::download_weights [--hf-token <token>]

Deploy:
  modal deploy modal/glm_ocr.py

Test (CLI):
  modal run modal/glm_ocr.py --pdf-path /path/to/doc.pdf
"""
from __future__ import annotations

import base64
import io
import subprocess
import time

import modal

app = modal.App("glm-ocr")

MODEL_ID    = "zai-org/GLM-OCR"
SERVED_NAME = "glm-ocr"          # --served-model-name used in API calls
GPU         = "T4"             # 24 GB VRAM — GLM-OCR 9B in bfloat16 ≈ 18 GB
VLLM_PORT   = 8000

hf_vol   = modal.Volume.from_name("glm-ocr-hf-cache",   create_if_missing=True)
vllm_vol = modal.Volume.from_name("glm-ocr-vllm-cache",  create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_vol,
    "/root/.cache/vllm":        vllm_vol,
}

# ── Images ────────────────────────────────────────────────────────────────────

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .run_commands(
        "pip install --no-cache-dir uv",
        "uv pip install --system --no-cache 'vllm==0.21.0'",
        "uv pip install --system --no-cache 'transformers==5.3.0'",
        "uv pip install --system --no-cache "
        "'huggingface_hub[hf_transfer]' requests Pillow numpy pypdfium2 'fastapi[standard]'",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE":         "1",
        "VLLM_SERVER_DEV_MODE":            "1",   # enables /sleep and /wake_up endpoints
        "TORCHINDUCTOR_COMPILE_THREADS":   "1",   # snapshot stability
        "PYTORCH_CUDA_ALLOC_CONF":         "expandable_segments:True",
        "TORCH_NCCL_ENABLE_MONITORING":    "0",   # disable heartbeat monitor
        "TORCH_CPP_LOG_LEVEL":             "ERROR",  # suppress C++ WARNING-level TCPStore/NCCL broken-pipe logs
    })
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "pip install --no-cache-dir uv",
        "uv pip install --system --no-cache pypdfium2 Pillow 'fastapi[standard]'",
    )
)


# ── Weight downloader (run once) ──────────────────────────────────────────────

@app.function(image=vllm_image, volumes=VOLUMES, timeout=3600)
def download_weights(hf_token: str = ""):
    from huggingface_hub import snapshot_download
    kwargs = {"token": hf_token} if hf_token else {}
    print(f"Downloading {MODEL_ID} ...")
    snapshot_download(MODEL_ID, **kwargs)
    hf_vol.commit()
    print("Done.")


# ── Server lifecycle helpers ──────────────────────────────────────────────────

def _wait_ready(port: int, timeout: int = 300) -> None:
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")


def _warmup(port: int) -> None:
    """Compile Triton kernels for all task types before snapshot.

    Uses pixel budgets matching the pipeline's _TASK_PARAMS so the kernel shapes
    captured in the snapshot cover real workloads. max_tokens=10 triggers the
    MTP rejection_greedy_sample_kernel without wasting time on long outputs.
    """
    import requests
    from PIL import Image
    import numpy as np

    warmup_cases = [
        # (image_size,    min_pixels, max_pixels, max_tokens, prompt)
        # text crop — small image, tight pixel budget
        ((336, 336),  112_896,  512_000,  10, "Text Recognition:"),
        # table crop — larger image, full pixel budget, high max_tokens for OTSL→HTML
        ((672, 672),  112_896, 1_003_520, 10, "Table Recognition:"),
        # formula crop — small image
        ((336, 168),  112_896,  512_000,  10, "Formula Recognition:"),
    ]

    for i, (size, min_px, max_px, max_tok, prompt) in enumerate(warmup_cases):
        dummy = Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))
        buf   = io.BytesIO()
        dummy.save(buf, format="JPEG")
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        payload  = {
            "model": SERVED_NAME,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": data_url, "min_pixels": min_px, "max_pixels": max_px,
                }},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": max_tok,
            "temperature": 0.0,
        }
        r = requests.post(
            f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=120
        )
        print(f"[warmup {i+1}/{len(warmup_cases)}] {prompt}  status={r.status_code}")


# ── GLM-OCR vLLM worker (GPU) ─────────────────────────────────────────────────

@app.cls(
    gpu=GPU,
    image=vllm_image,
    volumes=VOLUMES,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=5,
    timeout=300,
    max_containers=2,
)
@modal.concurrent(max_inputs=16, target_inputs=8)
class GLMOCRWorker:
    @modal.enter(snap=True)
    def start(self) -> None:
        import requests

        cmd = [
            "vllm", "serve", MODEL_ID,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--enable-sleep-mode",
            "--gpu-memory-utilization", "0.5",
            "--max-model-len", "8192",
            "--max-num-seqs", "4",
            "--dtype", "bfloat16",
            "--served-model-name", "glm-ocr",
            "--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}',
        ]
        self._proc = subprocess.Popen(cmd)
        print("[glm-ocr] waiting for server ...")
        _wait_ready(VLLM_PORT)
        print("[glm-ocr] warming up (CUDA kernel compilation) ...")
        _warmup(VLLM_PORT)
        print("[glm-ocr] putting server to sleep for snapshot ...")
        requests.post(f"http://localhost:{VLLM_PORT}/sleep", timeout=120)
        print("[glm-ocr] sleeping — snapshot will be taken now")

    @modal.enter(snap=False)
    def wake(self) -> None:
        import requests
        print("[glm-ocr] waking server from snapshot ...")
        requests.post(f"http://localhost:{VLLM_PORT}/wake_up", timeout=120)
        _wait_ready(VLLM_PORT)
        print("[glm-ocr] server ready")

    @modal.exit()
    def stop(self) -> None:
        self._proc.terminate()

    @modal.method()
    def recognize(
        self,
        image_bytes: bytes,
        prompt:     str = "Text Recognition:",
        max_tokens: int = 2048,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> dict:
        import requests

        t0 = time.time()
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()

        image_url_obj: dict = {"url": data_url}
        if min_pixels is not None:
            image_url_obj["min_pixels"] = min_pixels
        if max_pixels is not None:
            image_url_obj["max_pixels"] = max_pixels

        resp = requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json={
                "model": SERVED_NAME,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": image_url_obj},
                    {"type": "text", "text": prompt},
                ]}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text   = resp.json()["choices"][0]["message"]["content"]
        exec_s = round(time.time() - t0, 3)
        print(f"[glm-ocr] recognized in {exec_s}s  prompt={prompt!r}  max_tokens={max_tokens}  chars={len(text)}")
        return {"text": text, "_start_ts": t0, "exec_s": exec_s}


# ── CPU frontend — PDF → first page → OCR ────────────────────────────────────

from pydantic import BaseModel as _BaseModel


class _OCRRequest(_BaseModel):
    file: str      # base64-encoded PDF
    page: int = 0  # 0-indexed page (default: first)
    dpi:  int = 200


@app.cls(
    image=cpu_image,
    timeout=300,
    scaledown_window=5,
    max_containers=4,   # 4 × 8 = 32 total slots — matches GLMOCRWorker (2 × 16 = 32)
    enable_memory_snapshot=True,
    min_containers=1,
)
@modal.concurrent(max_inputs=8, target_inputs=4)
class OCRFrontend:
    @modal.enter(snap=True)
    def load(self) -> None:
        import pypdfium2
        from PIL import Image
        _ = (pypdfium2, Image)  # pre-import C extensions — captured in snapshot
        print("[frontend] ready")

    @modal.fastapi_endpoint(method="GET")
    async def prime(self) -> dict:
        """Trigger GLMOCRWorker snapshot creation. Call once after deploy and wait for it to return."""
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=0).save(buf, format="JPEG")
        t_call  = time.time()
        result  = await GLMOCRWorker().recognize.remote.aio(buf.getvalue())
        wall_s  = round(time.time() - t_call, 3)
        queued_s = round(result["_start_ts"] - t_call, 3)
        return {"status": "ready", "wall_s": wall_s, "queued_s": queued_s,
                "exec_s": result["exec_s"], "chars": len(result["text"])}

    @modal.fastapi_endpoint(method="POST")
    async def process(self, req: _OCRRequest) -> dict:
        import pypdfium2 as pdfium

        t0 = time.time()

        raw_b64 = req.file
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            pdf_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            return {"error": f"Bad base64: {e}"}

        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
            n = len(pdf)
            if req.page >= n:
                return {"error": f"Page {req.page} out of range — document has {n} pages"}
            pg    = pdf[req.page]
            scale = req.dpi / 72
            pil   = pg.render(scale=scale).to_pil().convert("RGB")
            pg.close()
            pdf.close()
        except Exception as e:
            return {"error": f"PDF render failed: {e}"}

        t_render = time.time()

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        image_bytes = buf.getvalue()

        t_call = time.time()
        try:
            result = await GLMOCRWorker().recognize.remote.aio(image_bytes)
        except Exception as e:
            return {"error": f"OCR worker failed: {e}"}
        t_done = time.time()

        queued_s = round(result["_start_ts"] - t_call, 3)
        exec_s   = result["exec_s"]
        ocr_wall = round(t_done - t_call, 3)

        return {
            "text": result["text"],
            "meta": {
                "page":      req.page,
                "width_px":  pil.width,
                "height_px": pil.height,
                "dpi":       req.dpi,
                "timing": {
                    "render_s":      round(t_render - t0, 3),
                    "ocr_queued_s":  queued_s,
                    "ocr_exec_s":    exec_s,
                    "ocr_wall_s":    ocr_wall,
                    "total_s":       round(t_done - t0, 3),
                },
            },
        }


# ── CLI test ──────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(pdf_path: str = ""):
    if not pdf_path:
        print("Usage: modal run modal/glm_ocr.py --pdf-path /path/to/doc.pdf")
        return

    import pypdfium2 as pdfium

    with open(pdf_path, "rb") as f:
        raw = f.read()

    pdf = pdfium.PdfDocument(raw)
    pg  = pdf[0]
    pil = pg.render(scale=200 / 72).to_pil().convert("RGB")
    pg.close()
    pdf.close()

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    image_bytes = buf.getvalue()

    print(f"Page 0: {pil.width}×{pil.height}px  JPEG: {len(image_bytes) / 1024:.1f} KB")
    text = GLMOCRWorker().recognize.remote(image_bytes)
    print("=== OCR Result ===")
    print(text)
