"""Modal serverless deployment — Unlimited-OCR via SGLang.

Architecture:
  UnlimitedOCRWorker  GPU class — SGLang server subprocess (H100, FA3 backend)
  OCRFrontend         CPU class — PDF → page JPEG → UnlimitedOCRWorker.recognize

Key differences from GLM-OCR:
  - SGLang instead of vLLM; no sleep/wake snapshot (SGLang lacks that mechanism)
  - Requires H100 — FA3 (FlashAttention 3) attention backend is Hopper-only
  - image_mode: "gundam" (high-quality, slower) or "base" (faster)
  - Custom SGLang wheel from the repo, kernels==0.11.7, Python 3.12, CUDA 12.6

One-time setup:
  modal run modal/unlimited_ocr.py::download_weights [--hf-token <token>]

Deploy:
  modal deploy modal/unlimited_ocr.py

Test (CLI):
  modal run modal/unlimited_ocr.py --pdf-path /path/to/doc.pdf
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import time

import modal

app = modal.App("unlimited-ocr")

MODEL_ID    = "baidu/Unlimited-OCR"
SERVED_NAME = "Unlimited-OCR"
GPU         = "H100"   # FA3 attention backend requires Hopper (H100)
SGLANG_PORT = 10000

# Custom SGLang dev build required by Unlimited-OCR
_SGLANG_WHEEL = (
    "https://github.com/baidu/Unlimited-OCR/raw/main/wheel/"
    "sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl"
)

hf_vol = modal.Volume.from_name("unlimited-ocr-hf-cache", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_vol,
}

# ── Images ────────────────────────────────────────────────────────────────────

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .run_commands(
        "pip install --no-cache-dir uv",
        # Custom SGLang wheel, then GPU kernel helpers
        f"uv pip install --system --no-cache '{_SGLANG_WHEEL}'",
        "uv pip install --system --no-cache 'kernels==0.11.7'",
        # PyTorch — version from repo; CUDA 12.6 index
        "uv pip install --system --no-cache "
        "'torch==2.10.0' 'torchvision==0.25.0' "
        "--extra-index-url https://download.pytorch.org/whl/cu126",
        # Model and serving deps
        "uv pip install --system --no-cache "
        "'transformers==4.57.1' einops addict easydict "
        "'pymupdf==1.27.2.2' psutil Pillow requests "
        "'huggingface_hub[hf_transfer]' 'fastapi[standard]'",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE":   "1",
        "PYTORCH_CUDA_ALLOC_CONF":   "expandable_segments:True",
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

@app.function(image=gpu_image, volumes=VOLUMES, timeout=3600)
def download_weights(hf_token: str = ""):
    from huggingface_hub import snapshot_download
    kwargs = {"token": hf_token} if hf_token else {}
    print(f"Downloading {MODEL_ID} ...")
    snapshot_download(MODEL_ID, **kwargs)
    hf_vol.commit()
    print("Done.")


# ── Server lifecycle helpers ──────────────────────────────────────────────────

def _wait_ready(port: int, timeout: int = 600) -> None:
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"SGLang server not ready after {timeout}s")


# ── Unlimited-OCR SGLang worker (GPU) ────────────────────────────────────────

@app.cls(
    gpu=GPU,
    image=gpu_image,
    volumes=VOLUMES,
    scaledown_window=30,
    timeout=600,
    max_containers=2,
)
@modal.concurrent(max_inputs=8, target_inputs=4)
class UnlimitedOCRWorker:
    @modal.enter()
    def start(self) -> None:
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model", MODEL_ID,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", str(SGLANG_PORT),
            "--attention-backend", "fa3",
            "--page-size", "1",
            "--mem-fraction-static", "0.8",
            "--context-length", "32768",
            "--enable-custom-logit-processor",
            "--disable-overlap-schedule",
            "--skip-server-warmup",
        ]
        self._proc = subprocess.Popen(cmd)
        print("[unlimited-ocr] waiting for SGLang server ...")
        _wait_ready(SGLANG_PORT)
        print("[unlimited-ocr] server ready")

    @modal.exit()
    def stop(self) -> None:
        self._proc.terminate()

    @modal.method()
    def recognize(
        self,
        image_bytes: bytes,
        prompt: str = "document parsing.",
        image_mode: str = "gundam",
    ) -> dict:
        import requests

        t0 = time.time()
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()

        payload = {
            "model": SERVED_NAME,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0,
            "skip_special_tokens": False,
            "stream": True,
            "images_config": {"image_mode": image_mode},
        }

        resp = requests.post(
            f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=1200,
        )
        resp.raise_for_status()

        chunks: list[str] = []
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj   = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks.append(delta)
            except Exception:
                pass

        text   = "".join(chunks)
        exec_s = round(time.time() - t0, 3)
        print(f"[unlimited-ocr] done in {exec_s}s  image_mode={image_mode!r}  chars={len(text)}")
        return {"text": text, "_start_ts": t0, "exec_s": exec_s}


# ── CPU frontend — PDF → page → OCR ──────────────────────────────────────────

from pydantic import BaseModel as _BaseModel


class _OCRRequest(_BaseModel):
    file:       str       # base64-encoded PDF
    page:       int = 0   # 0-indexed page (default: first)
    dpi:        int = 200
    image_mode: str = "gundam"   # "gundam" (high quality) or "base" (faster)


@app.cls(
    image=cpu_image,
    timeout=600,
    scaledown_window=30,
    max_containers=4,
    min_containers=1,
)
@modal.concurrent(max_inputs=8, target_inputs=4)
class OCRFrontend:
    @modal.enter()
    def load(self) -> None:
        import pypdfium2
        from PIL import Image
        _ = (pypdfium2, Image)
        print("[frontend] ready")

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
            n   = len(pdf)
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
            result = await UnlimitedOCRWorker().recognize.remote.aio(
                image_bytes, image_mode=req.image_mode
            )
        except Exception as e:
            return {"error": f"OCR worker failed: {e}"}
        t_done = time.time()

        queued_s = round(result["_start_ts"] - t_call, 3)
        exec_s   = result["exec_s"]
        ocr_wall = round(t_done - t_call, 3)

        return {
            "text": result["text"],
            "meta": {
                "page":       req.page,
                "width_px":   pil.width,
                "height_px":  pil.height,
                "dpi":        req.dpi,
                "image_mode": req.image_mode,
                "timing": {
                    "render_s":     round(t_render - t0, 3),
                    "ocr_queued_s": queued_s,
                    "ocr_exec_s":   exec_s,
                    "ocr_wall_s":   ocr_wall,
                    "total_s":      round(t_done - t0, 3),
                },
            },
        }


# ── CLI test ──────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(pdf_path: str = ""):
    if not pdf_path:
        print("Usage: modal run modal/unlimited_ocr.py --pdf-path /path/to/doc.pdf")
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
    result = UnlimitedOCRWorker().recognize.remote(image_bytes)
    print("=== OCR Result ===")
    print(result["text"])
