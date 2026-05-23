"""GLM-OCR single-container pipeline — official glmocr SDK + Modal cold-start.

Replaces the hand-rolled layout/OCR code with the official glmocr[selfhosted]
pipeline while keeping all Modal snapshot/wake machinery.

  - glmocr handles layout (PP-DocLayoutV3 with model-predicted reading order),
    parallel OCR, repetition-penalty, correct pixel scaling, post-processing
  - vLLM serves GLM-OCR on GPU via localhost:8000
  - PP-DocLayoutV3 loads on CPU during snap=True, moves to GPU in wake()

Cold start from snapshot: ~6-8s

Deploy:
  modal deploy modal/glm_ocr_single.py

Test:
  python modal/test_glm_ocr_pipeline.py document.pdf \\
    --endpoint https://ringkasan-net--glm-ocr-single-documentocrworker-process.modal.run
"""
from __future__ import annotations

import base64
import io
import json as _json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import modal
from pydantic import BaseModel as _BaseModel

app = modal.App("glm-ocr-single")

GLM_MODEL_ID    = "zai-org/GLM-OCR"
LAYOUT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
SERVED_NAME     = "glm-ocr"
GPU             = "L4"
VLLM_PORT       = 8000

hf_vol   = modal.Volume.from_name("glm-ocr-hf-cache",   create_if_missing=True)
vllm_vol = modal.Volume.from_name("glm-ocr-vllm-cache",  create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_vol,
    "/root/.cache/vllm":        vllm_vol,
}

# ── Image ─────────────────────────────────────────────────────────────────────
# vllm brings torch; glmocr base brings pymupdf + pydantic + requests etc.
# selfhosted extras (opencv, sentencepiece, accelerate) added manually to avoid
# pip re-resolving torch after vllm has already pinned it.

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .run_commands(
        # System libs required by opencv-python-headless
        "apt-get update -qq && apt-get install -y --no-install-recommends "
        "libglib2.0-0 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*",
        "pip install --no-cache-dir uv",
        "uv pip install --system --no-cache 'vllm==0.21.0'",
        # Don't pin transformers: PPDocLayoutV3ImageProcessor requires a version
        # newer than 5.3.0; let vLLM's resolver pick the highest compatible version.
        "uv pip install --system --no-cache 'sentencepiece' 'accelerate'",
        "uv pip install --system --no-cache 'glmocr==0.1.5' 'opencv-python-headless'",
        "uv pip install --system --no-cache "
        "'huggingface_hub[hf_transfer]' 'fastapi[standard]'",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE":       "1",
        "VLLM_SERVER_DEV_MODE":          "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF":       "expandable_segments:True",
        "TORCH_CPP_LOG_LEVEL":           "ERROR",
        "TORCH_NCCL_ENABLE_MONITORING":  "0",
    })
    .add_local_file("warmup_doc.jpg", "/root/warmup_doc.jpg")
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

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
    raise TimeoutError(f"vLLM not ready after {timeout}s")


def _build_pipeline_cfg():
    """Build glmocr PipelineConfig for self-hosted mode pointing at local vLLM."""
    from glmocr.config import load_config
    cfg = load_config(
        mode="selfhosted",
        ocr_api_host="127.0.0.1",
        ocr_api_port=VLLM_PORT,
        layout_device="cpu",  # keep weights on CPU for snapshot; wake() moves to GPU
        api_key="",           # no auth for local vLLM
    )
    cfg.pipeline.ocr_api.model              = SERVED_NAME
    cfg.pipeline.ocr_api.connection_pool_size = 128
    cfg.pipeline.ocr_api.request_timeout    = 120
    cfg.pipeline.max_workers                = 16
    # max_tokens must be < max_model_len (8192) to leave room for input tokens.
    # glmocr's default is 8192, which leaves 0 tokens for the prompt and image.
    # 4096 output tokens covers all OCR cases; remaining 4096 carry image+text.
    cfg.pipeline.page_loader.max_tokens     = 4096
    # Cap visual tokens to the encoder cache budget (= max_model_len = 8192).
    # Official image_expect_length=6144 tokens @ 196 px/token → 1,204,224 px.
    # 8192 tokens @ 196 px/token = 1,605,632 px — we cap slightly below that.
    cfg.pipeline.page_loader.max_pixels     = 1_500_000
    return cfg.pipeline


# ── Request schema ─────────────────────────────────────────────────────────────

class _Request(_BaseModel):
    file:      str
    pages:     Optional[list[int]] = None
    num_pages: Optional[int]       = None
    dpi:       int                 = 200


# ── Single-container worker ────────────────────────────────────────────────────

@app.cls(
    gpu=GPU,
    image=image,
    volumes=VOLUMES,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=60,
    timeout=600,
    max_containers=10,
)
@modal.concurrent(max_inputs=2, target_inputs=1)
class DocumentOCRWorker:

    @modal.enter(snap=True)
    def start(self) -> None:
        import requests
        from PIL import Image
        from glmocr.layout import PPDocLayoutDetector, _raise_layout_import_error
        if PPDocLayoutDetector is None:
            _raise_layout_import_error()  # surfaces the real ImportError

        # ── 1. Load layout model weights on CPU ───────────────────────────────
        # CPU tensors go into the CPU snapshot; wake() only pays for .to("cuda").
        print("[single] loading PP-DocLayoutV3 on CPU ...")
        t0 = time.time()
        self._pipeline_cfg = _build_pipeline_cfg()
        self._layout_detector = PPDocLayoutDetector(self._pipeline_cfg.layout)
        self._layout_detector.start()   # layout_device="cpu" → stays on CPU
        print(f"[single] layout weights on CPU in {time.time()-t0:.2f}s")

        # ── 2. Start vLLM ─────────────────────────────────────────────────────
        import torch
        # bfloat16 requires compute capability >= 8.0 (Ampere+).
        # T4 is 7.5 → must use float16. L4/A100/H100 are 8.x → bfloat16.
        cc = torch.cuda.get_device_capability()
        dtype = "bfloat16" if cc[0] >= 8 else "half"
        print(f"[single] GPU compute capability {cc[0]}.{cc[1]} → dtype={dtype}")
        cmd = [
            "vllm", "serve", GLM_MODEL_ID,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--enable-sleep-mode",
            "--gpu-memory-utilization", "0.6",
            "--max-model-len",          "8192",
            "--max-num-seqs",            "32",
            "--max-num-batched-tokens", "32768",
            "--dtype",                  dtype,
            "--served-model-name",      SERVED_NAME,
            "--speculative-config",     '{"method": "mtp", "num_speculative_tokens": 3}',
        ]
        self._proc = subprocess.Popen(cmd)
        print("[single] waiting for vLLM ...")
        _wait_ready(VLLM_PORT)

        # ── 3. Sequential warmup (compile Triton ops across token-count range) ─
        # Pixel count → visual tokens (CogViT 14×14 patches, ~196 px/token):
        #   112 896 px → 576 tok  |  512 000 px → 2612 tok  |  1 003 520 px → 5120 tok
        print("[single] sequential warmup ...")
        session = requests.Session()
        warmup_cases = [
            ((336,  336),  112_896,   512_000, 128, "Text Recognition:"),
            ((1500, 200),  112_896,   512_000, 128, "Text Recognition:"),
            ((640,  800),  112_896,   512_000, 128, "Text Recognition:"),
            ((672,  450),  112_896, 1_003_520, 128, "Table Recognition:"),
            ((1500, 500),  112_896, 1_003_520, 128, "Table Recognition:"),
            ((1000, 1000), 112_896, 1_003_520, 128, "Table Recognition:"),
            ((336,  168),  112_896,   512_000, 128, "Formula Recognition:"),
            ((640,  800),  112_896,   512_000, 128, "Formula Recognition:"),
        ]
        n = len(warmup_cases)
        for i, (size, min_px, max_px, max_tok, prompt) in enumerate(warmup_cases):
            dummy = Image.new("RGB", size)
            buf = io.BytesIO()
            dummy.save(buf, format="JPEG")
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            r = session.post(
                f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                json={
                    "model": SERVED_NAME,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": data_url, "min_pixels": min_px, "max_pixels": max_px,
                        }},
                        {"type": "text", "text": prompt},
                    ]}],
                    "max_tokens": max_tok, "temperature": 0.0,
                },
                timeout=300,
            )
            print(f"[warmup {i+1}/{n}] {size} {prompt}  status={r.status_code}")

        # ── 4. Batch warmup (compile batch-N Triton kernels) ──────────────────
        self._run_batch_warmup(session, n=16)

        # ── 5. Move layout model to GPU before snapshot ────────────────────────
        # GPU snapshot captures PyTorch tensors left on GPU after vLLM sleep.
        # vLLM sleep frees its ~14 GB; the layout model's ~100 MB stays on GPU.
        # wake() then only needs a forward pass (~0.1s) instead of a CPU→GPU
        # transfer (~0.5s). Tradeoff: snapshot is ~100 MB larger to load.
        print("[single] moving layout model to GPU for snapshot ...")
        t_gpu = time.time()
        self._layout_detector._model  = self._layout_detector._model.to("cuda")
        self._layout_detector._device = "cuda"
        dummy_layout = Image.new("RGB", (224, 224))
        self._layout_detector.process([dummy_layout])
        print(f"[single] layout on GPU in {time.time()-t_gpu:.2f}s")

        # ── 6. Sleep for snapshot ──────────────────────────────────────────────
        print("[single] sleeping for snapshot ...")
        requests.post(f"http://localhost:{VLLM_PORT}/sleep", timeout=120)
        print("[single] snapshot ready")

    def _run_batch_warmup(self, session, n: int = 16) -> float:
        """Send n concurrent requests to compile batch-N Triton kernels.

        snap=True warmup is sequential (batch=1). Real inference submits up to 16
        simultaneous requests (batch=16). Triton compiles separate kernels per
        batch size, so this pre-compiles batch-16 before the first user request.
        """
        from PIL import Image

        warmup_cases = [
            ((336,  336),  112_896,   512_000, "Text Recognition:"),
            ((1500, 200),  112_896,   512_000, "Text Recognition:"),
            ((640,  800),  112_896,   512_000, "Text Recognition:"),
            ((672,  450),  112_896, 1_003_520, "Table Recognition:"),
            ((1500, 500),  112_896, 1_003_520, "Table Recognition:"),
            ((1000, 1000), 112_896, 1_003_520, "Table Recognition:"),
            ((336,  168),  112_896,   512_000, "Formula Recognition:"),
            ((640,  800),  112_896,   512_000, "Formula Recognition:"),
        ]
        source = Image.open("/root/warmup_doc.jpg").convert("RGB")

        def _send(idx: int) -> int:
            size, min_px, max_px, prompt = warmup_cases[idx % len(warmup_cases)]
            img = source.resize(size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            payload = {
                "model": SERVED_NAME,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": data_url, "min_pixels": min_px, "max_pixels": max_px,
                    }},
                    {"type": "text", "text": prompt},
                ]}],
                "max_tokens": 32, "temperature": 0.0,
            }
            # Retry 500s: vLLM's MMReceiverCache can return assertion errors
            # immediately after wake_up while its internal state settles.
            for attempt in range(4):
                resp = session.post(
                    f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                    json=payload, timeout=120,
                )
                if resp.status_code == 200 or resp.status_code < 500:
                    return resp.status_code
                time.sleep(1.0 * (attempt + 1))
            return resp.status_code

        print(f"[single] batch warmup ({n} concurrent) ...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as pool:
            statuses = list(pool.map(_send, range(n)))
        elapsed = round(time.time() - t0, 3)
        ok = sum(1 for s in statuses if s == 200)
        print(f"[single] batch warmup done  {ok}/{n} ok  {elapsed:.2f}s")
        return elapsed

    def _prime_mm_cache(self, session) -> float:
        """Send one sequential multimodal request after wake_up to prime the MM cache.

        GPU-snapshot kernels are already restored; we don't need to recompile them.
        The only thing needed is one round-trip through the MM pipeline so that
        vLLM's MMReceiverCache is initialised before concurrent production requests
        arrive — concurrent requests before priming trigger an assertion in
        vLLM 0.21.0 that aborts all in-flight connections.
        Failure is swallowed so wake() never fails on a warmup error.
        """
        from PIL import Image

        dummy = Image.new("RGB", (336, 336))
        buf = io.BytesIO()
        dummy.save(buf, format="JPEG")
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        t0 = time.time()
        print("[single] priming MM cache ...")
        try:
            r = session.post(
                f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                json={
                    "model": SERVED_NAME,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": data_url, "min_pixels": 112_896, "max_pixels": 512_000,
                        }},
                        {"type": "text", "text": "Text Recognition:"},
                    ]}],
                    "max_tokens": 8, "temperature": 0.0,
                },
                timeout=60,
            )
            status = r.status_code
        except Exception as e:
            print(f"[single] MM cache prime failed (non-fatal): {e}")
            status = -1
        elapsed = round(time.time() - t0, 3)
        print(f"[single] MM cache prime  status={status}  {elapsed:.2f}s")
        return elapsed

    @modal.enter(snap=False)
    def wake(self) -> None:
        import requests
        from requests.adapters import HTTPAdapter
        from glmocr.pipeline import Pipeline
        from PIL import Image

        t0 = time.time()

        # ── 1. Wake vLLM ──────────────────────────────────────────────────────
        requests.post(f"http://localhost:{VLLM_PORT}/wake_up", timeout=120)
        t_wakeup = time.time()   # POST /wake_up returned (weights restored)
        _wait_ready(VLLM_PORT)
        t_ready = time.time()    # /health 200 (server accepting requests)

        # ── 2. Warm up layout model (already on GPU from snapshot) ───────────────
        dummy = Image.new("RGB", (224, 224))
        self._layout_detector.process([dummy])
        t_layout = time.time()

        # ── 3. Sequential warmup — prime MM cache, stabilise after wake ──────────
        # vLLM 0.21.0 bug: after wake_up, sending concurrent multimodal requests
        # triggers an abort-all cascade in EngineCore (MMReceiverCache assertion).
        # Sequential requests avoid the race condition.  Batch-N Triton kernels
        # will compile on the first real user request (~4s, once per container).
        session = requests.Session()
        session.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=32))
        batch_warmup_s = self._prime_mm_cache(session)
        t_warmup = time.time()

        # ── 4. Build glmocr pipeline ───────────────────────────────────────────
        # Pass layout_detector directly so Pipeline reuses the GPU-loaded model.
        # OCRClient._session is None here; it self-initializes on first request.
        self._pipeline = Pipeline(self._pipeline_cfg, layout_detector=self._layout_detector)
        t_done = time.time()

        self._cold_start_timing = {
            "wakeup_s":        round(t_wakeup  - t0,        3),
            "health_s":        round(t_ready   - t_wakeup,  3),
            "layout_gpu_s":    round(t_layout  - t_ready,   3),
            "batch_warmup_s":  batch_warmup_s,
            "pipeline_init_s": round(t_done    - t_warmup,  3),
            "total_s":         round(t_done    - t0,        3),
        }
        print(
            f"[single] ready  wakeup={t_wakeup-t0:.2f}s "
            f"health={t_ready-t_wakeup:.2f}s "
            f"layout_gpu={t_layout-t_ready:.2f}s "
            f"batch_warmup={batch_warmup_s:.2f}s "
            f"total={t_done-t0:.2f}s"
        )

    @modal.exit()
    def stop(self) -> None:
        if hasattr(self, "_pipeline"):
            self._pipeline.stop()
        if hasattr(self, "_proc"):
            self._proc.terminate()

    @modal.fastapi_endpoint(method="POST")
    def process(self, req: _Request) -> dict:
        import fitz  # PyMuPDF — installed via glmocr base dep

        t0 = time.time()

        # ── Decode and open PDF ────────────────────────────────────────────────
        raw_b64 = req.file
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            pdf_bytes = base64.b64decode(raw_b64)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            return {"error": f"PDF decode failed: {e}"}

        n_pages      = doc.page_count
        page_indices = req.pages or (
            list(range(min(req.num_pages, n_pages))) if req.num_pages else list(range(n_pages))
        )
        page_indices = [i for i in page_indices if 0 <= i < n_pages]
        if not page_indices:
            doc.close()
            return {"error": "No valid pages"}

        # ── Render selected pages to JPEG bytes ───────────────────────────────
        # We pre-render here (rather than passing the full PDF to glmocr) so that
        # non-contiguous page selection works and DPI is respected.
        scale = req.dpi / 72.0
        page_jpeg_list: list[bytes] = []
        pages_info: list[dict] = []
        from PIL import Image as _PILImage
        for pi in page_indices:
            page = doc.load_page(pi)
            pix  = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pil  = _PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf  = io.BytesIO()
            pil.save(buf, format="JPEG", quality=92)
            page_jpeg_list.append(buf.getvalue())
            pages_info.append({"page": pi, "width_px": pil.width, "height_px": pil.height})
        doc.close()
        t_render = time.time()

        # ── Run glmocr pipeline ────────────────────────────────────────────────
        # Each image_bytes item is one unit → one PipelineResult per page.
        request_data = {
            "messages": [{"role": "user", "content": [
                {"type": "image_bytes", "data": b} for b in page_jpeg_list
            ]}]
        }
        results = list(self._pipeline.process(request_data, save_layout_visualization=False))
        t_pipeline = time.time()

        # ── Convert PipelineResults to our response format ────────────────────
        all_blocks: list[dict] = []
        markdown_parts: list[str] = []

        for pi, result in zip(page_indices, results):
            markdown_parts.append(result.markdown_result or "")

            raw = result.json_result
            # json_result is a JSON string of [[{index, label, content, bbox_2d, ...}]]
            page_regions = (_json.loads(raw)[0] if isinstance(raw, str) else raw[0]) if raw else []
            img_files = result.image_files or {}

            for region in page_regions:
                block: dict = {
                    "page":  pi,
                    "order": region.get("index", 0),
                    "label": region.get("native_label") or region.get("label", "text"),
                    "text":  region.get("content") or "",
                    "bbox":  region.get("bbox_2d"),
                }
                img_path = region.get("image_path")
                if img_path:
                    fname = img_path.split("/")[-1]
                    pil_img = img_files.get(fname)
                    if pil_img is not None:
                        buf2 = io.BytesIO()
                        pil_img.save(buf2, format="JPEG", quality=85)
                        block["image_b64"] = base64.b64encode(buf2.getvalue()).decode()
                        block["text"] = None
                all_blocks.append(block)

        t_assemble = time.time()

        ocr_regions  = sum(1 for b in all_blocks if not b.get("image_b64"))
        skip_regions = sum(1 for b in all_blocks if     b.get("image_b64"))

        timing = {
            "render_s":   round(t_render   - t0,          3),
            # glmocr pipeline covers layout + OCR + post-process in one call;
            # individual stage times are not exposed by the integrated pipeline.
            "ocr_wall_s": round(t_pipeline - t_render,    3),
            "assemble_s": round(t_assemble - t_pipeline,  3),
            "total_s":    round(t_assemble - t0,          3),
        }
        if hasattr(self, "_cold_start_timing"):
            timing["cold_start"] = self._cold_start_timing
            del self._cold_start_timing

        print(
            f"[single] pages={len(page_indices)} regions={len(all_blocks)} "
            f"ocr={ocr_regions} skip={skip_regions} "
            f"render={timing['render_s']:.2f}s "
            f"ocr_wall={timing['ocr_wall_s']:.2f}s "
            f"total={timing['total_s']:.2f}s"
        )

        return {
            "markdown": "\n\n".join(p for p in markdown_parts if p),
            "blocks":   all_blocks,
            "meta": {
                "pages":         page_indices,
                "pages_info":    pages_info,
                "total_regions": len(all_blocks),
                "ocr_regions":   ocr_regions,
                "skip_regions":  skip_regions,
                "timing":        timing,
            },
        }


# ── Weight downloader (run once) ──────────────────────────────────────────────

@app.function(image=image, volumes=VOLUMES, timeout=3600)
def download_weights(hf_token: str = ""):
    from huggingface_hub import snapshot_download
    kwargs = {"token": hf_token} if hf_token else {}
    print(f"Downloading {GLM_MODEL_ID} ...")
    snapshot_download(GLM_MODEL_ID, **kwargs)
    print(f"Downloading {LAYOUT_MODEL_ID} ...")
    snapshot_download(LAYOUT_MODEL_ID, **kwargs)
    hf_vol.commit()
    print("Done.")
