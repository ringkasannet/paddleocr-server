"""GLM-OCR single-container pipeline — layout + OCR in one GPU container.

Everything runs in one L4 container with zero cross-container network hops:
  - PP-DocLayoutV3 runs on CPU (in-process, ~1s per page)
  - vLLM runs on GPU with max_num_seqs=16 so all region crops fit in one batch (~5s)
  - All OCR calls go to localhost:8000 (loopback, no serialization overhead)

Expected latency (warm container):
  render    ~0.1s
  layout    ~1s     (CPU, in-process)
  OCR       ~5s     (16 seqs, 1-2 batches via loopback HTTP)
  total     ~6-7s per page

Cold start from snapshot: ~6-7s (vLLM weight restore; layout model already in RAM)

Deploy:
  modal deploy modal/glm_ocr_single.py

Test (reuses pipeline test script with different endpoint):
  python modal/test_glm_ocr_pipeline.py document.pdf \\
    --endpoint https://ringkasan-net--glm-ocr-single-documentocrworker-process.modal.run
"""
from __future__ import annotations

import base64
import io
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import modal
from pydantic import BaseModel as _BaseModel

app = modal.App("glm-ocr-single")

GLM_MODEL_ID  = "zai-org/GLM-OCR"
LAYOUT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
SERVED_NAME   = "glm-ocr"
GPU           = "L4"
VLLM_PORT     = 8000

hf_vol   = modal.Volume.from_name("glm-ocr-hf-cache",   create_if_missing=True)
vllm_vol = modal.Volume.from_name("glm-ocr-vllm-cache",  create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_vol,
    "/root/.cache/vllm":        vllm_vol,
}

# ── Image ─────────────────────────────────────────────────────────────────────
# Same base as glm_ocr.py — transformers already includes AutoModelForObjectDetection
# which is all PP-DocLayoutV3 needs.

image = (
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
        "HF_XET_HIGH_PERFORMANCE":       "1",
        "VLLM_SERVER_DEV_MODE":          "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF":       "expandable_segments:True",
        "TORCH_CPP_LOG_LEVEL":           "ERROR",
        "TORCH_NCCL_ENABLE_MONITORING":  "0",
    })
)


# ── Label routing ─────────────────────────────────────────────────────────────

_TASK: dict[str, str] = {
    "text":              "text",
    "title":             "text",
    "paragraph_title":   "text",
    "abstract":          "text",
    "content":           "text",
    "doc_title":         "text",
    "reference_content": "text",
    "vertical_text":     "text",
    "vision_footnote":   "text",
    "seal":              "text",
    "algorithm":         "text",
    "table":             "table",
    "display_formula":   "formula",
    "inline_formula":    "formula",
    "image":             "skip",
    "figure":            "skip",
    "chart":             "skip",
    "figure_title":      "skip",
    "table_title":       "skip",
    "chart_title":       "skip",
    "header":            "abandon",
    "footer":            "abandon",
    "number":            "abandon",
    "footnote":          "abandon",
    "aside_text":        "abandon",
    "reference":         "abandon",
    "footer_image":      "abandon",
    "header_image":      "abandon",
}

_PROMPT: dict[str, str] = {
    "text":    "Text Recognition:",
    "table":   "Table Recognition:",
    "formula": "Formula Recognition:",
}

_TASK_PARAMS: dict[str, dict] = {
    "text":    {"min_pixels": 112_896, "max_pixels":   512_000, "max_tokens": 4096},
    "table":   {"min_pixels": 112_896, "max_pixels": 1_003_520, "max_tokens": 4096},
    "formula": {"min_pixels": 112_896, "max_pixels":   512_000, "max_tokens": 4096},
}


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


def _nms(regions: list, iou_thresh: float = 0.65) -> list:
    if len(regions) <= 1:
        return regions
    by_score = sorted(regions, key=lambda r: r["score"], reverse=True)
    kept: list = []
    for cand in by_score:
        cx0, cy0, cx1, cy1 = cand["bbox"]
        ca = max(0, cx1 - cx0) * max(0, cy1 - cy0)
        if ca == 0:
            continue
        drop = False
        for k in kept:
            kx0, ky0, kx1, ky1 = k["bbox"]
            ka = max(0, kx1 - kx0) * max(0, ky1 - ky0)
            ix0, iy0 = max(cx0, kx0), max(cy0, ky0)
            ix1, iy1 = min(cx1, kx1), min(cy1, ky1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            if (ix1 - ix0) * (iy1 - iy0) / min(ca, ka) > iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(cand)
    return kept


def _reading_order(regions: list, page_width: int) -> list:
    if len(regions) <= 1:
        return list(regions)
    mid = page_width / 2
    left  = sorted([r for r in regions if r["bbox"][2] <= mid * 1.1], key=lambda r: r["bbox"][1])
    right = sorted([r for r in regions if r["bbox"][0] >= mid * 0.9], key=lambda r: r["bbox"][1])
    full  = sorted([r for r in regions
                    if r["bbox"][2] > mid * 1.1 and r["bbox"][0] < mid * 0.9],
                   key=lambda r: r["bbox"][1])
    result: list = []
    li = ri = fi = 0
    while li < len(left) or ri < len(right) or fi < len(full):
        tops = []
        if li < len(left):  tops.append(("l", left[li]["bbox"][1]))
        if ri < len(right): tops.append(("r", right[ri]["bbox"][1]))
        if fi < len(full):  tops.append(("f", full[fi]["bbox"][1]))
        nxt = min(tops, key=lambda x: x[1])[0]
        if nxt == "l":   result.append(left[li]);  li += 1
        elif nxt == "r": result.append(right[ri]); ri += 1
        else:            result.append(full[fi]);  fi += 1
    return result


def _build_markdown(blocks: list) -> str:
    parts: list[str] = []
    img_counter = 0
    for b in blocks:
        label = b.get("label", "text")
        text  = (b.get("text") or "").strip()
        if label == "doc_title":
            parts.append(f"# {text}")
        elif label in ("title", "paragraph_title"):
            parts.append(f"## {text}")
        elif label == "abstract":
            parts.append(f"> {text}")
        elif label == "table":
            parts.append(text)
        elif label == "display_formula":
            parts.append(f"$$\n{text}\n$$")
        elif label == "inline_formula":
            parts.append(f"${text}$")
        elif label in ("image", "figure", "chart"):
            ref = f"p{b['page']}_r{b['order']}_{label}_{img_counter}.jpg"
            parts.append(f"![{label}]({ref})")
            img_counter += 1
        elif text:
            parts.append(text)
    return "\n\n".join(parts)


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
    max_containers=2,
)
@modal.concurrent(max_inputs=2, target_inputs=1)
class DocumentOCRWorker:

    @modal.enter(snap=True)
    def start(self) -> None:
        import requests
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        # ── 1. Pre-load layout model weights on CPU ───────────────────────────
        # CPU tensors go into the CPU snapshot. wake() only pays for .to("cuda").
        print("[single] pre-loading PP-DocLayoutV3 weights on CPU ...")
        t_lstart = time.time()
        self._layout_processor = AutoImageProcessor.from_pretrained(LAYOUT_MODEL_ID)
        self._layout_model = AutoModelForObjectDetection.from_pretrained(LAYOUT_MODEL_ID).eval()
        print(f"[single] layout weights ready on CPU in {time.time()-t_lstart:.2f}s")

        # ── 2. Start vLLM on GPU ─────────────────────────────────────────────
        cmd = [
            "vllm", "serve", GLM_MODEL_ID,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--enable-sleep-mode",
            "--gpu-memory-utilization", "0.6",
            "--max-model-len",          "8192",
            "--max-num-seqs",            "16",
            "--max-num-batched-tokens", "8192",
            "--dtype",                  "bfloat16",
            "--served-model-name",      SERVED_NAME,
            "--speculative-config",     '{"method": "mtp", "num_speculative_tokens": 3}',
        ]
        self._proc = subprocess.Popen(cmd)
        print("[single] waiting for vLLM ...")
        _wait_ready(VLLM_PORT)

        # ── 3. Exhaustive vLLM warmup ─────────────────────────────────────────
        # Covers the full range of visual token counts produced by real document
        # crops so rotary_kernel and other Triton ops are compiled before snapshot.
        # Image pixel count → visual tokens (CogViT 14×14 patches, 196 px/token):
        #   112 896 px →  576 tok  |  302 400 px → 1543 tok  |  512 000 px → 2612 tok
        #   750 000 px → 3826 tok  |  1 003 520 px → 5120 tok
        print("[single] warming up vLLM ...")
        session = requests.Session()
        warmup_cases = [
            # text (max_pixels=512 000) — 576, ~1530, ~2612 visual tokens
            ((336,  336),  112_896,   512_000, 128, "Text Recognition:"),
            ((1500, 200),  112_896,   512_000, 128, "Text Recognition:"),
            ((640,  800),  112_896,   512_000, 128, "Text Recognition:"),
            # table (max_pixels=1 003 520) — ~1543, ~3826, ~5120 visual tokens
            ((672,  450),  112_896, 1_003_520, 128, "Table Recognition:"),
            ((1500, 500),  112_896, 1_003_520, 128, "Table Recognition:"),
            ((1000, 1000), 112_896, 1_003_520, 128, "Table Recognition:"),
            # formula — ~576, ~2612 visual tokens
            ((336,  168),  112_896,   512_000, 128, "Formula Recognition:"),
            ((640,  800),  112_896,   512_000, 128, "Formula Recognition:"),
        ]
        n = len(warmup_cases)
        for i, (size, min_px, max_px, max_tok, prompt) in enumerate(warmup_cases):
            dummy_img = Image.new("RGB", size)
            buf = io.BytesIO()
            dummy_img.save(buf, format="JPEG")
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
                    "max_tokens": max_tok,
                    "temperature": 0.0,
                },
                timeout=300,
            )
            print(f"[warmup {i+1}/{n}] {size} {prompt}  status={r.status_code}")

        # ── 4. Sleep for snapshot ─────────────────────────────────────────────
        print("[single] sleeping for snapshot ...")
        requests.post(f"http://localhost:{VLLM_PORT}/sleep", timeout=120)
        print("[single] snapshot will be taken now")

    def _activate_layout_gpu(self) -> None:
        """Move layout model (CPU weights from snapshot) to GPU and run warmup inference."""
        import torch
        from PIL import Image
        t0 = time.time()
        print("[single] activating layout model on GPU ...")
        self._layout_model = self._layout_model.to("cuda").eval()
        t_cuda = time.time()
        dummy = Image.new("RGB", (224, 224))
        inputs = self._layout_processor(images=[dummy], return_tensors="pt").to("cuda")
        with torch.no_grad():
            self._layout_model(**inputs)
        t_warmup = time.time()
        self._layout_load_detail = {
            "model_cuda_s": round(t_cuda   - t0,     3),
            "warmup_s":     round(t_warmup - t_cuda, 3),
            "total_s":      round(t_warmup - t0,     3),
        }
        print(f"[single] layout GPU ready  cuda={t_cuda-t0:.2f}s warmup={t_warmup-t_cuda:.2f}s")

    @modal.enter(snap=False)
    def wake(self) -> None:
        import requests
        t0 = time.time()
        print("[single] waking vLLM from snapshot ...")
        requests.post(f"http://localhost:{VLLM_PORT}/wake_up", timeout=120)
        t_wakeup = time.time()
        _wait_ready(VLLM_PORT)
        t_ready = time.time()
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=32)
        self._session.mount("http://", adapter)
        self._activate_layout_gpu()
        t_done = time.time()
        self._cold_start_timing = {
            "wakeup_s":      round(t_wakeup - t0,       3),
            "health_s":      round(t_ready  - t_wakeup, 3),
            "layout_gpu_s":  round(t_done   - t_ready,  3),
            "layout_detail": getattr(self, "_layout_load_detail", None),
            "total_s":       round(t_done   - t0,       3),
        }
        print(f"[single] ready  wakeup={t_wakeup-t0:.2f}s  health={t_ready-t_wakeup:.2f}s  layout_gpu={t_done-t_ready:.2f}s  total={t_done-t0:.2f}s")

    @modal.exit()
    def stop(self) -> None:
        if hasattr(self, "_proc"):
            self._proc.terminate()

    # ── Layout detection (GPU, in-process) ────────────────────────────────────

    def _detect_layout(self, pil_images: list) -> list:
        import torch

        THRESHOLD         = 0.3
        HEADING_THRESHOLD = 0.2
        HEADING_LABELS    = {"paragraph_title", "doc_title"}
        BATCH_SIZE        = 4

        all_results = []
        for chunk_start in range(0, len(pil_images), BATCH_SIZE):
            chunk  = pil_images[chunk_start : chunk_start + BATCH_SIZE]
            inputs = self._layout_processor(images=chunk, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self._layout_model(**inputs)
            detections = self._layout_processor.post_process_object_detection(
                outputs,
                threshold=min(THRESHOLD, HEADING_THRESHOLD),
                target_sizes=[img.size[::-1] for img in chunk],
            )
            for pil_img, det in zip(chunk, detections):
                page_detections = []
                for score, label_id, box in zip(det["scores"], det["labels"], det["boxes"]):
                    label  = self._layout_model.config.id2label[label_id.item()]
                    cutoff = HEADING_THRESHOLD if label in HEADING_LABELS else THRESHOLD
                    if score.item() < cutoff:
                        continue
                    x0, y0, x1, y1 = box.tolist()
                    page_detections.append({
                        "type":  label,
                        "bbox":  [int(x0), int(y0), int(x1), int(y1)],
                        "score": round(score.item(), 4),
                    })
                all_results.append({
                    "width_px":  pil_img.width,
                    "height_px": pil_img.height,
                    "detections": page_detections,
                })
        return all_results

    # ── Single-region OCR (loopback HTTP to local vLLM) ───────────────────────

    def _ocr_region(self, crop_bytes: bytes, task: str) -> tuple[str, float]:
        params   = _TASK_PARAMS[task]
        data_url = "data:image/jpeg;base64," + base64.b64encode(crop_bytes).decode()
        t0 = time.time()
        resp = self._session.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json={
                "model": SERVED_NAME,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url":        data_url,
                        "min_pixels": params["min_pixels"],
                        "max_pixels": params["max_pixels"],
                    }},
                    {"type": "text", "text": _PROMPT[task]},
                ]}],
                "max_tokens":  params["max_tokens"],
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return text, round(time.time() - t0, 3)

    # ── HTTP endpoint ─────────────────────────────────────────────────────────

    @modal.fastapi_endpoint(method="POST")
    def process(self, req: _Request) -> dict:
        import pypdfium2 as pdfium
        from PIL import Image

        t0 = time.time()

        # ── Decode PDF ──────────────────────────────────────────────────────
        raw_b64 = req.file
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            pdf_bytes = base64.b64decode(raw_b64)
            pdf       = pdfium.PdfDocument(pdf_bytes)
        except Exception as e:
            return {"error": f"PDF decode failed: {e}"}

        n_pages      = len(pdf)
        page_indices = req.pages or (
            list(range(min(req.num_pages, n_pages))) if req.num_pages else list(range(n_pages))
        )
        page_indices = [i for i in page_indices if 0 <= i < n_pages]
        if not page_indices:
            pdf.close()
            return {"error": "No valid pages"}

        t_decode = time.time()

        # ── Render ─────────────────────────────────────────────────────────
        scale     = req.dpi / 72
        pil_pages = {}
        for pi in page_indices:
            pg = pdf[pi]
            pil_pages[pi] = pg.render(scale=scale).to_pil().convert("RGB")
            pg.close()
        pdf.close()
        t_render = time.time()

        # ── Layout detection (GPU, in-process) ─────────────────────────────
        raw_layout = self._detect_layout(list(pil_pages.values()))
        t_layout   = time.time()

        # ── Classify, NMS, reading order, crop ────────────────────────────
        to_recognize: list[tuple] = []
        skipped:      list[dict]  = []

        for seq_idx, (pi, raw_page) in enumerate(zip(page_indices, raw_layout)):
            pil     = pil_pages[pi]
            regions = _nms(raw_page["detections"])
            regions = _reading_order(regions, raw_page["width_px"])

            for order, region in enumerate(regions):
                label = region.get("type", "text")
                task  = _TASK.get(label, "text")
                if task == "abandon":
                    continue
                x0, y0, x1, y1 = region["bbox"]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(pil.width, x1), min(pil.height, y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = pil.crop((x0, y0, x1, y1))
                buf  = io.BytesIO()
                crop.save(buf, format="JPEG", quality=92)
                crop_bytes = buf.getvalue()

                if task == "skip":
                    skipped.append({
                        "page":      pi,
                        "order":     order,
                        "label":     label,
                        "bbox":      [x0, y0, x1, y1],
                        "text":      None,
                        "image_b64": base64.b64encode(crop_bytes).decode(),
                    })
                else:
                    to_recognize.append((pi, order, label, crop_bytes, task))

        t_crop = time.time()

        # ── Parallel OCR via loopback HTTP ─────────────────────────────────
        ocr_results: list[dict] = []
        exec_times:  list[float] = []
        t_ocr_first_result: float | None = None

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(self._ocr_region, cb, task): (pi, order, label)
                for pi, order, label, cb, task in to_recognize
            }
            for future in as_completed(futures):
                if t_ocr_first_result is None:
                    t_ocr_first_result = time.time()
                pi, order, label = futures[future]
                try:
                    text, exec_s = future.result()
                    ocr_results.append({
                        "page": pi, "order": order, "label": label,
                        "text": text, "exec_s": exec_s,
                    })
                    exec_times.append(exec_s)
                except Exception as e:
                    ocr_results.append({
                        "page": pi, "order": order, "label": label,
                        "text": f"[OCR error: {e}]", "exec_s": None,
                    })

        t_ocr = time.time()

        # ── Assemble ────────────────────────────────────────────────────────
        all_blocks = ocr_results + skipped
        all_blocks.sort(key=lambda b: (b["page"], b["order"]))
        markdown = _build_markdown(all_blocks)
        t_assemble = time.time()

        avg_exec = round(sum(exec_times) / len(exec_times), 3) if exec_times else None
        min_exec = round(min(exec_times), 3)                    if exec_times else None
        max_exec = round(max(exec_times), 3)                    if exec_times else None
        first_result_s = round(t_ocr_first_result - t_crop, 3) if t_ocr_first_result else None

        timing = {
            "decode_s":             round(t_decode   - t0,         3),
            "render_s":             round(t_render   - t_decode,   3),
            "layout_s":             round(t_layout   - t_render,   3),
            "crop_s":               round(t_crop     - t_layout,   3),
            "ocr_wall_s":           round(t_ocr      - t_crop,     3),
            "ocr_first_result_s":   first_result_s,
            "ocr_avg_exec_s":       avg_exec,
            "ocr_min_exec_s":       min_exec,
            "ocr_max_exec_s":       max_exec,
            "assemble_s":           round(t_assemble - t_ocr,      3),
            "total_s":              round(t_assemble - t0,         3),
        }
        if hasattr(self, "_cold_start_timing"):
            timing["cold_start"] = self._cold_start_timing
            del self._cold_start_timing

        pages_info = [
            {"page": pi, "width_px": pil_pages[pi].width, "height_px": pil_pages[pi].height}
            for pi in page_indices
        ]

        print(
            f"[single] pages={len(page_indices)} regions={len(to_recognize)}  "
            f"decode={timing['decode_s']:.2f}s render={timing['render_s']:.2f}s "
            f"layout={timing['layout_s']:.2f}s ocr_wall={timing['ocr_wall_s']:.2f}s "
            f"first={first_result_s}s avg={avg_exec}s min={min_exec}s max={max_exec}s "
            f"total={timing['total_s']:.2f}s"
        )

        return {
            "markdown": markdown,
            "blocks":   all_blocks,
            "meta": {
                "pages":         page_indices,
                "pages_info":    pages_info,
                "total_regions": len(all_blocks),
                "ocr_regions":   len(to_recognize),
                "skip_regions":  len(skipped),
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
