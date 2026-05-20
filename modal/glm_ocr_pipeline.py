"""Modal serverless deployment — GLM-OCR full pipeline (layout detection + OCR).

Pipeline per page:
  1. Render PDF → JPEG pages
  2. Layout detection  (cross-app: layout-worker / LayoutDetector.detect)
  3. NMS + reading-order sort on raw detections
  4. Region classification → text / table / formula / skip / abandon
  5. Crop each non-abandon region
  6. Parallel GLM-OCR (cross-app: glm-ocr / GLMOCRWorker.recognize)
     - text    → "Text Recognition:"
     - table   → "Table Recognition:"
     - formula → "Formula Recognition:"
     - skip    → no OCR, placeholder in output
  7. Assemble Markdown in reading order

One-time setup:
  (Both layout-worker and glm-ocr must already be deployed and have snapshots.)

Deploy:
  modal deploy modal/glm_ocr_pipeline.py

Test:
  python modal/test_glm_ocr_pipeline.py document.pdf
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Optional

import modal
from pydantic import BaseModel as _BaseModel

app = modal.App("glm-ocr-pipeline")

# ── Cross-app references ───────────────────────────────────────────────────────

_RemoteLayoutDetector = modal.Cls.from_name("layout-worker", "LayoutDetector")
_RemoteGLMOCRWorker   = modal.Cls.from_name("glm-ocr",       "GLMOCRWorker")

# ── Label → task routing (mirrors GLM-OCR SDK label_task_mapping) ─────────────

_TASK: dict[str, str] = {
    # send to GLM-OCR
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
    # keep crop, no OCR
    "image":             "skip",
    "figure":            "skip",
    "chart":             "skip",
    "figure_title":      "skip",
    "table_title":       "skip",
    "chart_title":       "skip",
    # discard entirely
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

# Per-task pixel budget and token limit.
# min/max_pixels control how the image processor resizes the crop before
# encoding — fewer pixels → fewer visual tokens → faster prefill.
# max_tokens caps the decode phase — text crops are short, tables moderate,
# formulas compact. Matching these to actual output length keeps the GPU moving.
_TASK_PARAMS: dict[str, dict] = {
    "text":    {"min_pixels": 112_896, "max_pixels":   512_000, "max_tokens": 4096},
    "table":   {"min_pixels": 112_896, "max_pixels": 1_003_520, "max_tokens": 4096},
    "formula": {"min_pixels": 112_896, "max_pixels":   512_000, "max_tokens": 4096},
}

# ── CPU image ──────────────────────────────────────────────────────────────────

_cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "pip install --no-cache-dir uv",
        "uv pip install --system --no-cache pypdfium2 Pillow numpy 'fastapi[standard]'",
    )
)

# ── Request / response schemas ─────────────────────────────────────────────────

class _PipelineRequest(_BaseModel):
    file:  str                        # base64-encoded PDF
    pages: Optional[list[int]] = None # page indices to process (None = all)
    dpi:   int = 200


# ── NMS + reading-order helpers ────────────────────────────────────────────────

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
    """Simple two-column reading order: left column → right column, top to bottom."""
    if len(regions) <= 1:
        return list(regions)
    mid = page_width / 2
    left  = sorted([r for r in regions if r["bbox"][2] <= mid * 1.1], key=lambda r: r["bbox"][1])
    right = sorted([r for r in regions if r["bbox"][0] >= mid * 0.9], key=lambda r: r["bbox"][1])
    full  = sorted([r for r in regions if r["bbox"][2] > mid * 1.1 and r["bbox"][0] < mid * 0.9],
                   key=lambda r: r["bbox"][1])

    result: list = []
    li = ri = fi = 0
    while li < len(left) or ri < len(right) or fi < len(full):
        tops = []
        if li < len(left):   tops.append(("l", left[li]["bbox"][1]))
        if ri < len(right):  tops.append(("r", right[ri]["bbox"][1]))
        if fi < len(full):   tops.append(("f", full[fi]["bbox"][1]))
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


# ── Pipeline frontend (CPU) ────────────────────────────────────────────────────

@app.cls(
    image=_cpu_image,
    timeout=900,    # > client timeout (600s) so client times out cleanly, never a server 500
    scaledown_window=30,
    max_containers=20,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=10, target_inputs=5)
class PipelineFrontend:

    @modal.enter(snap=True)
    def load(self) -> None:
        import pypdfium2
        from PIL import Image
        import numpy as np
        _ = (pypdfium2, Image, np)
        print("[pipeline] ready")

    @modal.fastapi_endpoint(method="POST")
    async def process(self, req: _PipelineRequest) -> dict:
        import pypdfium2 as pdfium
        from PIL import Image

        t0 = time.time()

        # ── Decode PDF ──────────────────────────────────────────────────────
        raw_b64 = req.file
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            pdf_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            return {"error": f"Bad base64: {e}"}

        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
        except Exception as e:
            return {"error": f"PDF open failed: {e}"}

        n_pages     = len(pdf)
        page_indices = req.pages if req.pages is not None else list(range(n_pages))
        page_indices = [i for i in page_indices if 0 <= i < n_pages]
        if not page_indices:
            pdf.close()
            return {"error": "No valid pages requested"}

        # ── Render pages ────────────────────────────────────────────────────
        scale = req.dpi / 72
        pil_pages:  dict[int, Image.Image] = {}
        page_jpegs: list[bytes]            = []

        for pi in page_indices:
            pg  = pdf[pi]
            pil = pg.render(scale=scale).to_pil().convert("RGB")
            pg.close()
            pil_pages[pi] = pil
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=92)
            page_jpegs.append(buf.getvalue())
        pdf.close()

        t_render = time.time()

        # ── Layout detection (cross-app GPU call) ───────────────────────────
        try:
            layout_result = await _RemoteLayoutDetector().detect.remote.aio(page_jpegs)
        except Exception as e:
            return {"error": f"Layout detection failed: {e}"}

        t_layout = time.time()

        # ── Classify, NMS, reading-order, crop ──────────────────────────────
        to_recognize: list[tuple] = []  # (page_idx, order, label, crop_bytes, task)
        skipped:      list[dict]  = []  # image/chart/figure — kept without OCR

        raw_pages = layout_result.get("pages", [])
        for seq_idx, (pi, raw_page) in enumerate(zip(page_indices, raw_pages)):
            pil         = pil_pages[pi]
            width_px    = raw_page["width_px"]
            detections  = raw_page.get("detections", [])

            regions = _nms(detections)
            regions = _reading_order(regions, width_px)

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

        # ── Parallel OCR (cross-app GPU calls) ──────────────────────────────
        async def _ocr_one(pi, order, label, crop_bytes, task):
            prompt  = _PROMPT[task]
            params  = _TASK_PARAMS[task]
            t_call  = time.time()
            try:
                result = await _RemoteGLMOCRWorker().recognize.remote.aio(
                    crop_bytes,
                    prompt,
                    params["max_tokens"],
                    params["min_pixels"],
                    params["max_pixels"],
                )
                t_done = time.time()
                return {
                    "page": pi, "order": order, "label": label,
                    "text":      result["text"],
                    "queued_s":  round(result["_start_ts"] - t_call, 3),
                    "exec_s":    result["exec_s"],
                    "wall_s":    round(t_done - t_call, 3),
                }
            except Exception as e:
                return {
                    "page": pi, "order": order, "label": label,
                    "text": f"[OCR error: {e}]",
                    "queued_s": None, "exec_s": None,
                    "wall_s": round(time.time() - t_call, 3),
                }

        ocr_results = await asyncio.gather(*[
            _ocr_one(pi, ord_, lbl, cb, tsk)
            for pi, ord_, lbl, cb, tsk in to_recognize
        ])

        t_ocr = time.time()

        # ── Timing aggregates ────────────────────────────────────────────────
        valid_ocr = [r for r in ocr_results if r["queued_s"] is not None]
        avg_queued = round(sum(r["queued_s"] for r in valid_ocr) / len(valid_ocr), 3) if valid_ocr else None
        avg_exec   = round(sum(r["exec_s"]   for r in valid_ocr) / len(valid_ocr), 3) if valid_ocr else None
        max_wall   = round(max((r["wall_s"]  for r in ocr_results), default=0), 3)

        # ── Assemble + render Markdown ───────────────────────────────────────
        all_blocks = list(ocr_results) + skipped
        all_blocks.sort(key=lambda b: (b["page"], b["order"]))

        markdown = _build_markdown(all_blocks)

        return {
            "markdown": markdown,
            "blocks":   all_blocks,
            "meta": {
                "pages":         page_indices,
                "total_regions": len(all_blocks),
                "ocr_regions":   len(to_recognize),
                "skip_regions":  len(skipped),
                "timing": {
                    "render_s":        round(t_render - t0,       3),
                    "layout_s":        round(t_layout - t_render, 3),
                    "crop_s":          round(t_crop   - t_layout, 3),
                    "ocr_wall_s":      round(t_ocr    - t_crop,   3),  # asyncio.gather wall time
                    "ocr_avg_queued_s": avg_queued,                    # avg time waiting for GPU slot
                    "ocr_avg_exec_s":   avg_exec,                      # avg vLLM execution time
                    "ocr_max_wall_s":   max_wall,                      # slowest single region
                    "total_s":         round(t_ocr    - t0,       3),
                },
            },
        }
