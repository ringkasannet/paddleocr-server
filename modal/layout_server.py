"""Local FastAPI server — PP-DocLayoutV3 layout detection (transformers backend).

Same interface as the Modal worker:
  POST /  {"file": "<base64>", "fileType": 0|1, "dpi": 200}
  → {"pages": [...], "dpi": 200, "searchable": bool}

Dependencies:
  pip install transformers torch torchvision opencv-python-headless pypdfium2 Pillow fastapi uvicorn

Download weights once (optional — omit to download from HuggingFace on first start):
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download('PaddlePaddle/PP-DocLayoutV3_safetensors', local_dir='weights/PP-DocLayoutV3')
  "

Start:
  python modal/layout_server.py
  uvicorn modal.layout_server:app --host 0.0.0.0 --port 8000

Test:
  python modal/test_layout_local.py /path/to/doc.pdf
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForObjectDetection

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_ID   = "PaddlePaddle/PP-DocLayoutV3_safetensors"
MODEL_DIR  = os.environ.get("MODEL_DIR", "weights/PP-DocLayoutV3")

DETECT_THRESHOLD  = float(os.environ.get("DETECT_THRESHOLD", "0.3"))
HEADING_THRESHOLD = float(os.environ.get("HEADING_THRESHOLD", "0.2"))
DETECT_BATCH_SIZE = int(os.environ.get("DETECT_BATCH_SIZE", "8"))

_HEADING_LABELS = {"paragraph_title", "doc_title"}

TEXT_REGION_TYPES = {
    "text", "title", "paragraph_title", "abstract",
    "content", "reference", "footnote", "doc_title",
    "header", "footer", "reference_content", "aside_text",
    "figure_title", "table_title", "chart_title",
    "number", "vision_footnote",
}

_pdf_lock  = threading.Lock()  # pypdfium2 global C state is not thread-safe
_gpu_lock  = threading.Lock()
_processor = None
_model     = None
_device    = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _model, _device
    import pypdfium2 as pdfium

    model_src = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "config.json")) else MODEL_ID
    log.info(f"[init] device={'cuda' if torch.cuda.is_available() else 'cpu'}  model={model_src}")

    log.info("[init] loading processor ...")
    _processor = AutoImageProcessor.from_pretrained(model_src)

    log.info("[init] loading model weights ...")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model  = AutoModelForObjectDetection.from_pretrained(model_src).to(_device).eval()

    log.info(f"[init] warming up model on {_device} ...")
    dummy  = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    inputs = _processor(images=[dummy], return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        _model(**inputs)

    log.info("[init] pre-initializing pypdfium2 ...")
    _dummy = pdfium.PdfDocument.new()
    del _dummy

    log.info("[init] ready")
    yield


app = FastAPI(lifespan=lifespan)


class _Request(BaseModel):
    file: str
    fileType: int = 0
    dpi: int = 200


# ── helpers ───────────────────────────────────────────────────────────────────

def _searchable_pages(pdf, min_chars: int = 10) -> set[int]:
    result = set()
    for i in range(len(pdf)):
        page = pdf[i]
        textpage = page.get_textpage()
        if len(textpage.get_text_range()) > min_chars:
            result.add(i)
        textpage.close()
        page.close()
    return result


def _extract_text(pdf, page_num: int, bbox_px: list[int], dpi: int) -> str:
    scale = dpi / 72
    x0, y0, x1, y1 = [c / scale for c in bbox_px]
    page = pdf[page_num]
    h = page.get_height()
    textpage = page.get_textpage()
    m = 2
    text = textpage.get_text_bounded(
        left=x0 + m, bottom=h - y1 + m, right=x1 - m, top=h - y0 - m
    )
    textpage.close()
    page.close()
    text = (text or "").strip()
    text = text.replace('\x02', '-')  # pypdfium2 encodes line-break hyphens as STX
    return re.sub(r"([a-z])([A-Z])", r"\1-\2", text)


def _nms_regions(regions: list) -> list:
    if len(regions) <= 1:
        return regions
    by_score = sorted(regions, key=lambda r: r["score"], reverse=True)
    kept: list = []
    for cand in by_score:
        cx0, cy0, cx1, cy1 = cand["bbox"]
        ca = max(0, cx1 - cx0) * max(0, cy1 - cy0)
        if ca == 0:
            continue
        dominated = False
        for k in kept:
            kx0, ky0, kx1, ky1 = k["bbox"]
            ka = max(0, kx1 - kx0) * max(0, ky1 - ky0)
            if ka == 0:
                continue
            ix0, iy0 = max(cx0, kx0), max(cy0, ky0)
            ix1, iy1 = min(cx1, kx1), min(cy1, ky1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            if (ix1 - ix0) * (iy1 - iy0) / min(ca, ka) > 0.65:
                if not k["text"] and cand["text"]:
                    k["text"] = cand["text"]
                dominated = True
                break
        if not dominated:
            kept.append(cand)
    return kept


import re as _re
_ID_DOC_TITLE  = _re.compile(r'\b(PERATURAN|UNDANG-UNDANG|KEPUTUSAN|INSTRUKSI|PERDA)\b')
_ID_PASAL      = _re.compile(r'^Pasal\s+\d+$')
_ID_BULLET     = _re.compile(r'^[a-z0-9]{1,3}[.)]\s')
_FIGURE_TYPES  = {"image", "chart", "figure", "table", "footer_image", "header_image"}
_CAPTION_TYPES = {"figure_title", "table_title", "chart_title"}


def _is_indonesian_regulation(result_pages: list) -> bool:
    has_title = any(
        r.get("type") == "doc_title" and _ID_DOC_TITLE.search(r.get("text", ""))
        for p in result_pages for r in p["regions"]
    )
    if not has_title:
        return False
    return any(
        r.get("type") == "paragraph_title" and _ID_PASAL.fullmatch(r.get("text", "").strip())
        for p in result_pages for r in p["regions"]
    )


def _demote_bullet_items(regions: list) -> list:
    result = []
    for r in regions:
        if r["type"] == "paragraph_title" and _ID_BULLET.match(r.get("text", "")):
            r = dict(r)
            r["type"] = "text"
        result.append(r)
    return result


def _demote_orphan_caption_types(regions: list) -> list:
    """Demote figure_title/table_title/chart_title → paragraph_title when neither
    the preceding nor the following region is a figure/table/chart/image.
    Captions appear directly before or after their figure; if no adjacent
    figure-like region exists the model has misclassified a heading."""
    if not regions:
        return regions
    result = []
    for i, r in enumerate(regions):
        if r["type"] in _CAPTION_TYPES:
            prev_type = regions[i - 1]["type"] if i > 0 else None
            next_type = regions[i + 1]["type"] if i + 1 < len(regions) else None
            if prev_type not in _FIGURE_TYPES and next_type not in _FIGURE_TYPES:
                r = dict(r)
                r["type"] = "paragraph_title"
        result.append(r)
    return result


def _reclassify_centered_headings(regions: list, page_height: int,
                                  center_tol: float = 0.08,
                                  width_ratio: float = 0.25,
                                  bottom_margin: float = 0.92) -> list:
    if not regions:
        return regions
    content_x0 = min(r["bbox"][0] for r in regions)
    content_x1 = max(r["bbox"][2] for r in regions)
    content_center = (content_x0 + content_x1) / 2
    content_width  = max(content_x1 - content_x0, 1)
    bottom_limit   = page_height * bottom_margin

    result = []
    for r in regions:
        rtype = r["type"]
        if rtype == "text":
            x0, _, x1, y1 = r["bbox"]
            w = x1 - x0
            centered   = abs((x0 + x1) / 2 - content_center) / content_width < center_tol
            narrow     = w / content_width < width_ratio
            not_footer = y1 < bottom_limit
            if centered and narrow and not_footer:
                r = dict(r)
                r["type"] = "paragraph_title"
        result.append(r)
    return result


def _merge_centered_titles(regions: list,
                           gap_px: int = 30, center_tol: float = 0.08) -> list:
    if len(regions) <= 1:
        return regions
    content_x0 = min(r["bbox"][0] for r in regions)
    content_x1 = max(r["bbox"][2] for r in regions)
    content_center = (content_x0 + content_x1) / 2
    content_width  = max(content_x1 - content_x0, 1)

    def _centered(bbox):
        return abs((bbox[0] + bbox[2]) / 2 - content_center) / content_width < center_tol

    result = [dict(regions[0])]
    for r in regions[1:]:
        prev = result[-1]
        if (prev["type"] == "paragraph_title"
                and r["type"] == "paragraph_title"
                and _centered(prev["bbox"])
                and _centered(r["bbox"])
                and r["bbox"][1] - prev["bbox"][3] < gap_px):
            result[-1] = {
                "type":  "paragraph_title",
                "bbox":  [min(prev["bbox"][0], r["bbox"][0]),
                          prev["bbox"][1],
                          max(prev["bbox"][2], r["bbox"][2]),
                          r["bbox"][3]],
                "score": max(prev["score"], r["score"]),
                "text":  "\r\n".join(t for t in [prev["text"], r["text"]] if t),
            }
        else:
            result.append(dict(r))
    return result


def _reading_order(regions: list, page_width: int) -> list:
    if len(regions) <= 1:
        return list(regions)
    w = page_width
    y_sorted = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    def _is_column(r):
        x0, _, x1, _ = r["bbox"]
        return (x0 < w / 4 and x1 < w * 3 / 5) or (x0 > w * 2 / 5)

    result: list = []
    seg:    list = []
    for reg in y_sorted:
        if _is_column(reg):
            seg.append(reg)
        else:
            if seg:
                result.extend(_xycut_segment(seg, w))
                seg = []
            result.append(reg)
    if seg:
        result.extend(_xycut_segment(seg, w))
    return result


def _xycut_segment(regions: list, w: float) -> list:
    if len(regions) <= 1:
        return list(regions)
    boxes = np.array([r["bbox"] for r in regions], dtype=float)
    ordered: list = []
    _recursive_xy_cut(boxes, list(range(len(regions))), ordered)
    if len(ordered) == len(regions):
        return [regions[i] for i in ordered]
    return _sorted_layout_boxes(regions, w)


def _sorted_layout_boxes(regions: list, w: float) -> list:
    if len(regions) <= 1:
        return list(regions)
    boxes = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    new_res, res_left, res_right = [], [], []
    for reg in boxes:
        x0, _, x1, _ = reg["bbox"]
        if x0 < w / 4 and x1 < w * 3 / 5:
            res_left.append(reg)
        elif x0 > w * 2 / 5:
            res_right.append(reg)
        else:
            new_res += res_left + res_right
            new_res.append(reg)
            res_left, res_right = [], []
    res_left.sort(key=lambda r: r["bbox"][1])
    res_right.sort(key=lambda r: r["bbox"][1])
    return new_res + res_left + res_right


def _recursive_xy_cut(boxes, indices: list, res: list, min_gap: int = 1) -> None:
    if len(boxes) == 0:
        return
    x_ord = boxes[:, 0].argsort()
    bx, ix = boxes[x_ord], np.array(indices)[x_ord]
    x_segs = _segments(_projection(bx, 0), min_gap)
    if x_segs is None:
        return
    for xs, xe in zip(*x_segs):
        mask = (bx[:, 0] >= xs) & (bx[:, 0] < xe)
        cb, ci = bx[mask], ix[mask]
        y_ord = cb[:, 1].argsort()
        cb, ci = cb[y_ord], ci[y_ord]
        y_segs = _segments(_projection(cb, 1), min_gap)
        if y_segs is None:
            continue
        if len(y_segs[0]) == 1:
            res.extend(ci.tolist())
            continue
        for ys, ye in zip(*y_segs):
            ym = (cb[:, 1] >= ys) & (cb[:, 1] < ye)
            _recursive_xy_cut(cb[ym], ci[ym].tolist(), res, min_gap)


def _projection(boxes, axis: int):
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    vals = boxes[:, axis::2].astype(int)
    proj = np.zeros(int(vals.max()) + 1, dtype=int)
    for lo, hi in vals:
        proj[lo:hi] += 1
    return proj


def _segments(arr, min_gap: int = 1):
    sig = np.where(arr > 0)[0]
    if not len(sig):
        return None
    gaps   = np.where(np.diff(sig) > min_gap)[0]
    starts = np.concatenate([[sig[0]], sig[gaps + 1]])
    ends   = np.concatenate([sig[gaps], [sig[-1] + 1]])
    return starts, ends


def _detect(page_jpegs: list[bytes]) -> dict:
    t0 = time.time()
    log.info(f"[detect] started  pages={len(page_jpegs)}  batch_size={DETECT_BATCH_SIZE}")

    pil_images = [Image.open(io.BytesIO(j)).convert("RGB") for j in page_jpegs]

    raw_pages = []
    # Lock covers preprocessing too — _processor uses PyTorch ops and is not thread-safe
    with _gpu_lock:
        cpu_chunks = []
        for chunk_start in range(0, len(pil_images), DETECT_BATCH_SIZE):
            chunk      = pil_images[chunk_start : chunk_start + DETECT_BATCH_SIZE]
            cpu_inputs = _processor(images=chunk, return_tensors="pt")
            cpu_chunks.append((chunk, cpu_inputs))

        for chunk, cpu_inputs in cpu_chunks:
            gpu_inputs = {k: v.to(_device) for k, v in cpu_inputs.items()}
            with torch.no_grad():
                outputs = _model(**gpu_inputs)
            batch_detections = _processor.post_process_object_detection(
                outputs, threshold=min(DETECT_THRESHOLD, HEADING_THRESHOLD),
                target_sizes=[img.size[::-1] for img in chunk],
            )
            for pil_img, detections in zip(chunk, batch_detections):
                page_detections = []
                for score, label_id, box in zip(
                    detections["scores"], detections["labels"], detections["boxes"]
                ):
                    label  = _model.config.id2label[label_id.item()]
                    cutoff = HEADING_THRESHOLD if label in _HEADING_LABELS else DETECT_THRESHOLD
                    if score.item() < cutoff:
                        continue
                    x0, y0, x1, y1 = box.tolist()
                    page_detections.append({
                        "type":  label,
                        "bbox":  [int(x0), int(y0), int(x1), int(y1)],
                        "score": round(score.item(), 4),
                        "text":  "",
                    })
                raw_pages.append({
                    "width_px":   pil_img.width,
                    "height_px":  pil_img.height,
                    "detections": page_detections,
                })

    detect_s = round(time.time() - t0, 3)
    log.info(f"[detect] done  detect_s={detect_s}s  pages={len(raw_pages)}")
    return {"pages": raw_pages, "detect_s": detect_s}


# ── endpoint ──────────────────────────────────────────────────────────────────

@app.post("/")
def process(req: _Request) -> dict:
    import pypdfium2 as pdfium

    t0 = time.time()
    log.info(f"[process] received  fileType={req.fileType}  dpi={req.dpi}")

    raw_b64 = req.file
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        file_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        return {"error": f"Bad request: {e}"}

    file_type = req.fileType
    dpi       = req.dpi

    try:
        if file_type == 1:
            pil_pages = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
            pdf = None
            searchable_set: set[int] = set()
        else:
            with _pdf_lock:
                pdf   = pdfium.PdfDocument(file_bytes)
                scale = dpi / 72
                pil_pages = []
                for i in range(len(pdf)):
                    pg = pdf[i]
                    pil_pages.append(pg.render(scale=scale).to_pil())
                    pg.close()
                searchable_set = _searchable_pages(pdf)
    except Exception as e:
        return {"error": f"Render failed: {e}"}
    t_render = time.time()

    page_jpegs = []
    for img in pil_pages:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        page_jpegs.append(buf.getvalue())

    log.info(f"[process] calling detect  pages={len(page_jpegs)}")
    gpu_result = _detect(page_jpegs)
    t_gpu_done = time.time()

    if "error" in gpu_result:
        return gpu_result

    detect_s  = gpu_result.pop("detect_s", 0)
    raw_pages = gpu_result["pages"]

    # Pass 1: NMS + reading order + text extraction
    # Text must be populated before regulation detection can run.
    result_pages = []
    for page_num, (page_data, _) in enumerate(zip(raw_pages, pil_pages)):
        regions = page_data["detections"]
        regions = _nms_regions(regions)
        regions = _reading_order(regions, page_data["width_px"])
        result_pages.append({
            "page_num":  page_num,
            "width_px":  page_data["width_px"],
            "height_px": page_data["height_px"],
            "regions":   regions,
        })

    if pdf is not None:
        with _pdf_lock:
            for page in result_pages:
                if page["page_num"] in searchable_set:
                    for region in page["regions"]:
                        if region["type"] in TEXT_REGION_TYPES:
                            region["text"] = _extract_text(
                                pdf, page["page_num"], region["bbox"], dpi
                            )
            pdf.close()

    # Pass 2: regulation-aware post-processing on text-populated regions
    is_id_reg = _is_indonesian_regulation(result_pages)
    for page in result_pages:
        regions = page["regions"]
        if is_id_reg:
            regions = _demote_bullet_items(regions)
        regions = _demote_orphan_caption_types(regions)
        regions = _reclassify_centered_headings(regions, page["height_px"])
        regions = _merge_centered_titles(regions)
        for i, r in enumerate(regions):
            r["order"] = i
        page["regions"] = regions
    t_text = time.time()

    page_count = len(result_pages)
    render_s   = round(t_render - t0, 3)
    text_s     = round(t_text - t_gpu_done, 3)
    wall_s     = round(t_text - t0, 3)
    log.info(f"[process] done  pages={page_count}  render_s={render_s}  detect_s={detect_s}  text_s={text_s}  wall_s={wall_s}")

    return {
        "pages":      result_pages,
        "dpi":        dpi,
        "searchable": bool(searchable_set),
        "meta": {
            "page_count":    page_count,
            "total_regions": sum(len(p["regions"]) for p in result_pages),
            "timing": {
                "render_s":    render_s,
                "detect_s":    detect_s,
                "text_s":      text_s,
                "wall_s":      wall_s,
            },
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)