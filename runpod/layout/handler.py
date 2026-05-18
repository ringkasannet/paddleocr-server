"""RunPod serverless handler — PP-DocLayoutV3 layout detection + text extraction.

Input:
  {"file": "<base64>", "fileType": 0, "dpi": 150}
  fileType 0 = PDF (default), 1 = single image

Output:
  {"pages": [{"page_num": 0, "width_px": W, "height_px": H,
               "regions": [{"type": "text", "bbox": [x0,y0,x1,y1],
                             "score": 0.94, "order": 3, "text": "..."}]}],
   "dpi": 150, "searchable": true}

Text is extracted via pypdfium2 for text/title regions in searchable PDFs.
Non-text regions (table, figure, formula, etc.) always have text: "".

Model weights are provided via RunPod model caching.
Set endpoint cached model to: PaddlePaddle/PP-DocLayoutV3_safetensors
"""

import base64
import io
import os
import re

import numpy as np
import pypdfium2 as pdfium
import runpod
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID   = "PaddlePaddle/PP-DocLayoutV3_safetensors"
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))
THRESHOLD  = float(os.environ.get("DETECT_THRESHOLD", "0.3"))

# Region types where text extraction from the PDF layer is meaningful.
# Tables/figures/formulas/seals always go to VLM — excluded here.
TEXT_REGION_TYPES = {
    "text", "title", "paragraph_title", "abstract",
    "content", "reference", "footnote", "doc_title",
    "header", "footer", "reference_content", "aside_text",
    "figure_title", "table_title", "chart_title",
    "number", "vision_footnote",
}

# ── Model load (once per worker, before any request) ─────────────────────────
print(f"[init] Loading {MODEL_ID} …")
_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
_model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
_model = _model.to("cuda").eval()
print("[init] Model ready")


# ── PDF helpers ───────────────────────────────────────────────────────────────
def _pdf_to_pil_pages(pdf: pdfium.PdfDocument, dpi: int) -> list[Image.Image]:
    scale = dpi / 72  # PDFium renders at 72 dpi by default; scale factor adjusts it
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale, rotation=0)
        pages.append(bitmap.to_pil())
        page.close()
    return pages


def _searchable_pages(pdf: pdfium.PdfDocument, min_chars: int = 10) -> set[int]:
    """Return indices of pages that have a meaningful PDF text layer.

    Checked per-page so mixed documents (e.g. text cover + scanned body)
    extract text only from the pages that actually carry a text layer.
    min_chars=10 handles sparse forms/invoices with short field labels.
    """
    result = set()
    for i in range(len(pdf)):
        page = pdf[i]
        textpage = page.get_textpage()
        if len(textpage.get_text_range()) > min_chars:
            result.add(i)
        textpage.close()
        page.close()
    return result


def _extract_text_at_bbox(pdf: pdfium.PdfDocument, page_num: int,
                           bbox_px: list[int], dpi: int) -> str:
    scale = dpi / 72
    # Convert pixel coords back to PDF points
    x0 = bbox_px[0] / scale
    y0 = bbox_px[1] / scale
    x1 = bbox_px[2] / scale
    y1 = bbox_px[3] / scale

    page = pdf[page_num]
    # PDFium y-axis is bottom-up; page height needed to flip
    h = page.get_height()
    textpage = page.get_textpage()
    # 2-pt inward margin: integer pixel→point conversion can land on the edge
    # of adjacent column glyphs; shrinking slightly prevents bleed-in without
    # clipping real content (model bbox precision is >>2 pt).
    m = 2
    text = textpage.get_text_bounded(
        left   = x0 + m,
        bottom = h - y1 + m,
        right  = x1 - m,
        top    = h - y0 - m,
    )
    textpage.close()
    page.close()
    text = (text or "").strip()
    text = re.sub(r"([a-z])([A-Z])", r"\1-\2", text)
    text = text.replace("", "").replace("­", "")
    return text


# ── Post-processing (adapted from PaddleX — pure NumPy, no GPU) ──────────────

def _projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """1-D projection histogram of bboxes along axis (0=x, 1=y)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    vals = boxes[:, axis::2].astype(int)
    max_val = int(vals.max()) + 1
    proj = np.zeros(max_val, dtype=int)
    for lo, hi in vals:
        proj[lo:hi] += 1
    return proj


def _split_projection_profile(arr: np.ndarray, min_gap: int = 1):
    """Return (starts, ends) of contiguous non-zero segments separated by
    gaps of at least min_gap, or None if no non-zero values exist."""
    sig = np.where(arr > 0)[0]
    if not len(sig):
        return None
    gaps = np.where(np.diff(sig) > min_gap)[0]
    starts = np.concatenate([[sig[0]], sig[gaps + 1]])
    ends   = np.concatenate([sig[gaps], [sig[-1] + 1]])
    return starts, ends


def _recursive_xy_cut(
    boxes: np.ndarray, indices: list, res: list, min_gap: int = 1
) -> None:
    """Recursive XY-cut: find natural X column gaps first, then Y row gaps.
    Adapted from PaddleX xycut_enhanced/utils.py:recursive_xy_cut."""
    if len(boxes) == 0:
        return
    x_ord = boxes[:, 0].argsort()
    bx, ix = boxes[x_ord], np.array(indices)[x_ord]

    x_segs = _split_projection_profile(_projection_by_bboxes(bx, 0), min_gap)
    if x_segs is None:
        return

    for xs, xe in zip(*x_segs):
        mask = (bx[:, 0] >= xs) & (bx[:, 0] < xe)
        cb, ci = bx[mask], ix[mask]
        y_ord = cb[:, 1].argsort()
        cb, ci = cb[y_ord], ci[y_ord]

        y_segs = _split_projection_profile(_projection_by_bboxes(cb, 1), min_gap)
        if y_segs is None:
            continue
        if len(y_segs[0]) == 1:
            res.extend(ci.tolist())
            continue
        for ys, ye in zip(*y_segs):
            ym = (cb[:, 1] >= ys) & (cb[:, 1] < ye)
            _recursive_xy_cut(cb[ym], ci[ym].tolist(), res, min_gap)


def _sorted_layout_boxes(regions: list, w: float) -> list:
    """Column-aware sort using PaddleX x-thresholds (fallback).
    Adapted from PaddleX inference/pipelines/layout_parsing/utils.py."""
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


def _xycut_segment(regions: list, w: float) -> list:
    """Order one column segment with recursive_xy_cut; fall back to
    _sorted_layout_boxes if xy_cut returns a mismatched index count."""
    if len(regions) <= 1:
        return list(regions)
    boxes = np.array([r["bbox"] for r in regions], dtype=float)
    ordered: list = []
    _recursive_xy_cut(boxes, list(range(len(regions))), ordered)
    if len(ordered) == len(regions):
        return [regions[i] for i in ordered]
    return _sorted_layout_boxes(regions, w)


def _reading_order(regions: list, page_width: int) -> list:
    """Two-phase column-aware reading order.

    Phase 1 (PaddleX thresholds): classify each region as left-column,
      right-column, or full-width. Full-width regions act as dividers,
      separating the page into segments. Left-before-right is enforced at
      each divider, matching PaddleX sorted_layout_boxes behaviour.

    Phase 2 (recursive_xy_cut): within each column segment, use projection-
      histogram XY-cut to detect natural column gaps without fixed thresholds.
      This correctly handles uneven columns and avoids the gap-filling problem
      caused by full-width elements (which are already removed in Phase 1).
    """
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


def _nms_regions(regions: list) -> list:
    """Containment-based NMS to remove cross-class duplicate detections.

    HuggingFace post_process_object_detection applies per-class NMS only;
    the same bbox can survive as two different region types. Uses PaddleX
    threshold (0.65 overlap/min-area). Higher-score region wins. If the
    winner has no text but the dominated region does, text is transferred
    before the duplicate is dropped.
    """
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
            containment = (ix1 - ix0) * (iy1 - iy0) / min(ca, ka)
            if containment > 0.65:
                if not k["text"] and cand["text"]:
                    k["text"] = cand["text"]
                dominated = True
                break
        if not dominated:
            kept.append(cand)
    return kept


# ── Detection ─────────────────────────────────────────────────────────────────
def _detect_pages(pages: list[Image.Image]) -> list[dict]:
    results = []
    for page_num, pil_img in enumerate(pages):
        inputs = _processor(images=[pil_img], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)

        detections = _processor.post_process_object_detection(
            outputs, threshold=THRESHOLD, target_sizes=[pil_img.size[::-1]]
        )[0]

        regions = []
        for score, label_id, box in zip(
            detections["scores"], detections["labels"], detections["boxes"]
        ):
            x0, y0, x1, y1 = box.tolist()
            regions.append({
                "type":  _model.config.id2label[label_id.item()],
                "bbox":  [int(x0), int(y0), int(x1), int(y1)],
                "score": round(score.item(), 4),
                "text":  "",
            })

        regions = _nms_regions(regions)
        regions = _reading_order(regions, pil_img.width)
        for i, r in enumerate(regions):
            r["order"] = i

        results.append({
            "page_num":  page_num,
            "width_px":  pil_img.width,
            "height_px": pil_img.height,
            "regions":   regions,
        })
    return results


# ── Per-request handler ───────────────────────────────────────────────────────
def handler(job):
    job_input = job.get("input", {})

    raw_b64 = job_input.get("file", "")
    if not raw_b64:
        return {"error": "Missing 'file' field"}

    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        file_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        return {"error": f"Invalid base64: {e}"}

    file_type = int(job_input.get("fileType", 0))
    dpi = int(job_input.get("dpi", RENDER_DPI))

    try:
        if file_type == 1:
            pil_pages = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
            pdf = None
            searchable_set: set[int] = set()
        else:
            pdf = pdfium.PdfDocument(file_bytes)
            pil_pages = _pdf_to_pil_pages(pdf, dpi)
            searchable_set = _searchable_pages(pdf)
    except Exception as e:
        return {"error": f"Render failed: {e}"}

    try:
        result_pages = _detect_pages(pil_pages)
    except Exception as e:
        if pdf:
            pdf.close()
        return {"error": f"Detection failed: {e}"}

    if pdf is not None:
        for page in result_pages:
            if page["page_num"] in searchable_set:
                for region in page["regions"]:
                    if region["type"] in TEXT_REGION_TYPES:
                        region["text"] = _extract_text_at_bbox(
                            pdf, page["page_num"], region["bbox"], dpi
                        )
        pdf.close()

    return {"pages": result_pages, "dpi": dpi, "searchable": bool(searchable_set)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
