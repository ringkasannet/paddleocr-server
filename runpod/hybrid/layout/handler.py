"""RunPod serverless handler — Layout Detection endpoint.

Startup (once per worker):
  1. Load PP-DocLayoutV3 from RunPod volume / HF cache
  2. pdfplumber is stateless — no startup needed

Per-request:
  - Render PDF pages with PyMuPDF at RENDER_DPI
  - Run layout detection in batches of BATCH_SIZE
  - Extract text via pdfplumber for text-type regions
  - Crop each region to PNG base64
  - Return structured response

Input:  {"images": ["data:application/pdf;base64,<b64>"]}
Output: {"pages": [...], "meta": {...}}
"""

from __future__ import annotations

import base64
import io
import os
import time
import traceback
from collections import defaultdict

import fitz  # PyMuPDF
import pdfplumber
import runpod
from PIL import Image

from glmocr.config import LayoutConfig
from glmocr.layout.layout_detector import PPDocLayoutDetector

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_LAYOUT = os.environ.get("MODEL_LAYOUT", "PaddlePaddle/PP-DocLayoutV3_safetensors")
RENDER_DPI   = int(os.environ.get("RENDER_DPI", "150"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "4"))
MAX_SIDE_PX  = int(os.environ.get("MAX_SIDE_PX", "3500"))

# Native labels for which pdfplumber text extraction is attempted.
# All other labels (table, formula, chart, image, etc.) always go to VLM.
TEXT_LABELS = frozenset({
    "abstract", "algorithm", "content", "doc_title", "figure_title",
    "paragraph_title", "reference_content", "text", "vertical_text",
    "vision_footnote", "seal", "formula_number",
})

# ── Model configuration ────────────────────────────────────────────────────────

_ID2LABEL = {
    0: "abstract",       1: "algorithm",       2: "aside_text",
    3: "chart",          4: "content",         5: "display_formula",
    6: "doc_title",      7: "figure_title",    8: "footer",
    9: "footer_image",  10: "footnote",        11: "formula_number",
    12: "header",       13: "header_image",    14: "image",
    15: "inline_formula", 16: "number",        17: "paragraph_title",
    18: "reference",    19: "reference_content", 20: "seal",
    21: "table",        22: "text",            23: "vertical_text",
    24: "vision_footnote",
}

_LABEL_TASK_MAPPING = {
    "text": [
        "abstract", "algorithm", "content", "doc_title", "figure_title",
        "paragraph_title", "reference_content", "text", "vertical_text",
        "vision_footnote", "seal", "formula_number",
    ],
    "table":   ["table"],
    "formula": ["display_formula", "inline_formula"],
    "skip":    ["chart", "image"],
    "abandon": [
        "header", "footer", "number", "footnote", "aside_text",
        "reference", "footer_image", "header_image",
    ],
}

# Per-class bbox merge mode — from PaddleOCR-VL-1.5.yaml (authoritative for PP-DocLayoutV3).
# "union" = keep all boxes; "large" = drop contained box, keep container.
# glmocr/config.yaml incorrectly uses "large" for most classes; the paddlex
# config for the same model uses "union" for all except the 5 classes below.
_MERGE_BBOXES_MODE = {
    0: "union", 1: "union", 2: "union", 3: "large", 4: "union",
    5: "large", 6: "large", 7: "union", 8: "union", 9: "union",
    10: "union", 11: "union", 12: "union", 13: "union", 14: "union",
    15: "large", 16: "union", 17: "large", 18: "union", 19: "union",
    20: "union", 21: "union", 22: "union", 23: "union", 24: "union",
}

# ── Model init (runs once before first request) ────────────────────────────────

_t0 = time.time()
print(f"[init] Loading PP-DocLayoutV3 from {MODEL_LAYOUT} ...")

_layout_config = LayoutConfig(
    model_dir=MODEL_LAYOUT,
    threshold=0.3,
    batch_size=BATCH_SIZE,
    layout_nms=True,
    layout_unclip_ratio=[1.0, 1.0],
    layout_merge_bboxes_mode=_MERGE_BBOXES_MODE,
    id2label=_ID2LABEL,
    label_task_mapping=_LABEL_TASK_MAPPING,
    device="cuda:0",
)

_detector = PPDocLayoutDetector(_layout_config)
_detector.start()
print(f"[init] PP-DocLayoutV3 ready in {time.time() - _t0:.1f}s")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_source(data_uri: str) -> tuple[bytes, bool]:
    """Base64-decode a data URI. Returns (raw_bytes, is_pdf).

    Accepts:
      data:application/pdf;base64,<b64>   → is_pdf=True
      data:image/jpeg;base64,<b64>        → is_pdf=False
      data:image/png;base64,<b64>         → is_pdf=False
      <bare b64 string>                    → sniff magic bytes
    """
    if "," in data_uri:
        header, b64 = data_uri.split(",", 1)
        raw = base64.b64decode(b64)
        is_pdf = "application/pdf" in header
    else:
        raw = base64.b64decode(data_uri)
        is_pdf = raw[:4] == b"%PDF"
    return raw, is_pdf


def _render_pdf(pdf_bytes: bytes) -> list[Image.Image]:
    """Render all PDF pages to PIL Images using PyMuPDF at RENDER_DPI."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    base_scale = RENDER_DPI / 72.0
    pages: list[Image.Image] = []
    for page in doc:
        rect = page.rect
        long_side = max(rect.width, rect.height)
        scale = min(base_scale, MAX_SIDE_PX / long_side)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return pages


def _build_page_regions(
    page_idx: int,
    page_img: Image.Image,
    regions: list[dict],
) -> tuple[dict, list[dict]]:
    """Convert layout detector output to API format for one page.

    Returns (page_dict_without_crops, flat_region_list_with_bbox_px).
    """
    w_px, h_px = page_img.size
    scale_pt = 72.0 / RENDER_DPI

    page_regions: list[dict] = []
    for region in regions:
        # bbox_2d is normalized 0–1000 from layout_detector.process()
        x1_n, y1_n, x2_n, y2_n = region["bbox_2d"]
        x1_px = int(x1_n / 1000.0 * w_px)
        y1_px = int(y1_n / 1000.0 * h_px)
        x2_px = int(x2_n / 1000.0 * w_px)
        y2_px = int(y2_n / 1000.0 * h_px)

        page_regions.append({
            "region_id":    f"p{page_idx}_r{region['index']}",
            "bbox_px":      [x1_px, y1_px, x2_px, y2_px],
            "bbox_pt":      [
                round(x1_px * scale_pt, 2), round(y1_px * scale_pt, 2),
                round(x2_px * scale_pt, 2), round(y2_px * scale_pt, 2),
            ],
            "label":        region["task_type"],    # text / table / formula / skip
            "native_label": region["label"],         # paragraph_title / table / etc.
            "score":        region["score"],
            "order":        region["index"],         # sequential reading-order index
            "polygon":      region["polygon"],
            "crop":         None,   # filled in below
            "text":         None,   # filled in by pdfplumber
        })

    page_dict = {
        "page_index": page_idx,
        "width_px":   w_px,
        "height_px":  h_px,
        "regions":    page_regions,
    }
    return page_dict


def _extract_pdfplumber(
    pdf_bytes: bytes,
    pages: list[dict],
) -> dict[str, str | None]:
    """Extract text for text-label regions in one PDF source.

    pages is the slice of the global pages list that belongs to this PDF.
    Returns mapping of region_id → extracted text string (or None).
    """
    text_map: dict[str, str | None] = {}
    scale_pt = 72.0 / RENDER_DPI

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_within_src, page_data in enumerate(pages):
                if page_within_src >= len(pdf.pages):
                    break
                pdf_page = pdf.pages[page_within_src]

                for region in page_data["regions"]:
                    rid = region["region_id"]
                    if region["native_label"] not in TEXT_LABELS:
                        text_map[rid] = None
                        continue

                    x1_px, y1_px, x2_px, y2_px = region["bbox_px"]
                    bbox_pt = (
                        x1_px * scale_pt,
                        y1_px * scale_pt,
                        x2_px * scale_pt,
                        y2_px * scale_pt,
                    )
                    try:
                        text = pdf_page.within_bbox(bbox_pt).extract_text()
                        text_map[rid] = text.strip() if text and text.strip() else None
                    except Exception:
                        text_map[rid] = None
    except Exception:
        # Non-searchable PDF — mark all regions as None
        for page_data in pages:
            for region in page_data["regions"]:
                text_map[region["region_id"]] = None

    return text_map


def _encode_crop(page_img: Image.Image, bbox_px: list[int]) -> str:
    """Crop a bounding box from a page image and return PNG base64 data URI."""
    x1, y1, x2, y2 = bbox_px
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(page_img.width, x2), min(page_img.height, y2)
    crop = page_img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ── Per-request handler ────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    try:
        job_input = job.get("input", {})
        images = job_input.get("images", [])
        if not images:
            return {"error": "input.images is required"}

        # ── Step 1: Decode + render ALL sources into a flat PIL page list ──────
        #
        # Each item in `images` can be a PDF (multiple pages) or a single image.
        # We render everything into one flat list so the layout model batches
        # pages across sources — same pattern as GLM-OCR's page_loader which
        # yields pages from all sources together into the layout worker queue.
        #
        # source_map[i] = (source_idx, page_within_source, pdf_bytes_or_None)
        #   source_idx          — which item in `images` this page came from
        #   page_within_source  — page number within that PDF (0 for images)
        #   pdf_bytes_or_None   — raw bytes if source is a PDF (for pdfplumber)

        all_pages_pil: list[Image.Image] = []
        source_map: list[tuple[int, int, bytes | None]] = []

        for src_idx, img_uri in enumerate(images):
            raw, is_pdf = _decode_source(img_uri)
            if is_pdf:
                src_pages = _render_pdf(raw)
                for page_within_src, page_img in enumerate(src_pages):
                    all_pages_pil.append(page_img)
                    source_map.append((src_idx, page_within_src, raw))
            else:
                page_img = Image.open(io.BytesIO(raw)).convert("RGB")
                all_pages_pil.append(page_img)
                source_map.append((src_idx, 0, None))  # no PDF bytes → no pdfplumber

        if not all_pages_pil:
            return {"error": "No pages rendered from input"}

        # ── Step 2: Layout detection over ALL pages in one batched call ────────
        #
        # PPDocLayoutDetector.process() chunks internally:
        #   for chunk_start in range(0, len(all_pages_pil), batch_size=4):
        #       forward_pass(chunk)
        #
        # Pages from different source PDFs share the same chunks when adjacent —
        # GPU utilisation is the same as if they came from one large PDF.

        all_results, _ = _detector.process(all_pages_pil)

        # ── Step 3: Build structured page dicts ───────────────────────────────

        pages = [
            _build_page_regions(global_idx, page_img, regions)
            for global_idx, (page_img, regions) in enumerate(
                zip(all_pages_pil, all_results)
            )
        ]

        # ── Step 4: pdfplumber — per source PDF, applied to its own pages ─────
        #
        # Group pages by source so each PDF's pdfplumber call uses the correct
        # page numbering (page_within_source, not the global index).

        text_map: dict[str, str | None] = {}

        # Collect per-source page slices
        src_page_groups: dict[int, list[tuple[int, bytes | None]]] = defaultdict(list)
        for global_idx, (src_idx, page_within_src, pdf_bytes) in enumerate(source_map):
            src_page_groups[src_idx].append((global_idx, pdf_bytes))

        for src_idx, page_entries in src_page_groups.items():
            global_indices = [g for g, _ in page_entries]
            pdf_bytes = page_entries[0][1]  # same pdf_bytes for every page in the source

            src_pages = [pages[g] for g in global_indices]

            if pdf_bytes is not None:
                text_map.update(_extract_pdfplumber(pdf_bytes, src_pages))
            else:
                # Image source: no text layer, all regions go to VLM
                for page_data in src_pages:
                    for region in page_data["regions"]:
                        text_map[region["region_id"]] = None

        # ── Step 5: Attach crops and text, accumulate stats ───────────────────

        total_regions = 0
        text_extracted = 0
        vlm_needed = 0

        for page_data, page_img in zip(pages, all_pages_pil):
            for region in page_data["regions"]:
                total_regions += 1
                rid = region["region_id"]
                region["crop"] = _encode_crop(page_img, region["bbox_px"])
                region["text"] = text_map.get(rid)
                if region["text"] is not None:
                    text_extracted += 1
                elif region["label"] != "skip":
                    vlm_needed += 1

        return {
            "pages": pages,
            "meta": {
                "total_regions": total_regions,
                "text_extracted": text_extracted,
                "vlm_needed": vlm_needed,
                "searchable": text_extracted > 0,
                "num_sources": len(images),
                "dpi": RENDER_DPI,
            },
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
