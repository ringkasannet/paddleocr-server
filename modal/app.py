"""Modal serverless deployment — layout-only PDF extractor (profiling build).

Architecture:
  LayoutDetector   PP-DocLayoutV3 + pdfplumber   L4, GPU snapshot
  process_pdf      Orchestrator                   CPU-only

One-time setup:
  modal run modal/app.py::download_weights

Deploy:
  modal deploy modal/app.py

Test:
  python modal/test.py
"""

from __future__ import annotations

import base64
import os

import modal

app = modal.App("paddleocr-hybrid")

vol = modal.Volume.from_name("paddleocr-weights", create_if_missing=True)
WEIGHTS_PATH = "/weights"

LAYOUT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"

# ── Image ────────────────────────────────────────────────────────────────────────

layout_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .run_commands(
        "pip install --no-cache-dir uv",
        "uv pip install --system --no-cache 'glmocr[selfhosted]' pypdfium2 pdfplumber 'huggingface_hub[hf_transfer]' 'fastapi[standard]'",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ── Weight downloader (run once) ──────────────────────────────────────────────────

@app.function(
    image=layout_image,
    volumes={WEIGHTS_PATH: vol},
    timeout=1800,
)
def download_weights(hf_token: str = ""):
    from huggingface_hub import snapshot_download

    kwargs = {"token": hf_token} if hf_token else {}

    layout_dir = os.path.join(WEIGHTS_PATH, "PP-DocLayoutV3")
    if not os.path.exists(os.path.join(layout_dir, "config.json")):
        print(f"Downloading {LAYOUT_MODEL_ID} ...")
        snapshot_download(LAYOUT_MODEL_ID, local_dir=layout_dir, **kwargs)
        print("Layout model downloaded.")
    else:
        print("Layout model already in Volume.")

    vol.commit()
    print("Done.")


# ── Label maps ───────────────────────────────────────────────────────────────────

_ID2LABEL = {
    0: "abstract",        1: "algorithm",        2: "aside_text",
    3: "chart",           4: "content",          5: "display_formula",
    6: "doc_title",       7: "figure_title",     8: "footer",
    9: "footer_image",   10: "footnote",         11: "formula_number",
    12: "header",        13: "header_image",     14: "image",
    15: "inline_formula", 16: "number",          17: "paragraph_title",
    18: "reference",     19: "reference_content", 20: "seal",
    21: "table",         22: "text",             23: "vertical_text",
    24: "vision_footnote",
}

_MERGE_BBOXES_MODE = {i: "large" for i in range(25)}
_MERGE_BBOXES_MODE[18] = "small"

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

_TEXT_LABELS = frozenset({
    "abstract", "algorithm", "content", "doc_title", "figure_title",
    "paragraph_title", "reference_content", "text", "vertical_text",
    "vision_footnote", "seal", "formula_number",
})


# ── Layout detector ───────────────────────────────────────────────────────────────

@app.cls(
    gpu="L4",
    image=layout_image,
    volumes={WEIGHTS_PATH: vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=5,
    timeout=120,
)
class LayoutDetector:
    @modal.enter(snap=True)
    def load(self):
        import time
        from glmocr.config import LayoutConfig
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        t0 = time.time()
        cfg = LayoutConfig(
            model_dir=os.path.join(WEIGHTS_PATH, "PP-DocLayoutV3"),
            threshold=0.3,
            batch_size=4,
            layout_nms=True,
            layout_unclip_ratio=[1.0, 1.0],
            layout_merge_bboxes_mode=_MERGE_BBOXES_MODE,
            id2label=_ID2LABEL,
            label_task_mapping=_LABEL_TASK_MAPPING,
            device="cuda:0",
        )
        self.detector = PPDocLayoutDetector(cfg)
        self.detector.start()
        self._load_time_s = round(time.time() - t0, 2)
        print(f"[layout] model ready in {self._load_time_s:.2f}s")

    @modal.method()
    def detect(self, pdf_b64: str) -> dict:
        import io
        import time
        import pypdfium2 as pdfium
        import pdfplumber
        from PIL import Image

        t_start = time.time()

        RENDER_DPI  = 150
        MAX_SIDE_PX = 3500
        scale_render = RENDER_DPI / 72.0
        scale_pt     = 72.0 / RENDER_DPI

        # ── Step 1: decode + render ──────────────────────────────────────────────
        # Keep doc open — reused for pypdfium2 text extraction below.
        raw = base64.b64decode(pdf_b64)
        doc = pdfium.PdfDocument(raw)
        pages_pil: list[Image.Image] = []
        page_heights_pt: list[float] = []
        for page in doc:
            long_side = max(page.get_width(), page.get_height())
            s = min(scale_render, MAX_SIDE_PX / long_side)
            pages_pil.append(page.render(scale=s).to_pil())
            page_heights_pt.append(page.get_height())
        t_render = time.time()
        print(f"[layout] render {len(pages_pil)}p: {t_render - t_start:.3f}s")

        # ── Step 2: layout detection ─────────────────────────────────────────────
        all_results, _ = self.detector.process(pages_pil)
        t_detect = time.time()
        print(f"[layout] detect: {t_detect - t_render:.3f}s")

        # ── Step 3: build region dicts ───────────────────────────────────────────
        pages = []
        for page_idx, (page_img, regions) in enumerate(zip(pages_pil, all_results)):
            w_px, h_px = page_img.size
            page_regions = []
            for region in regions:
                x1_n, y1_n, x2_n, y2_n = region["bbox_2d"]
                x1 = int(x1_n / 1000.0 * w_px)
                y1 = int(y1_n / 1000.0 * h_px)
                x2 = int(x2_n / 1000.0 * w_px)
                y2 = int(y2_n / 1000.0 * h_px)
                page_regions.append({
                    "region_id":    f"p{page_idx}_r{region['index']}",
                    "bbox_px":      [x1, y1, x2, y2],
                    "bbox_pt":      [
                        round(x1 * scale_pt, 2), round(y1 * scale_pt, 2),
                        round(x2 * scale_pt, 2), round(y2 * scale_pt, 2),
                    ],
                    "label":        region["task_type"],
                    "native_label": region["native_label"] if "native_label" in region else region.get("label", ""),
                    "score":        region["score"],
                    "order":        region["index"],
                    "text":         None,
                })
            pages.append({"page_index": page_idx, "width_px": w_px, "height_px": h_px, "regions": page_regions})
        t_build = time.time()
        print(f"[layout] build dicts: {t_build - t_detect:.3f}s")

        # ── Step 4: text extraction via pypdfium2 (C-level, reuses open doc) ──────
        # pypdfium2's get_text_bounded is a C call per region — much faster than
        # pdfplumber which parses the character stream in Python.
        # Coordinates: pypdfium2 uses PDF space (origin bottom-left, y up).
        text_extracted = 0
        _pdfium_ok = False
        try:
            for page_idx, page_data in enumerate(pages):
                if page_idx >= len(doc):
                    break
                pdf_page = doc[page_idx]
                textpage = pdf_page.get_textpage()
                page_h_pt = page_heights_pt[page_idx]

                for region in page_data["regions"]:
                    if region["native_label"] not in _TEXT_LABELS:
                        continue
                    x1, y1, x2, y2 = region["bbox_px"]
                    # screen → PDF points, then flip y axis (PDF y goes up)
                    left   = x1 * scale_pt
                    right  = x2 * scale_pt
                    top_pdf    = page_h_pt - y1 * scale_pt   # screen top → pdf top
                    bottom_pdf = page_h_pt - y2 * scale_pt   # screen bottom → pdf bottom
                    text = textpage.get_text_bounded(
                        left=left, bottom=bottom_pdf, right=right, top=top_pdf
                    )
                    if text and text.strip():
                        region["text"] = text.strip()
                        text_extracted += 1
            _pdfium_ok = True
        except Exception as e:
            print(f"[layout] pypdfium2 text failed ({e}), falling back to pdfplumber")

        # Fallback: pdfplumber if pypdfium2 text extraction is unavailable
        if not _pdfium_ok:
            try:
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    for page_idx, page_data in enumerate(pages):
                        if page_idx >= len(pdf.pages):
                            break
                        pdf_page = pdf.pages[page_idx]
                        all_words = pdf_page.extract_words(x_tolerance=3, y_tolerance=3)
                        if not all_words:
                            continue
                        for region in page_data["regions"]:
                            if region["native_label"] not in _TEXT_LABELS:
                                continue
                            x1, y1, x2, y2 = region["bbox_px"]
                            x1_pt, y1_pt, x2_pt, y2_pt = (
                                x1 * scale_pt, y1 * scale_pt, x2 * scale_pt, y2 * scale_pt
                            )
                            rwords = [
                                w for w in all_words
                                if w["x0"] >= x1_pt - 2 and w["x1"] <= x2_pt + 2
                                and w["top"] >= y1_pt - 2 and w["bottom"] <= y2_pt + 2
                            ]
                            if rwords:
                                rwords.sort(key=lambda w: (round(w["top"]), w["x0"]))
                                lines: list[str] = []
                                line: list[str] = []
                                prev_top: float | None = None
                                for w in rwords:
                                    if prev_top is None or abs(w["top"] - prev_top) < 5:
                                        line.append(w["text"])
                                    else:
                                        lines.append(" ".join(line))
                                        line = [w["text"]]
                                    prev_top = w["top"]
                                if line:
                                    lines.append(" ".join(line))
                                text = "\n".join(lines).strip()
                                if text:
                                    region["text"] = text
                                    text_extracted += 1
            except Exception:
                pass

        doc.close()
        t_plumber = time.time()
        print(f"[layout] text extract ({text_extracted} regions, pdfium={'yes' if _pdfium_ok else 'no'}): {t_plumber - t_build:.3f}s")

        total = t_plumber - t_start
        print(f"[layout] TOTAL: {total:.3f}s for {len(pages)} pages")

        return {
            "pages": pages,
            "meta": {
                "total_regions":  sum(len(p["regions"]) for p in pages),
                "text_extracted": text_extracted,
                "timing": {
                    "load_s":    getattr(self, "_load_time_s", None),
                    "render_s":  round(t_render - t_start, 3),
                    "detect_s":  round(t_detect - t_render, 3),
                    "build_s":   round(t_build - t_detect, 3),
                    "plumber_s": round(t_plumber - t_build, 3),
                    "total_s":   round(total, 3),
                },
                "dpi": RENDER_DPI,
            },
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────────

try:
    from pydantic import BaseModel as _BaseModel
except ImportError:
    _BaseModel = object  # type: ignore[assignment,misc]


class _PDFRequest(_BaseModel):
    pdf_b64: str


@app.function(image=layout_image, timeout=300, scaledown_window=1000)
@modal.fastapi_endpoint(method="POST")
def process_pdf(request: _PDFRequest) -> dict:
    import time
    t0 = time.time()
    result = LayoutDetector().detect.remote(request.pdf_b64)
    elapsed = round(time.time() - t0, 2)
    result["meta"]["elapsed_s"] = elapsed
    return result


# ── CLI entrypoint ────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(pdf_path: str = ""):
    if not pdf_path:
        print("Usage: modal run modal/app.py --pdf-path /path/to/doc.pdf")
        return
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()
    result = process_pdf.remote(pdf_b64)
    m = result["meta"]
    t = m.get("timing", {})
    print(f"Pages          : {len(result['pages'])}")
    print(f"Total regions  : {m['total_regions']}")
    print(f"Text extracted : {m['text_extracted']}")
    print(f"Elapsed (wall) : {m['elapsed_s']}s")
    print(f"  render       : {t.get('render_s')}s")
    print(f"  detect       : {t.get('detect_s')}s")
    print(f"  build dicts  : {t.get('build_s')}s")
    print(f"  pdfplumber   : {t.get('plumber_s')}s")
    print(f"  total inside : {t.get('total_s')}s")
