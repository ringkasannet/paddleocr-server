"""Modal serverless deployment — PP-DocLayoutV3 layout detection (transformers backend).

Same interface as the RunPod layout worker:
  POST /  {"file": "<base64>", "fileType": 0|1, "dpi": 150}
  → {"pages": [...], "dpi": 150, "searchable": bool}

One-time setup:
  modal run modal/layout.py::download_weights

Deploy:
  modal deploy modal/layout.py

Test:
  modal run modal/layout.py --pdf-path /path/to/doc.pdf
"""
from __future__ import annotations

import base64
import os

import modal

app = modal.App("layout-worker")

GPU = os.environ.get("MODAL_GPU", "T4").upper()
GPU_RATES = {"T4": 0.000164, "L4": 0.000222}

vol = modal.Volume.from_name("layout-weights", create_if_missing=True)
WEIGHTS_PATH = "/weights"
MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"

layout_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .run_commands(
        "pip install --no-cache-dir uv",
        "uv pip install --system --no-cache "
        "'transformers>=5.3.0' torch torchvision opencv-python-headless pypdfium2 Pillow "
        "'huggingface_hub[hf_transfer]' 'fastapi[standard]'",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "pip install --no-cache-dir uv",
        "uv pip install --system --no-cache "
        "pypdfium2 Pillow numpy 'fastapi[standard]'",
    )
)

TEXT_REGION_TYPES = {
    "text", "title", "paragraph_title", "abstract",
    "content", "reference", "footnote", "doc_title",
    "header", "footer", "reference_content", "aside_text",
    "figure_title", "table_title", "chart_title",
    "number", "vision_footnote",
}


# ── Weight downloader (run once) ──────────────────────────────────────────────

@app.function(image=layout_image, volumes={WEIGHTS_PATH: vol}, timeout=1800)
def download_weights(hf_token: str = ""):
    from huggingface_hub import snapshot_download
    kwargs = {"token": hf_token} if hf_token else {}
    model_dir = os.path.join(WEIGHTS_PATH, "PP-DocLayoutV3")
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Downloading {MODEL_ID} ...")
        snapshot_download(MODEL_ID, local_dir=model_dir, **kwargs)
        print("Downloaded.")
    else:
        print("Already in Volume.")
    vol.commit()
    print("Done.")


# ── Layout detector ───────────────────────────────────────────────────────────

import threading
_gpu_lock = threading.Lock()  # serializes CUDA ops across concurrent threads in one container

@app.cls(
    gpu=GPU,
    image=layout_image,
    volumes={WEIGHTS_PATH: vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=60,    # keep GPU containers warm for 60s between requests
    timeout=120,
    max_containers=10,      # hard cap at hobby plan's 10 GPU slots
)
@modal.concurrent(max_inputs=4, target_inputs=3)
class LayoutDetector:
    @modal.enter(snap=True)
    def load(self):
        import torch
        import numpy as np
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        model_dir = os.path.join(WEIGHTS_PATH, "PP-DocLayoutV3")
        model_src = model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else MODEL_ID
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoImageProcessor.from_pretrained(model_src)
        self._model = AutoModelForObjectDetection.from_pretrained(model_src)
        self._model = self._model.to(self._device).eval()
        # Warm-up: trigger CUDA kernel compilation so it's captured in the snapshot.
        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        inputs = self._processor(images=[dummy], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            self._model(**inputs)
        print(f"[init] Model ready on {self._device} (kernels warm)")

    @modal.method()
    def detect(self, page_jpegs: list[bytes]) -> dict:
        """GPU-only: batch preprocess → inference → raw detections. No file I/O or text work."""
        import io
        import time
        import torch
        from PIL import Image

        THRESHOLD         = float(os.environ.get("DETECT_THRESHOLD",  "0.3"))
        HEADING_THRESHOLD = float(os.environ.get("HEADING_THRESHOLD", "0.2"))
        BATCH_SIZE        = int(os.environ.get("DETECT_BATCH_SIZE", "8"))
        HEADING_LABELS    = {"paragraph_title", "doc_title"}
        t0 = time.time()
        print(f"[detect] started  ts={t0:.3f}  pages={len(page_jpegs)}  batch_size={BATCH_SIZE}")

        # CPU: decode all pages and preprocess into chunks — runs concurrently across threads
        pil_images   = [Image.open(io.BytesIO(j)).convert("RGB") for j in page_jpegs]
        cpu_chunks   = []
        for chunk_start in range(0, len(pil_images), BATCH_SIZE):
            chunk = pil_images[chunk_start : chunk_start + BATCH_SIZE]
            cpu_inputs = self._processor(images=chunk, return_tensors="pt")
            cpu_chunks.append((chunk, cpu_inputs))

        # GPU: serialized via lock — prevents concurrent threads racing on CUDA allocator
        raw_pages = []
        with _gpu_lock:
            for chunk, cpu_inputs in cpu_chunks:
                gpu_inputs = {k: v.to(self._device) for k, v in cpu_inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**gpu_inputs)
                batch_detections = self._processor.post_process_object_detection(
                    outputs, threshold=min(THRESHOLD, HEADING_THRESHOLD),
                    target_sizes=[img.size[::-1] for img in chunk],
                )
                for pil_img, detections in zip(chunk, batch_detections):
                    page_detections = []
                    for score, label_id, box in zip(
                        detections["scores"], detections["labels"], detections["boxes"]
                    ):
                        label  = self._model.config.id2label[label_id.item()]
                        cutoff = HEADING_THRESHOLD if label in HEADING_LABELS else THRESHOLD
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
                        "width_px":  pil_img.width,
                        "height_px": pil_img.height,
                        "detections": page_detections,
                    })

        detect_s = round(time.time() - t0, 3)
        print(f"[detect] returning  detect_s={detect_s}s  pages={len(raw_pages)}")
        return {"pages": raw_pages, "_detect_start_ts": t0, "detect_s": detect_s}


# ── Processing helpers (module-level so they're importable in detect) ─────────

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
    import re
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
    text = re.sub(r"([a-z])([A-Z])", r"\1-\2", text)
    return text


import re as _re
_ID_REGULATION  = _re.compile(r'\b(BAB\s+[IVXLC]+|Pasal\s+\d+)\b')
_ID_BULLET      = _re.compile(r'^[a-z0-9]{1,3}[.)]\s')


def _is_indonesian_regulation(regions: list) -> bool:
    return any(
        r.get("type") == "paragraph_title"
        and _ID_REGULATION.fullmatch(r.get("text", "").strip())
        for r in regions
    )


def _demote_bullet_items(regions: list) -> list:
    """Demote paragraph_title → text for left-aligned bullet items in Indonesian regulations.
    Bullets: a. b. 1. 2. etc. at the start of the text (lowercase or digit, max 3 chars).
    """
    result = []
    for r in regions:
        if r["type"] == "paragraph_title" and _ID_BULLET.match(r.get("text", "")):
            r = dict(r)
            r["type"] = "text"
        result.append(r)
    return result


def _reclassify_centered_headings(regions: list, page_height: int,
                                  center_tol: float = 0.08,
                                  width_ratio: float = 0.25,
                                  bottom_margin: float = 0.92) -> list:
    """Reclassify mistyped regions as 'paragraph_title' when they are centered,
    narrow, and not near the page bottom.

    Two cases handled:
    - type='text': model under-confident on headings like 'Pasal 10'
    - type='figure_title'/'table_title'/'chart_title' NOT followed by their
      expected content type: a stray 'figure_title' with no figure after it
      is a misclassified heading, not a real figure caption
    """
    if not regions:
        return regions
    content_x0 = min(r["bbox"][0] for r in regions)
    content_x1 = max(r["bbox"][2] for r in regions)
    content_center = (content_x0 + content_x1) / 2
    content_width  = max(content_x1 - content_x0, 1)
    bottom_limit   = page_height * bottom_margin

    # Map each label-title type to the content type that should follow it
    _expected_next = {
        "figure_title": "figure",
        "table_title":  "table",
        "chart_title":  "figure",
    }

    result = []
    for i, r in enumerate(regions):
        rtype = r["type"]
        should_check = rtype == "text"
        if not should_check and rtype in _expected_next:
            expected = _expected_next[rtype]
            next_type = regions[i + 1]["type"] if i + 1 < len(regions) else None
            should_check = next_type != expected  # no matching content follows → likely misclassified

        if should_check:
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
    """Merge consecutive paragraph_title regions that are both centered with a small gap.

    Handles model splitting a chapter label from its subtitle, e.g.:
        BAB II                              ← narrow, centered
        NILAI LAIN SEBAGAI DASAR PENGENAAN  ← wide or narrow, centered
    Left-aligned list items (A., I.) are excluded because they fail the centering check.
    Chains of 3+ stacked titles are handled iteratively.
    """
    if len(regions) <= 1:
        return regions
    # Use content bounding box as reference — titles may not be centered on the
    # full page if the document has asymmetric margins.
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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    vals = boxes[:, axis::2].astype(int)
    proj = np.zeros(int(vals.max()) + 1, dtype=int)
    for lo, hi in vals:
        proj[lo:hi] += 1
    return proj


def _segments(arr, min_gap: int = 1):
    import numpy as np
    sig = np.where(arr > 0)[0]
    if not len(sig):
        return None
    gaps = np.where(np.diff(sig) > min_gap)[0]
    starts = np.concatenate([[sig[0]], sig[gaps + 1]])
    ends   = np.concatenate([sig[gaps], [sig[-1] + 1]])
    return starts, ends


# ── HTTP endpoint ─────────────────────────────────────────────────────────────

from pydantic import BaseModel

class _Request(BaseModel):
    file: str       # base64-encoded PDF or image bytes
    fileType: int = 0  # 0 = PDF, 1 = image
    dpi: int = 200

GPU_RATE_PER_S = GPU_RATES.get(GPU, GPU_RATES["T4"])
IDLE_WINDOW_S  = 5  # GPU scaledown_window (used for cost estimation)


@app.cls(
    image=cpu_image,
    timeout=300,
    scaledown_window=100,
    max_containers=50,
    enable_memory_snapshot=True,
    min_containers=1,
)
@modal.concurrent(max_inputs=20, target_inputs=10)
class Processor:
    @modal.enter(snap=True)
    def load(self):
        import pypdfium2
        from PIL import Image
        import numpy as np
        _ = (pypdfium2, Image, np)  # pre-import: C extensions captured in memory snapshot
        print("[processor/init] ready")

    @modal.fastapi_endpoint(method="POST")
    async def process(self, req: _Request) -> dict:
        import io
        import time
        import pypdfium2 as pdfium
        from PIL import Image

        t0 = time.time()
        print(f"[process] received  ts={t0:.3f}")

        # ── Parse JSON+base64 ────────────────────────────────────────────────
        raw_b64 = req.file
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            file_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            return {"error": f"Bad request: {e}"}
        file_type = req.fileType
        dpi       = req.dpi

        # ── CPU: render PDF → PIL pages ──────────────────────────────────────
        try:
            if file_type == 1:
                pil_pages = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
                pdf = None
                searchable_set: set[int] = set()
            else:
                pdf = pdfium.PdfDocument(file_bytes)
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

        # Serialize pages as JPEG for transfer to GPU container
        page_jpegs = []
        for img in pil_pages:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            page_jpegs.append(buf.getvalue())

        # ── GPU: model inference only (async — yields event loop while waiting) ──
        t_call = time.time()
        print(f"[process] calling detect.remote()  pages={len(page_jpegs)}")
        gpu_result = await LayoutDetector().detect.remote.aio(page_jpegs)
        t_gpu_done = time.time()
        print(f"[process] detect.remote() returned  ts={t_gpu_done:.3f}")

        if "error" in gpu_result:
            return gpu_result

        detect_start_ts = gpu_result.pop("_detect_start_ts", None)
        detect_s        = gpu_result.pop("detect_s", 0)
        queued_s        = round(detect_start_ts - t_call, 3) if detect_start_ts else None
        raw_pages       = gpu_result["pages"]

        # ── CPU: NMS + reading order ──────────────────────────────────────────
        # Detect regulation at document level — mid-article pages have no BAB/Pasal heading
        is_id_reg = any(
            _is_indonesian_regulation(p["detections"]) for p in raw_pages
        )
        result_pages = []
        for page_num, (page_data, _) in enumerate(zip(raw_pages, pil_pages)):
            regions = page_data["detections"]
            regions = _nms_regions(regions)
            regions = _reading_order(regions, page_data["width_px"])
            if is_id_reg:
                regions = _demote_bullet_items(regions)
            regions = _reclassify_centered_headings(regions, page_data["height_px"])
            regions = _merge_centered_titles(regions)
            for i, r in enumerate(regions):
                r["order"] = i
            result_pages.append({
                "page_num":  page_num,
                "width_px":  page_data["width_px"],
                "height_px": page_data["height_px"],
                "regions":   regions,
            })

        # ── CPU: text extraction ──────────────────────────────────────────────
        if pdf is not None:
            for page in result_pages:
                if page["page_num"] in searchable_set:
                    for region in page["regions"]:
                        if region["type"] in TEXT_REGION_TYPES:
                            region["text"] = _extract_text(
                                pdf, page["page_num"], region["bbox"], dpi
                            )
            pdf.close()
        t_text = time.time()

        page_count  = len(result_pages)
        render_s    = round(t_render - t0, 3)
        text_s      = round(t_text - t_gpu_done, 3)
        wall_s      = round(t_text - t0, 3)

        # ── Cost (GPU-only billed time) ───────────────────────────────────────
        execution_s     = detect_s
        billed_s_lower  = execution_s + IDLE_WINDOW_S
        billed_s_queued = round(queued_s + execution_s + IDLE_WINDOW_S, 3) if queued_s is not None else None
        billed_s_wall   = round(wall_s + IDLE_WINDOW_S, 3)

        return {
            "pages":      result_pages,
            "dpi":        dpi,
            "searchable": bool(searchable_set),
            "meta": {
                "page_count":    page_count,
                "total_regions": sum(len(p["regions"]) for p in result_pages),
                "timing": {
                    "queued_s":    queued_s,
                    "render_s":    render_s,
                    "detect_s":    detect_s,
                    "text_s":      text_s,
                    "execution_s": execution_s,
                    "wall_s":      wall_s,
                },
                "cost": {
                    "gpu":                   GPU,
                    "rate_per_s":            GPU_RATE_PER_S,
                    "execution_s":           execution_s,
                    "queued_s":              queued_s,
                    "idle_s":                IDLE_WINDOW_S,
                    "billed_s_lower":        round(billed_s_lower, 3),
                    "billed_s_queued":       billed_s_queued,
                    "billed_s_wall":         billed_s_wall,
                    "note":                  "queued estimate is most accurate; excludes only the first cold start after deploy",
                    "estimated_usd_lower":   round(billed_s_lower * GPU_RATE_PER_S, 6),
                    "per_page_usd_lower":    round(billed_s_lower * GPU_RATE_PER_S / page_count, 6) if page_count else None,
                    "estimated_usd_queued":  round(billed_s_queued * GPU_RATE_PER_S, 6) if billed_s_queued else None,
                    "per_page_usd_queued":   round(billed_s_queued * GPU_RATE_PER_S / page_count, 6) if billed_s_queued and page_count else None,
                    "estimated_usd_wall":    round(billed_s_wall * GPU_RATE_PER_S, 6),
                    "per_page_usd_wall":     round(billed_s_wall * GPU_RATE_PER_S / page_count, 6) if page_count else None,
                },
            },
        }


# ── CLI entrypoint ────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(pdf_path: str = ""):
    if not pdf_path:
        print("Usage: modal run modal/layout.py --pdf-path /path/to/doc.pdf")
        return
    import io
    import pypdfium2 as pdfium
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    pdf = pdfium.PdfDocument(file_bytes)
    page_jpegs = []
    for i in range(len(pdf)):
        pg = pdf[i]
        pil = pg.render(scale=150 / 72).to_pil()
        pg.close()
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        page_jpegs.append(buf.getvalue())
    pdf.close()
    result = LayoutDetector().detect.remote(page_jpegs)
    print(f"Pages     : {len(result['pages'])}")
    print(f"Searchable: {result['searchable']}")
    for p in result["pages"]:
        print(f"  Page {p['page_num']}: {len(p['regions'])} regions")
