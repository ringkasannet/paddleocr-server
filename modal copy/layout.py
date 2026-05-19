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

@app.cls(
    gpu=GPU,
    image=layout_image,
    volumes={WEIGHTS_PATH: vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=20,
    timeout=120,
)
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
    def detect(self, file_b64: str, file_type: int = 0, dpi: int = 150) -> dict:
        import io
        import re
        import time
        import numpy as np
        import pypdfium2 as pdfium
        import torch
        from PIL import Image

        THRESHOLD = float(os.environ.get("DETECT_THRESHOLD", "0.3"))
        t0 = time.time()
        print(f"[detect] started  ts={t0:.3f}")

        if "," in file_b64:
            file_b64 = file_b64.split(",", 1)[1]
        try:
            file_bytes = base64.b64decode(file_b64)
        except Exception as e:
            return {"error": f"Invalid base64: {e}"}

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
                    page = pdf[i]
                    pil_pages.append(page.render(scale=scale).to_pil())
                    page.close()
                searchable_set = _searchable_pages(pdf)
        except Exception as e:
            return {"error": f"Render failed: {e}"}
        t_render = time.time()

        result_pages = []
        for page_num, pil_img in enumerate(pil_pages):
            inputs = self._processor(images=[pil_img], return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            detections = self._processor.post_process_object_detection(
                outputs, threshold=THRESHOLD, target_sizes=[pil_img.size[::-1]]
            )[0]
            regions = []
            for score, label_id, box in zip(
                detections["scores"], detections["labels"], detections["boxes"]
            ):
                x0, y0, x1, y1 = box.tolist()
                regions.append({
                    "type":  self._model.config.id2label[label_id.item()],
                    "bbox":  [int(x0), int(y0), int(x1), int(y1)],
                    "score": round(score.item(), 4),
                    "text":  "",
                })
            regions = _nms_regions(regions)
            regions = _reading_order(regions, pil_img.width)
            for i, r in enumerate(regions):
                r["order"] = i
            result_pages.append({
                "page_num":  page_num,
                "width_px":  pil_img.width,
                "height_px": pil_img.height,
                "regions":   regions,
            })
        t_detect = time.time()

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

        execution_s = round(t_text - t0, 3)
        page_count  = len(result_pages)
        print(f"[detect] returning ts={t_text:.3f}  execution={execution_s}s  pages={page_count}")
        return {
            "pages":      result_pages,
            "dpi":        dpi,
            "searchable": bool(searchable_set),
            "meta": {
                "page_count":    page_count,
                "total_regions": sum(len(p["regions"]) for p in result_pages),
                "timing": {
                    "_detect_start_ts": t0,
                    "render_s":         round(t_render - t0, 3),
                    "detect_s":         round(t_detect - t_render, 3),
                    "text_s":           round(t_text - t_detect, 3),
                    "execution_s":      execution_s,
                },
            },
        }


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
    text = re.sub(r"([a-z])([A-Z])", r"\1-\2", text)
    return text


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

try:
    from pydantic import BaseModel as _BaseModel
except ImportError:
    _BaseModel = object  # type: ignore[assignment,misc]


class _Request(_BaseModel):
    file: str
    fileType: int = 0
    dpi: int = 150


GPU_RATE_PER_S = GPU_RATES.get(GPU, GPU_RATES["T4"])
IDLE_WINDOW_S  = 5  # scaledown_window on LayoutDetector


@app.function(image=layout_image, timeout=300, scaledown_window=1000)
@modal.fastapi_endpoint(method="POST")
def process(request: _Request) -> dict:
    import time
    t0 = time.time()
    print(f"[process] received  ts={t0:.3f}")
    t_call = time.time()
    print(f"[process] calling detect.remote()")
    result = LayoutDetector().detect.remote(request.file, request.fileType, request.dpi)
    t_done = time.time()
    print(f"[process] detect.remote() returned  ts={t_done:.3f}")
    wall_s = round(t_done - t0, 3)

    if "error" in result:
        return result

    meta        = result["meta"]
    timing      = meta["timing"]
    execution_s = timing["execution_s"]
    page_count  = meta["page_count"]

    detect_start_ts = timing.pop("_detect_start_ts", None)
    queued_s = round(detect_start_ts - t_call, 3) if detect_start_ts else None

    timing["queued_s"] = queued_s
    timing["wall_s"]   = wall_s

    # Three cost estimates (all exclude startup on first cold start after deploy):
    #   lower : execution + idle only — warm-container lower bound
    #   queued: queued + execution + idle — best estimate; queued_s ≈ snapshot restore time
    #   wall  : wall_s + idle — safe upper bound (includes RPC/network overhead)
    billed_s_lower  = execution_s + IDLE_WINDOW_S
    billed_s_queued = (queued_s + execution_s + IDLE_WINDOW_S) if queued_s is not None else None
    billed_s_wall   = wall_s + IDLE_WINDOW_S
    meta["cost"] = {
        "gpu":                   GPU,
        "rate_per_s":            GPU_RATE_PER_S,
        "execution_s":           execution_s,
        "queued_s":              queued_s,
        "idle_s":                IDLE_WINDOW_S,
        "billed_s_lower":        round(billed_s_lower, 3),
        "billed_s_queued":       round(billed_s_queued, 3) if billed_s_queued else None,
        "billed_s_wall":         round(billed_s_wall, 3),
        "note":                  "queued estimate is most accurate; excludes only the very first cold start after deploy",
        "estimated_usd_lower":   round(billed_s_lower * GPU_RATE_PER_S, 6),
        "per_page_usd_lower":    round(billed_s_lower * GPU_RATE_PER_S / page_count, 6) if page_count else None,
        "estimated_usd_queued":  round(billed_s_queued * GPU_RATE_PER_S, 6) if billed_s_queued else None,
        "per_page_usd_queued":   round(billed_s_queued * GPU_RATE_PER_S / page_count, 6) if billed_s_queued and page_count else None,
        "estimated_usd_wall":    round(billed_s_wall * GPU_RATE_PER_S, 6),
        "per_page_usd_wall":     round(billed_s_wall * GPU_RATE_PER_S / page_count, 6) if page_count else None,
    }
    return result


# ── CLI entrypoint ────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(pdf_path: str = ""):
    if not pdf_path:
        print("Usage: modal run modal/layout.py --pdf-path /path/to/doc.pdf")
        return
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()
    result = LayoutDetector().detect.remote(pdf_b64, 0, 150)
    print(f"Pages     : {len(result['pages'])}")
    print(f"Searchable: {result['searchable']}")
    for p in result["pages"]:
        print(f"  Page {p['page_num']}: {len(p['regions'])} regions")
