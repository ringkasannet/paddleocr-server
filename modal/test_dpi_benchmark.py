"""Benchmark DPI strategies across Modal and ZAI endpoints.

Runs all four approaches concurrently, with configurable concurrent requests
per approach, and reports min/avg/max per approach.

Usage:
    python modal/test_dpi_benchmark.py page32.pdf
    python modal/test_dpi_benchmark.py page32.pdf --concurrent 3
    python modal/test_dpi_benchmark.py page32.pdf --concurrent 3 --repeat 2
    python modal/test_dpi_benchmark.py page32.pdf --page 0 --concurrent 5
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

MODAL_ENDPOINT = "https://ringkasan-net--glm-ocr-single-documentocrworker-process.modal.run"
ZAI_ENDPOINT   = "https://api.z.ai/api/paas/v4/layout_parsing"
ZAI_API_KEY    = os.environ.get("ZAI_API_KEY", "")

DETECT_PHRASE  = "Pasal 11"


# ── helpers ────────────────────────────────────────────────────────────────────

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _render_jpeg(pdf_bytes: bytes, page: int, dpi: int) -> tuple[bytes, float]:
    import pymupdf
    from PIL import Image
    t0    = time.time()
    doc   = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pg    = doc.load_page(page)
    scale = dpi / 72.0
    pix   = pg.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    pil   = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), round(time.time() - t0, 3)


def _detected(data: dict) -> bool:
    return DETECT_PHRASE in json.dumps(data, ensure_ascii=False)


# ── single-run approach functions ──────────────────────────────────────────────

def run_modal(pdf_bytes: bytes, dpi: int, page: int) -> dict:
    t0 = time.time()
    resp = requests.post(
        MODAL_ENDPOINT,
        json={"file": _b64(pdf_bytes), "pages": [page], "dpi": dpi},
        timeout=600,
    )
    wall   = round(time.time() - t0, 3)
    data   = resp.json()
    timing = data.get("meta", {}).get("timing", {})
    return {
        "status":    resp.status_code,
        "wall_s":    wall,
        "server_s":  timing.get("total_s"),
        "convert_s": None,
        "api_s":     None,
        "regions":   data.get("meta", {}).get("total_regions"),
        "detected":  _detected(data),
    }


def run_zai_pdf(pdf_bytes: bytes, _page: int) -> dict:
    t0 = time.time()
    resp = requests.post(
        ZAI_ENDPOINT,
        headers={"Authorization": f"Bearer {ZAI_API_KEY}"},
        json={"model": "glm-ocr", "file": "data:application/pdf;base64," + _b64(pdf_bytes)},
        timeout=300,
    )
    wall = round(time.time() - t0, 3)
    data = resp.json()
    return {
        "status":    resp.status_code,
        "wall_s":    wall,
        "server_s":  None,
        "convert_s": None,
        "api_s":     wall,
        "regions":   len(data.get("layout_details", [[]])[0]),
        "detected":  _detected(data),
    }


def run_zai_jpeg(pdf_bytes: bytes, page: int, dpi: int) -> dict:
    jpeg_bytes, convert_s = _render_jpeg(pdf_bytes, page, dpi)
    t0 = time.time()
    resp = requests.post(
        ZAI_ENDPOINT,
        headers={"Authorization": f"Bearer {ZAI_API_KEY}"},
        json={"model": "glm-ocr", "file": "data:image/jpeg;base64," + _b64(jpeg_bytes)},
        timeout=300,
    )
    api_s = round(time.time() - t0, 3)
    data  = resp.json()
    return {
        "status":    resp.status_code,
        "wall_s":    round(convert_s + api_s, 3),
        "server_s":  None,
        "convert_s": convert_s,
        "api_s":     api_s,
        "regions":   len(data.get("layout_details", [[]])[0]),
        "detected":  _detected(data),
    }


APPROACHES: list[tuple[str, callable]] = [
    ("Modal  200 DPI  (raw PDF)", lambda pdf, pg: run_modal(pdf, 200, pg)),
    ("Modal  300 DPI  (raw PDF)", lambda pdf, pg: run_modal(pdf, 300, pg)),
    ("ZAI    raw PDF",            lambda pdf, pg: run_zai_pdf(pdf, pg)),
    ("ZAI    300 DPI JPEG",       lambda pdf, pg: run_zai_jpeg(pdf, pg, 300)),
]


# ── stats ──────────────────────────────────────────────────────────────────────

def _stats(lst: list[float]) -> tuple[float, float, float]:
    return min(lst), sum(lst) / len(lst), max(lst)


def _f(v, w=7) -> str:
    return f"{v:>{w}.2f}s" if isinstance(v, (int, float)) else f"{'—':>{w}}"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--page",       type=int, default=0)
    ap.add_argument("--concurrent", type=int, default=1, help="concurrent requests per approach")
    ap.add_argument("--repeat",     type=int, default=1, help="rounds (each round fires --concurrent per approach)")
    args = ap.parse_args()

    if not ZAI_API_KEY:
        print("ERROR: ZAI_API_KEY not set"); return

    pdf_bytes  = Path(args.file).read_bytes()
    n          = args.concurrent
    total_each = n * args.repeat

    print(f"File       : {args.file}  ({len(pdf_bytes):,} bytes)  page={args.page}")
    print(f"Concurrent : {n} per approach  |  Rounds: {args.repeat}  |  Total per approach: {total_each}")
    print(f"Detect     : {DETECT_PHRASE!r}")
    print(f"Approaches : {len(APPROACHES)}\n")

    # results[label] = list of result dicts (one per individual call)
    results: dict[str, list[dict]] = {label: [] for label, _ in APPROACHES}

    for r in range(args.repeat):
        print(f"Round {r+1}/{args.repeat} — firing {n} × {len(APPROACHES)} requests concurrently …")
        t_round = time.time()

        futures: dict = {}
        with ThreadPoolExecutor(max_workers=n * len(APPROACHES)) as pool:
            for label, fn in APPROACHES:
                for i in range(n):
                    fut = pool.submit(fn, pdf_bytes, args.page)
                    futures[fut] = (label, i)

            for fut in as_completed(futures):
                label, i = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"status": None, "wall_s": None, "server_s": None,
                           "convert_s": None, "api_s": None, "regions": None,
                           "detected": False, "error": str(e)}
                results[label].append(res)

        print(f"  Round done in {time.time()-t_round:.2f}s\n")

    # ── print summary table ────────────────────────────────────────────────────
    W = 8
    def _fw(v): return f"{v:>{W}.2f}s" if isinstance(v, (int, float)) else f"{'—':>{W}}"

    print(f"\n{'═'*100}")
    print(f"  RESULTS — {DETECT_PHRASE!r}  ({total_each} calls each)")
    print(f"{'═'*100}")
    print(f"  {'approach':<28}  {'detected':>10}  {'regions':>7}  {'conv avg':>{W}}  {'server avg':>{W}}  {'wall min':>{W}}  {'wall avg':>{W}}  {'wall max':>{W}}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*7}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}")

    for label, _ in APPROACHES:
        runs      = results[label]
        ok_runs   = [r for r in runs if r.get("status") == 200]
        det_count = sum(1 for r in ok_runs if r["detected"])
        det_str   = f"{det_count}/{len(runs)}"

        regions   = next((r["regions"] for r in ok_runs if r.get("regions")), None)

        walls     = [r["wall_s"]    for r in ok_runs if r.get("wall_s")    is not None]
        servers   = [r["server_s"]  for r in ok_runs if r.get("server_s")  is not None]
        convs     = [r["convert_s"] for r in ok_runs if r.get("convert_s") is not None]

        wall_min, wall_avg, wall_max = _stats(walls)   if walls   else (None, None, None)
        _, serv_avg, _               = _stats(servers) if servers else (None, None, None)
        _, conv_avg, _               = _stats(convs)   if convs   else (None, None, None)

        print(
            f"  {label:<28}  {det_str:>10}  {regions or '?':>7}  "
            f"{_fw(conv_avg)}  {_fw(serv_avg)}  "
            f"{_fw(wall_min)}  {_fw(wall_avg)}  {_fw(wall_max)}"
        )

        # per-call breakdown if errors occurred
        errors = [r for r in runs if r.get("error")]
        if errors:
            print(f"    {len(errors)} error(s): {errors[0].get('error', '')[:80]}")

    print(f"{'═'*100}\n")

    # ── ZAI JPEG breakdown ────────────────────────────────────────────────────
    zai_jpeg_label = "ZAI    300 DPI JPEG"
    zai_runs = [r for r in results.get(zai_jpeg_label, []) if r.get("status") == 200]
    if zai_runs:
        convs = [r["convert_s"] for r in zai_runs if r.get("convert_s") is not None]
        apis  = [r["api_s"]     for r in zai_runs if r.get("api_s")     is not None]
        if convs and apis:
            print(f"  ZAI 300 DPI JPEG breakdown:")
            print(f"    convert  avg={sum(convs)/len(convs):.2f}s  min={min(convs):.2f}s  max={max(convs):.2f}s")
            print(f"    api      avg={sum(apis)/len(apis):.2f}s  min={min(apis):.2f}s  max={max(apis):.2f}s\n")


if __name__ == "__main__":
    main()
