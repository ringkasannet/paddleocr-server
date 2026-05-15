#!/usr/bin/env python3
"""
Benchmark test for the HPS stack (ringkasannet/paddleocr-hps image).
Compares multiple concurrency levels on Indonesian PMK finance PDF (31 pages).

Install deps first:
    pip install pypdf requests

Usage examples:
    # Default: sequential → 4 → all-at-once
    python test_kemenkeu.py --url http://localhost:8080/layout-parsing

    # Custom concurrency ladder
    python test_kemenkeu.py --url http://localhost:8080/layout-parsing --levels 1 4 8 16 32

    # Split into more chunks for higher concurrency tests
    python test_kemenkeu.py --url http://localhost:8080/layout-parsing --chunks 31 --levels 1 8 16 31

    # Skip sequential baseline (slow) and go straight to concurrent
    python test_kemenkeu.py --url http://localhost:8080/layout-parsing --levels 4 8 16 --no-save
"""
import argparse
import base64
import io
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from pypdf import PdfReader, PdfWriter

# ── Defaults (all overridable via CLI) ───────────────────────────────────────
DEFAULT_URL     = "https://pm5arbkgbnmz6c-8080.proxy.runpod.net/layout-parsing"
DEFAULT_CHUNKS  = 10
DEFAULT_TIMEOUT = 300
DEFAULT_PDF_URL = (
    "https://jdih.kemenkeu.go.id/api/download/"
    "52955502-8733-4fdd-98ce-bb03c31cda0b/2025pmkeuangan11.pdf"
)
OUT_DIR = Path(__file__).parent / "kemenkeu"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="PaddleOCR HPS benchmark")
    p.add_argument("--url", default=DEFAULT_URL,
                   help="Gateway endpoint URL (default: %(default)s)")
    p.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS,
                   help="Number of PDF chunks (default: %(default)s, max=31 for this PDF)")
    p.add_argument("--levels", type=int, nargs="+",
                   help="Concurrency levels to test, e.g. --levels 1 4 8 16 32  "
                        "(default: 1, 4, <chunks>)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Per-request timeout in seconds (default: %(default)s)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving JSON responses")
    p.add_argument("--pdf", default=DEFAULT_PDF_URL,
                   help="PDF URL to download (default: kemenkeu PMK 11/2025)")
    return p.parse_args()


# ── PDF helpers ───────────────────────────────────────────────────────────────

def download_pdf(url: str) -> bytes:
    cache = OUT_DIR / "source.pdf"
    if cache.exists():
        print(f"Using cached PDF ({cache.stat().st_size:,} bytes)")
        return cache.read_bytes()
    print("Downloading PDF...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    print(f"  {len(r.content):,} bytes (cached)")
    return r.content


def split_pdf(content: bytes, n: int) -> list:
    reader = PdfReader(io.BytesIO(content))
    total = len(reader.pages)
    n = min(n, total)
    size = math.ceil(total / n)
    print(f"Splitting {total} pages into {n} chunks (~{size} pages each)")
    chunks = []
    for i in range(n):
        start = i * size
        end = min(start + size, total)
        if start >= total:
            break
        w = PdfWriter()
        for p in range(start, end):
            w.add_page(reader.pages[p])
        buf = io.BytesIO()
        w.write(buf)
        chunks.append((i + 1, start + 1, end, buf.getvalue()))
        print(f"  chunk_{i+1:02d}: pages {start+1}–{end}")
    return chunks


# ── Request ───────────────────────────────────────────────────────────────────

def query(chunk_bytes: bytes, gateway_url: str, timeout: int, t0: float) -> tuple:
    t_start = time.time()
    r = requests.post(
        gateway_url,
        json={
            "file": base64.b64encode(chunk_bytes).decode(),
            "fileType": 0,
            "visualize": False,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json(), time.time() - t_start, t_start - t0


# ── GPU sampling ──────────────────────────────────────────────────────────────

def _gpu_sampler(interval: float, stop_event: threading.Event, samples: list):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            util, mem_used, mem_free, mem_total = out.split(", ")
            samples.append({
                "t": time.time(),
                "gpu_util":    int(util),
                "mem_used_mib": int(mem_used),
                "mem_free_mib": int(mem_free),
                "mem_total_mib": int(mem_total),
            })
        except Exception:
            pass
        time.sleep(interval)


def print_gpu_summary(samples: list):
    if not samples:
        print("  (no GPU samples — nvidia-smi not available)")
        return
    utils = [s["gpu_util"] for s in samples]
    mems  = [s["mem_used_mib"] for s in samples]
    total = samples[0]["mem_total_mib"]
    peak_pct = max(mems) / total * 100 if total else 0
    print(f"  GPU util : avg={sum(utils)/len(utils):.0f}%  "
          f"min={min(utils)}%  max={max(utils)}%")
    print(f"  VRAM used: avg={sum(mems)/len(mems):.0f} MiB  "
          f"peak={max(mems)} MiB / {total} MiB  ({peak_pct:.0f}% of total)")


# ── Single benchmark run ──────────────────────────────────────────────────────

def run_test(chunks: list, max_concurrent: int, label: str,
             gateway_url: str, timeout: int) -> tuple:
    max_concurrent = min(max_concurrent, len(chunks))
    print(f"\n{'='*62}")
    print(f" {label}  (max_concurrent={max_concurrent})")
    print(f"{'='*62}")

    gpu_samples: list = []
    stop_event = threading.Event()
    threading.Thread(
        target=_gpu_sampler, args=(0.5, stop_event, gpu_samples), daemon=True
    ).start()

    t0 = time.time()
    results: dict = {}
    timings: dict = {}

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(query, data, gateway_url, timeout, t0): (i, s, e)
            for i, s, e, data in chunks
        }
        for future in as_completed(futures):
            i, s, e = futures[future]
            try:
                result, elapsed, offset = future.result()
                results[i] = result
                timings[i] = (s, e, elapsed, offset)
                n_blocks = sum(
                    len(p.get("prunedResult", {}).get("parsing_res_list", []))
                    for p in result.get("result", {}).get("layoutParsingResults", [])
                )
                pps = (e - s + 1) / elapsed
                print(f"  chunk_{i:02d}  pages {s:2d}–{e:2d}  {elapsed:6.1f}s  "
                      f"+{offset:5.1f}s  {pps:.2f} p/s  {n_blocks} blocks")
            except Exception as ex_:
                print(f"  chunk_{i:02d} FAILED: {ex_}")
                results[i] = {"error": str(ex_)}
                timings[i] = (s, e, 0, 0)

    stop_event.set()
    total = time.time() - t0
    total_pages = sum(e - s + 1 for _, s, e, _ in chunks)
    pps_total = total_pages / total if total > 0 else 0

    print(f"{'─'*62}")
    print(f"  TOTAL  {total_pages} pages  {total:.1f}s  {pps_total:.2f} pages/s")
    print(f"\nGPU ({len(gpu_samples)} samples @ 0.5s):")
    print_gpu_summary(gpu_samples)

    return total, total_pages, results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    content = download_pdf(args.pdf)
    chunks  = split_pdf(content, args.chunks)

    # Build concurrency ladder
    levels = args.levels if args.levels else [1, 4, len(chunks)]
    # Remove duplicates, sort, cap at chunk count
    levels = sorted(set(min(lv, len(chunks)) for lv in levels))

    print(f"\nGateway : {args.url}")
    print(f"Chunks  : {len(chunks)}")
    print(f"Levels  : {levels}")
    print(f"Timeout : {args.timeout}s")

    last_results = {}
    summary_rows = []

    for level in levels:
        label = (
            "SEQUENTIAL (baseline)" if level == 1
            else f"{level} CONCURRENT"
        )
        total_time, total_pages, results = run_test(
            chunks, level, label, args.url, args.timeout
        )
        last_results = results
        summary_rows.append((level, total_time, total_pages))

    # Summary table across all runs
    print(f"\n{'='*62}")
    print(f" SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Concurrency':>12}  {'Total time':>12}  {'Pages/s':>10}  {'vs seq':>8}")
    baseline_pps = None
    for level, t, pages in summary_rows:
        pps = pages / t if t > 0 else 0
        if baseline_pps is None:
            baseline_pps = pps
        speedup = pps / baseline_pps if baseline_pps else 0
        print(f"  {level:>12}  {t:>10.1f}s  {pps:>10.2f}  {speedup:>7.2f}×")

    if not args.no_save:
        print("\nSaving responses from last run...")
        for i, _, _, _ in chunks:
            out = OUT_DIR / f"chunk_{i:02d}_response.json"
            out.write_text(
                json.dumps(last_results.get(i, {}), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  {out.name}")


if __name__ == "__main__":
    main()
