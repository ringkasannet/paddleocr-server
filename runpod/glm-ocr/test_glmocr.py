#!/usr/bin/env python3
"""
Benchmark for GLM-OCR self-hosted server (/glmocr/parse endpoint).
Each concurrency level gets its own non-overlapping slice of the PDF so
no page is processed more than once across all test runs.

Install deps first:
    pip install pypdf requests

Usage examples:
    # Default: 4 pages/chunk, levels 1 4 8
    python test_glmocr.py --url http://localhost:5002/glmocr/parse

    # Custom concurrency ladder
    python test_glmocr.py --url http://localhost:5002/glmocr/parse --levels 1 4 8 16

    # Override with fixed total chunk count instead
    python test_glmocr.py --url http://localhost:5002/glmocr/parse --chunks 40

    # Skip saving JSON responses
    python test_glmocr.py --url http://localhost:5002/glmocr/parse --no-save

    # Enable GPU sampling (only accurate when running this script ON the pod)
    python test_glmocr.py --url http://localhost:5002/glmocr/parse --gpu
"""
import argparse
import base64
import hashlib
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

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URL             = "http://localhost:5002/glmocr/parse"
DEFAULT_CHUNKS          = None
DEFAULT_PAGES_PER_CHUNK = 4
DEFAULT_TIMEOUT         = 300
DEFAULT_PDF_URL = (
    "https://jdih.kemenkeu.go.id/api/download/"
    "637047be-3dba-4347-aba1-98fa7fd5ab3f/2024pmkeuangan081.pdf"
)
OUT_DIR = Path(__file__).parent / "kemenkeu_glmocr"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="GLM-OCR benchmark")
    p.add_argument("--url", default=DEFAULT_URL,
                   help="GLM-OCR server endpoint (default: %(default)s)")
    p.add_argument("--pages-per-chunk", type=int, default=DEFAULT_PAGES_PER_CHUNK,
                   help="Pages per chunk (default: %(default)s); ignored when --chunks is given")
    p.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS,
                   help="Total chunk count split across all levels (overrides --pages-per-chunk)")
    p.add_argument("--levels", type=int, nargs="+",
                   help="Concurrency levels, e.g. --levels 1 4 8 16  (default: 1 4 8)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Per-request timeout in seconds (default: %(default)s)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving JSON responses")
    p.add_argument("--pdf", default=DEFAULT_PDF_URL,
                   help="PDF URL to download")
    p.add_argument("--gpu", action="store_true",
                   help="Sample GPU via nvidia-smi during each run (only accurate on the pod)")
    return p.parse_args()


# ── PDF helpers ───────────────────────────────────────────────────────────────

def download_pdf(url: str) -> bytes:
    key = hashlib.md5(url.encode()).hexdigest()[:8]
    cache = OUT_DIR / f"source_{key}.pdf"
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


def split_pdf_by_size(content: bytes, pages_per_chunk: int, max_chunks: int = None) -> list:
    reader = PdfReader(io.BytesIO(content))
    total = len(reader.pages)
    n = total // pages_per_chunk
    if max_chunks is not None:
        n = min(n, max_chunks)
    print(f"Splitting into {n} chunks of {pages_per_chunk} pages each "
          f"(PDF has {total} pages)")
    chunks = []
    for i in range(n):
        start = i * pages_per_chunk
        end = start + pages_per_chunk
        w = PdfWriter()
        for p in range(start, end):
            w.add_page(reader.pages[p])
        buf = io.BytesIO()
        w.write(buf)
        chunks.append((i + 1, start + 1, end, buf.getvalue()))
        print(f"  chunk_{i+1:02d}: pages {start+1}–{end}")
    return chunks


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


def distribute_chunks(all_chunks: list, levels: list, even: bool = False) -> dict:
    """Assign non-overlapping slices to each level — no page repeated across runs.

    Default (even=False): level 1 gets 8 chunks, level 4 gets 8, level 8 gets 8, etc.
    Ensures small levels still get enough chunks for a meaningful average.
    even=True: divide total evenly (used when --chunks is specified).
    """
    if even:
        n = len(levels)
        total = len(all_chunks)
        base = total // n
        remainder = total % n
        result = {}
        idx = 0
        for i, level in enumerate(levels):
            count = base + (1 if i < remainder else 0)
            result[level] = all_chunks[idx: idx + count]
            idx += count
        return result

    result = {}
    idx = 0
    for lv in sorted(levels):
        count = min(max(8, lv), len(all_chunks) - idx)
        result[lv] = all_chunks[idx: idx + count]
        idx += count
    return result


# ── Request ───────────────────────────────────────────────────────────────────

def query(chunk_bytes: bytes, endpoint: str, timeout: int, t0: float) -> tuple:
    b64 = base64.b64encode(chunk_bytes).decode()
    t_start = time.time()
    r = requests.post(
        endpoint,
        json={"images": [f"data:application/pdf;base64,{b64}"]},
        timeout=timeout,
    )
    payload_bytes = len(r.content)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json(), time.time() - t_start, t_start - t0, payload_bytes


def count_regions(result: dict) -> int:
    jr = result.get("json_result", {})
    if isinstance(jr, list):
        # json_result is a list-of-pages, each page is a list of regions
        return sum(len(page) for page in jr if isinstance(page, list))
    if isinstance(jr, dict):
        pages = jr.get("pages", [])
        if pages:
            return sum(len(p.get("regions", [])) for p in pages)
        return len(jr.get("regions", []))
    return 0


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
                "gpu_util":     int(util),
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
             endpoint: str, timeout: int, sample_gpu: bool = False) -> tuple:
    max_concurrent = min(max_concurrent, len(chunks))
    print(f"\n{'='*62}")
    print(f" {label}  (max_concurrent={max_concurrent})")
    print(f"{'='*62}")

    gpu_samples: list = []
    stop_event = threading.Event()
    if sample_gpu:
        threading.Thread(
            target=_gpu_sampler, args=(0.5, stop_event, gpu_samples), daemon=True
        ).start()

    t0 = time.time()
    results: dict = {}
    timings: dict = {}

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(query, data, endpoint, timeout, t0): (i, s, e)
            for i, s, e, data in chunks
        }
        for future in as_completed(futures):
            i, s, e = futures[future]
            try:
                result, elapsed, offset, payload_bytes = future.result()
                results[i] = result
                timings[i] = (s, e, elapsed, offset)
                n_regions = count_regions(result)
                spp = elapsed / (e - s + 1)
                kb = payload_bytes / 1024
                size_str = f"{kb:.0f} KB" if kb < 1024 else f"{kb/1024:.1f} MB"
                status = "OK" if n_regions > 0 else "EMPTY"
                print(f"  chunk_{i:02d}  pages {s:2d}–{e:2d}  {elapsed:6.1f}s  "
                      f"+{offset:5.1f}s  {spp:.1f} s/p  "
                      f"{n_regions:3d} regions  {size_str:>8}  [{status}]")
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
    if sample_gpu:
        print(f"\nGPU ({len(gpu_samples)} samples @ 0.5s):")
        print_gpu_summary(gpu_samples)

    return total, total_pages, results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    levels = sorted(set(args.levels if args.levels else [1, 4, 8]))

    content = download_pdf(args.pdf)
    if args.chunks is not None:
        chunks = split_pdf(content, args.chunks)
        level_chunks = distribute_chunks(chunks, levels, even=True)
    else:
        total_needed = sum(max(8, lv) for lv in levels)
        chunks = split_pdf_by_size(content, args.pages_per_chunk, max_chunks=total_needed)
        level_chunks = distribute_chunks(chunks, levels, even=False)

    print(f"\nEndpoint: {args.url}")
    print(f"Chunks  : {len(chunks)} total  ({args.pages_per_chunk} pages each)")
    print(f"Levels  : {levels}")
    for lv in levels:
        lc = level_chunks[lv]
        waves = len(lc) / lv if lv > 0 else len(lc)
        if lc:
            print(f"  level {lv:2d}  →  {len(lc):2d} chunks  {waves:.1f} waves  "
                  f"pages {lc[0][1]}–{lc[-1][2]}")
    print(f"Timeout : {args.timeout}s")

    all_results = {}
    summary_rows = []

    for level in levels:
        label = (
            "SEQUENTIAL (baseline)" if level == 1
            else f"{level} CONCURRENT"
        )
        total_time, total_pages, results = run_test(
            level_chunks[level], level, label, args.url, args.timeout,
            sample_gpu=args.gpu,
        )
        all_results.update(results)
        summary_rows.append((level, total_time, total_pages))

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
        print("\nSaving responses...")
        for i, _, _, _ in chunks:
            if i not in all_results:
                continue
            out = OUT_DIR / f"chunk_{i:02d}_response.json"
            out.write_text(
                json.dumps(all_results[i], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  {out.name}")


if __name__ == "__main__":
    main()
