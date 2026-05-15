#!/usr/bin/env python3
"""
Benchmark test for the ringkasannet/paddleocr-runpod:pod image (Section 4 deployment).
Compares sequential vs concurrent performance.

Usage:
    python test_pod.py

Set POD_URL to your RunPod proxy URL, e.g.:
    https://<pod-id>-8080.proxy.runpod.net
"""
import base64
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from pypdf import PdfReader, PdfWriter

# ── Config ────────────────────────────────────────────────────────────────────
POD_URL    = "https://eiqhnj3vhd6bkc-8080.proxy.runpod.net"
GATEWAY_URL = f"{POD_URL}/layout-parsing"

PDF_URL = (
    "https://jdih.kemenkeu.go.id/api/download/"
    "52955502-8733-4fdd-98ce-bb03c31cda0b/2025pmkeuangan11.pdf"
)
N_CHUNKS       = 8         # split 31 pages into 8 chunks (~4 pages each)
TIMEOUT        = 300       # Section 4 is slower — 5 min per chunk
OUT_DIR        = Path(__file__).parent / "kemenkeu"
# ─────────────────────────────────────────────────────────────────────────────


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
    return r.content


def split_pdf(content: bytes, n: int) -> list:
    reader = PdfReader(io.BytesIO(content))
    total = len(reader.pages)
    size = math.ceil(total / n)
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
    return chunks


def query(chunk_bytes: bytes, t0: float) -> tuple:
    t_start = time.time()
    r = requests.post(
        GATEWAY_URL,
        json={
            "file": base64.b64encode(chunk_bytes).decode(),
            "fileType": 0,
            "visualize": False,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json(), time.time() - t_start, t_start - t0


def run_test(chunks, max_concurrent: int, label: str):
    print(f"\n{'='*60}")
    print(f" {label}  (max_concurrent={max_concurrent})")
    print(f"{'='*60}")
    t0 = time.time()
    timings = {}
    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(query, data, t0): (i, s, e)
            for i, s, e, data in chunks
        }
        for future in as_completed(futures):
            i, s, e = futures[future]
            try:
                _, elapsed, offset = future.result()
                timings[i] = (s, e, elapsed, offset)
                pps = (e - s + 1) / elapsed
                print(f"  chunk_{i:02d}  pages {s:2d}–{e:2d}  {elapsed:6.1f}s  +{offset:.1f}s  {pps:.1f} p/s")
            except Exception as ex_:
                print(f"  chunk_{i:02d} FAILED: {ex_}")
                timings[i] = (s, e, 0, 0)

    total = time.time() - t0
    total_pages = sum(e - s + 1 for _, s, e, _ in chunks)
    print(f"{'─'*60}")
    print(f"  TOTAL  {total_pages} pages  {total:.1f}s  {total_pages/total:.2f} pages/s")
    return total, total_pages


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "REPLACE_ME" in POD_URL:
        print("ERROR: Set POD_URL at the top of this file.")
        return

    # Health check
    try:
        r = requests.get(f"{POD_URL}/health", timeout=10)
        print(f"Health: {r.status_code} — {r.text[:80]}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    content = download_pdf(PDF_URL)
    chunks = split_pdf(content, N_CHUNKS)
    print(f"Split into {len(chunks)} chunks")

    # Test 1: sequential (true Section 4 behavior — 1 at a time)
    run_test(chunks, max_concurrent=1, label="SEQUENTIAL (Section 4 baseline)")

    # Test 2: 4 concurrent (shows queuing penalty)
    run_test(chunks, max_concurrent=4, label="4 CONCURRENT (queuing)")

    # Test 3: all concurrent (maximum throughput attempt)
    run_test(chunks, max_concurrent=len(chunks), label=f"{len(chunks)} CONCURRENT (all at once)")


if __name__ == "__main__":
    main()
