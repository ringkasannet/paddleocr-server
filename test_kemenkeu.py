#!/usr/bin/env python3
"""
Benchmark test for the HPS stack (ringkasannet/paddleocr-hps image).
Compares sequential vs concurrent performance on Indonesian PMK finance PDF (31 pages).

Install deps first:
    pip install pypdf requests
"""
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

# ── Config ────────────────────────────────────────────────────────────────────
GATEWAY_URL = "https://pm5arbkgbnmz6c-8080.proxy.runpod.net/layout-parsing"
PDF_URL = (
    "https://jdih.kemenkeu.go.id/api/download/"
    "52955502-8733-4fdd-98ce-bb03c31cda0b/2025pmkeuangan11.pdf"
)
N_CHUNKS = 10
TIMEOUT  = 300
OUT_DIR  = Path(__file__).parent / "kemenkeu"
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
    print(f"  {len(r.content):,} bytes (cached)")
    return r.content


def split_pdf(content: bytes, n: int) -> list:
    reader = PdfReader(io.BytesIO(content))
    total = len(reader.pages)
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
    elapsed = time.time() - t_start
    return r.json(), elapsed, t_start - t0


def sample_gpu(interval=1.0, stop_event=None, samples=None):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free",
                 "--format=csv,noheader,nounits"], text=True
            ).strip()
            util, mem_used, mem_free = out.split(", ")
            samples.append({
                "t": time.time(),
                "gpu_util": int(util),
                "mem_used_mib": int(mem_used),
                "mem_free_mib": int(mem_free),
            })
        except Exception:
            pass
        time.sleep(interval)


def print_gpu_summary(samples):
    if not samples:
        return
    utils = [s["gpu_util"] for s in samples]
    mems = [s["mem_used_mib"] for s in samples]
    print(f"\nGPU during inference ({len(samples)} samples):")
    print(f"  Utilization: avg={sum(utils)/len(utils):.0f}%  "
          f"min={min(utils)}%  max={max(utils)}%")
    print(f"  Memory used: avg={sum(mems)/len(mems):.0f} MiB  "
          f"min={min(mems)} MiB  max={max(mems)} MiB")


def run_test(chunks, max_concurrent: int, label: str) -> tuple:
    print(f"\n{'='*60}")
    print(f" {label}  (max_concurrent={max_concurrent})")
    print(f"{'='*60}")
    t0 = time.time()
    results = {}
    timings = {}

    gpu_samples = []
    stop_event = threading.Event()
    threading.Thread(
        target=sample_gpu, args=(0.5, stop_event, gpu_samples), daemon=True
    ).start()

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(query, data, t0): (i, s, e)
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
                      f"+{offset:.1f}s  {pps:.1f} p/s  {n_blocks} blocks")
            except Exception as ex_:
                print(f"  chunk_{i:02d} FAILED: {ex_}")
                results[i] = {"error": str(ex_)}
                timings[i] = (s, e, 0, 0)

    stop_event.set()
    total = time.time() - t0
    total_pages = sum(e - s + 1 for _, s, e, _ in chunks)
    print(f"{'─'*60}")
    print(f"  TOTAL  {total_pages} pages  {total:.1f}s  {total_pages/total:.2f} pages/s")
    print_gpu_summary(gpu_samples)
    return total, total_pages, results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    content = download_pdf(PDF_URL)
    chunks = split_pdf(content, N_CHUNKS)

    # Test 1: sequential (baseline)
    run_test(chunks, max_concurrent=1, label="SEQUENTIAL (baseline)")

    # Test 2: 4 concurrent
    run_test(chunks, max_concurrent=4, label="4 CONCURRENT (queuing)")

    # Test 3: all concurrent — save responses from this run
    _, _, results = run_test(
        chunks,
        max_concurrent=len(chunks),
        label=f"{len(chunks)} CONCURRENT (all at once)",
    )

    print("\nSaving responses from full-concurrent run...")
    for i, _, _, _ in chunks:
        out = OUT_DIR / f"chunk_{i:02d}_response.json"
        out.write_text(
            json.dumps(results.get(i, {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  {out.name}")


if __name__ == "__main__":
    main()
