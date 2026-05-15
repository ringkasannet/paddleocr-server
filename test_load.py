#!/usr/bin/env python3
"""
Load test: simulate N concurrent users hitting the HPS gateway.
Uses asyncio + aiohttp for true async concurrency (no thread overhead).

Install deps:
    pip install aiohttp pypdf

Usage examples:
    # 100 users all at once (hammer test)
    python test_load.py --url http://localhost:8080 --users 100

    # 1000 users ramped over 60 seconds (realistic traffic)
    python test_load.py --url http://localhost:8080 --users 1000 --ramp 60

    # Use a smaller chunk (faster per-request, shows queue behavior more clearly)
    python test_load.py --url http://localhost:8080 --users 200 --pages 2
"""
import argparse
import asyncio
import base64
import io
import json
import math
import statistics
import time
from pathlib import Path

import aiohttp
import requests
from pypdf import PdfReader, PdfWriter

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URL     = "http://localhost:8080"
DEFAULT_PDF_URL = (
    "https://jdih.kemenkeu.go.id/api/download/"
    "52955502-8733-4fdd-98ce-bb03c31cda0b/2025pmkeuangan11.pdf"
)
OUT_DIR = Path(__file__).parent / "kemenkeu"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="HPS gateway load test")
    p.add_argument("--url", default=DEFAULT_URL,
                   help="Pod base URL (default: %(default)s)")
    p.add_argument("--users", type=int, default=100,
                   help="Number of concurrent users / total requests (default: %(default)s)")
    p.add_argument("--ramp", type=float, default=0,
                   help="Ramp-up duration in seconds. 0 = all at once (default: %(default)s)")
    p.add_argument("--pages", type=int, default=4,
                   help="Pages per request chunk (default: %(default)s)")
    p.add_argument("--timeout", type=int, default=700,
                   help="Per-request timeout in seconds (default: %(default)s)")
    p.add_argument("--pdf", default=DEFAULT_PDF_URL,
                   help="PDF URL to use as load test payload")
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


def extract_pages(content: bytes, start: int, n_pages: int) -> bytes:
    """Extract n_pages starting from start (0-indexed) as a PDF chunk."""
    reader = PdfReader(io.BytesIO(content))
    total = len(reader.pages)
    end = min(start + n_pages, total)
    w = PdfWriter()
    for p in range(start, end):
        w.add_page(reader.pages[p])
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


# ── Async request ─────────────────────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    gateway_url: str,
    payload: dict,
    user_id: int,
    t0: float,
    results: list,
):
    start = time.time()
    offset = start - t0
    status = 0
    error = None
    try:
        async with session.post(gateway_url, json=payload) as resp:
            status = resp.status
            await resp.json()  # consume body
    except asyncio.TimeoutError:
        error = "client_timeout"
        status = 0
    except Exception as e:
        error = str(e)[:80]
        status = 0

    elapsed = time.time() - start
    results.append({
        "user_id": user_id,
        "status":  status,
        "elapsed": elapsed,
        "offset":  offset,
        "error":   error,
    })

    symbol = "✓" if status == 200 else ("T" if error == "client_timeout" else "✗")
    print(f"  [{symbol}] user_{user_id:04d}  {elapsed:6.1f}s  +{offset:6.1f}s  HTTP {status}"
          + (f"  {error}" if error else ""))


# ── Main load test ────────────────────────────────────────────────────────────

async def run_load_test(
    gateway_url: str,
    chunk_b64: str,
    n_users: int,
    ramp_seconds: float,
    timeout_seconds: int,
):
    payload = {"file": chunk_b64, "fileType": 0, "visualize": False}
    results: list = []

    connector = aiohttp.TCPConnector(limit=0)  # no connection limit client-side
    client_timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    t0 = time.time()
    print(f"\nFiring {n_users} requests"
          + (f" ramped over {ramp_seconds}s" if ramp_seconds else " all at once")
          + "...")
    print(f"{'─'*62}")

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        if ramp_seconds <= 0:
            # All at once
            tasks = [
                asyncio.create_task(
                    send_request(session, gateway_url, payload, i, t0, results)
                )
                for i in range(1, n_users + 1)
            ]
            await asyncio.gather(*tasks)
        else:
            # Ramp: spread requests evenly over ramp_seconds
            interval = ramp_seconds / n_users
            tasks = []
            for i in range(1, n_users + 1):
                task = asyncio.create_task(
                    send_request(session, gateway_url, payload, i, t0, results)
                )
                tasks.append(task)
                await asyncio.sleep(interval)
            await asyncio.gather(*tasks)

    wall_time = time.time() - t0
    return results, wall_time


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(results: list, wall_time: float, n_users: int, pages_per_req: int):
    ok      = [r for r in results if r["status"] == 200]
    timeout = [r for r in results if r["error"] == "client_timeout"]
    errors  = [r for r in results if r["status"] != 200 and r["error"] != "client_timeout"]

    print(f"\n{'='*62}")
    print(f" LOAD TEST SUMMARY  ({n_users} users, {pages_per_req} pages/req)")
    print(f"{'='*62}")
    print(f"  Wall time       : {wall_time:.1f}s")
    print(f"  Successful      : {len(ok)} / {n_users}  "
          f"({100*len(ok)/n_users:.0f}%)")
    print(f"  Timed out       : {len(timeout)}")
    print(f"  Errors          : {len(errors)}")

    if ok:
        latencies = [r["elapsed"] for r in ok]
        latencies.sort()
        n = len(latencies)
        p = lambda pct: latencies[min(int(pct / 100 * n), n - 1)]
        print(f"\n  Latency (successful requests):")
        print(f"    min   = {min(latencies):.1f}s")
        print(f"    p50   = {p(50):.1f}s")
        print(f"    p75   = {p(75):.1f}s")
        print(f"    p90   = {p(90):.1f}s")
        print(f"    p95   = {p(95):.1f}s")
        print(f"    p99   = {p(99):.1f}s")
        print(f"    max   = {max(latencies):.1f}s")
        print(f"    mean  = {statistics.mean(latencies):.1f}s")

        throughput = len(ok) * pages_per_req / wall_time
        print(f"\n  Throughput      : {throughput:.2f} pages/s  "
              f"({len(ok)/wall_time:.2f} req/s)")

    if timeout:
        offsets = [r["offset"] for r in timeout]
        print(f"\n  Timed-out requests started at offsets: "
              f"{min(offsets):.1f}s – {max(offsets):.1f}s after t=0")
        print(f"  (These were queued too long — increase HPS_INFERENCE_TIMEOUT "
              f"or reduce concurrent users)")

    if errors:
        print(f"\n  Error breakdown:")
        from collections import Counter
        for msg, count in Counter(r["error"] for r in errors).most_common():
            print(f"    {count}× {msg}")

    # ASCII latency histogram (successful only)
    if ok and len(ok) > 1:
        print(f"\n  Latency histogram (successful):")
        min_l, max_l = min(latencies), max(latencies)
        n_bins = 10
        bin_w = max(1, (max_l - min_l) / n_bins)
        bins = [0] * n_bins
        for l in latencies:
            b = min(int((l - min_l) / bin_w), n_bins - 1)
            bins[b] += 1
        bar_max = max(bins)
        for b, count in enumerate(bins):
            lo = min_l + b * bin_w
            hi = lo + bin_w
            bar = "█" * int(30 * count / bar_max) if bar_max else ""
            print(f"    {lo:5.1f}s–{hi:5.1f}s  {bar:<30}  {count}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pod_base    = args.url.rstrip("/").removesuffix("/layout-parsing")
    gateway_url = f"{pod_base}/layout-parsing"

    # Health check
    try:
        r = requests.get(f"{pod_base}/health", timeout=10)
        print(f"Health: {r.status_code} — {r.text[:80]}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Prepare payload: extract first N pages as the reusable chunk
    content   = download_pdf(args.pdf)
    reader    = PdfReader(io.BytesIO(content))
    total_pg  = len(reader.pages)
    n_pages   = min(args.pages, total_pg)
    chunk     = extract_pages(content, 0, n_pages)
    chunk_b64 = base64.b64encode(chunk).decode()

    print(f"\nGateway  : {gateway_url}")
    print(f"Users    : {args.users}")
    print(f"Ramp     : {args.ramp}s" if args.ramp else "Ramp     : none (all at once)")
    print(f"Payload  : {n_pages} pages, {len(chunk):,} bytes → "
          f"{len(chunk_b64):,} bytes base64")
    print(f"Timeout  : {args.timeout}s per request")

    results, wall_time = asyncio.run(run_load_test(
        gateway_url, chunk_b64, args.users, args.ramp, args.timeout
    ))

    print_stats(results, wall_time, args.users, n_pages)

    # Save raw results
    out = OUT_DIR / "load_test_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaw results saved to {out}")


if __name__ == "__main__":
    main()
