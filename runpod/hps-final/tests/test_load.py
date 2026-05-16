#!/usr/bin/env python3
"""
Load test: simulate N concurrent users hitting the HPS gateway.
Uses asyncio + aiohttp for true async concurrency (no thread overhead).

Install deps:
    pip install aiohttp pypdf

Usage examples:
    # 100 users all at once (run on the pod for accurate results)
    python test_load.py --users 100

    # 1000 users ramped over 60 seconds
    python test_load.py --users 1000 --ramp 60

    # From outside via TCP port (bypasses RunPod proxy timeout)
    python test_load.py --url http://<pod-ip>:<tcp-port> --users 100

    # Smaller payload for faster queue cycling
    python test_load.py --users 500 --pages 1
"""
import argparse
import asyncio
import base64
import io
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import aiohttp
import requests
from pypdf import PdfReader, PdfWriter

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URL     = "http://localhost:8080"
DEFAULT_PDF_URL = (
    "https://jdih.kemenkeu.go.id/api/download/"
    "637047be-3dba-4347-aba1-98fa7fd5ab3f/2024pmkeuangan081.pdf"
)
OUT_DIR = Path(__file__).parent / "results"
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


def download_pdf(url: str) -> bytes:
    from urllib.parse import urlparse
    slug = Path(urlparse(url).path).stem
    cache = OUT_DIR / f"source_{slug}.pdf"
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


def make_chunks(content: bytes, pages_per_chunk: int, n_unique: int) -> list[bytes]:
    """
    Produce exactly n_unique unique PDF chunks.
    Pages drawn round-robin; each chunk stamped with unique metadata
    so bytes differ even when page content repeats.
    """
    reader = PdfReader(io.BytesIO(content))
    total_pages = len(reader.pages)
    chunks = []
    for idx in range(n_unique):
        w = PdfWriter()
        for j in range(pages_per_chunk):
            page_no = (idx * pages_per_chunk + j) % total_pages
            w.add_page(reader.pages[page_no])
        w.add_metadata({"/Keywords": f"load-test-chunk-{idx:05d}"})
        buf = io.BytesIO()
        w.write(buf)
        chunks.append(buf.getvalue())
    return chunks


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
    http_status = 0
    error = None
    n_blocks = 0
    raw = b""
    data = None

    try:
        async with session.post(gateway_url, json=payload) as resp:
            http_status = resp.status
            raw = await resp.read()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            snippet = raw[:120].decode("utf-8", errors="replace").strip()
            error = f"non-JSON body: {snippet}"
            if http_status == 524:
                error = "proxy_timeout (RunPod/Cloudflare ~140s limit)"

        if http_status == 200 and data is not None:
            error_code = data.get("errorCode", -1)
            if error_code != 0:
                error = f"errorCode={error_code}: {data.get('errorMsg', '')}"
                http_status = -1
            else:
                pages = (data.get("result") or {}).get("layoutParsingResults") or []
                if not pages:
                    error = "empty layoutParsingResults"
                    http_status = -1
                else:
                    n_blocks = sum(
                        len(p.get("prunedResult", {}).get("parsing_res_list", []))
                        for p in pages
                    )

    except asyncio.TimeoutError:
        error = "client_timeout"
    except Exception as e:
        error = str(e)[:80]

    elapsed = time.time() - start
    results.append({
        "user_id":    user_id,
        "status":     http_status,
        "elapsed":    elapsed,
        "offset":     offset,
        "error":      error,
        "n_blocks":   n_blocks,
        "resp_bytes": len(raw),
        "resp_data":  data,
    })

    if http_status == 200:
        symbol, detail = "✓", f"{n_blocks} blocks"
    elif error == "client_timeout":
        symbol, detail = "T", "client_timeout"
    elif http_status == 524:
        symbol, detail = "P", "proxy_timeout (RunPod/Cloudflare)"
    else:
        symbol, detail = "✗", error or f"HTTP {http_status}"

    print(f"  [{symbol}] user_{user_id:04d}  {elapsed:6.1f}s  +{offset:6.1f}s  "
          f"HTTP {http_status if http_status > 0 else '???'}  {detail}")


async def run_load_test(gateway_url, chunks_b64, n_users, ramp_seconds, timeout_seconds):
    results: list = []
    connector = aiohttp.TCPConnector(limit=0)
    client_timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    t0 = time.time()
    print(f"\nFiring {n_users} requests"
          + (f" ramped over {ramp_seconds}s" if ramp_seconds else " all at once")
          + f"  ({len(chunks_b64)} unique PDF chunks)")
    print(f"{'─'*62}")

    def payload_for(i):
        return {"file": chunks_b64[i - 1], "fileType": 0, "visualize": False}

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        if ramp_seconds <= 0:
            await asyncio.gather(*[
                asyncio.create_task(send_request(session, gateway_url, payload_for(i), i, t0, results))
                for i in range(1, n_users + 1)
            ])
        else:
            interval = ramp_seconds / n_users
            tasks = []
            for i in range(1, n_users + 1):
                tasks.append(asyncio.create_task(
                    send_request(session, gateway_url, payload_for(i), i, t0, results)
                ))
                await asyncio.sleep(interval)
            await asyncio.gather(*tasks)

    return results, time.time() - t0


def print_stats(results, wall_time, n_users, pages_per_req, n_chunks):
    ok            = [r for r in results if r["status"] == 200 and r["n_blocks"] > 0]
    empty         = [r for r in results if r["status"] == 200 and r["n_blocks"] == 0]
    proxy_timeout = [r for r in results if r["status"] == 524]
    client_to     = [r for r in results if r["error"] == "client_timeout"]
    errors        = [r for r in results if r["status"] not in (200, 524, 0)
                     and r["error"] != "client_timeout"]

    print(f"\n{'='*62}")
    print(f" LOAD TEST SUMMARY  ({n_users} users, {pages_per_req} pages/req, {n_chunks} unique chunks)")
    print(f"{'='*62}")
    print(f"  Wall time       : {wall_time:.1f}s")
    print(f"  Successful      : {len(ok)} / {n_users}  ({100*len(ok)/n_users:.0f}%)")
    print(f"  Empty results   : {len(empty)}  (HTTP 200 but no layout blocks)")
    print(f"  Proxy timeout   : {len(proxy_timeout)}  (HTTP 524 — RunPod/Cloudflare ~140s limit)")
    print(f"  Client timeout  : {len(client_to)}  (exceeded --timeout)")
    print(f"  Errors          : {len(errors)}")

    if ok:
        latencies = sorted(r["elapsed"] for r in ok)
        n = len(latencies)
        p = lambda pct: latencies[min(int(pct / 100 * n), n - 1)]
        print(f"\n  Latency (successful requests):")
        print(f"    min={min(latencies):.1f}s  p50={p(50):.1f}s  p75={p(75):.1f}s  "
              f"p90={p(90):.1f}s  p95={p(95):.1f}s  p99={p(99):.1f}s  max={max(latencies):.1f}s")
        print(f"    mean={statistics.mean(latencies):.1f}s")
        print(f"\n  Throughput : {len(ok)*pages_per_req/wall_time:.2f} pages/s  "
              f"({len(ok)/wall_time:.2f} req/s)")

    if client_to:
        offsets = [r["offset"] for r in client_to]
        print(f"\n  Client timeouts started at +{min(offsets):.1f}s – +{max(offsets):.1f}s")

    if errors:
        print(f"\n  Error breakdown:")
        for msg, count in Counter(r["error"] for r in errors).most_common():
            print(f"    {count}× {msg}")

    print(f"\n  {'User':<10} {'Status':>7} {'Time (s)':>10} {'Resp (KB)':>10} {'Blocks':>7}  Note")
    print(f"  {'─'*10} {'─'*7} {'─'*10} {'─'*10} {'─'*7}  {'─'*20}")
    for r in sorted(results, key=lambda x: x["user_id"]):
        kb = r["resp_bytes"] / 1024
        status_str = str(r["status"]) if r["status"] > 0 else "???"
        print(f"  user_{r['user_id']:04d}  {status_str:>7}  {r['elapsed']:>9.1f}s"
              f"  {kb:>9.1f}  {r['n_blocks']:>7}  {r['error'] or ''}")

    if ok and len(ok) > 1:
        latencies = sorted(r["elapsed"] for r in ok)
        min_l, max_l = latencies[0], latencies[-1]
        n_bins = 10
        bin_w = max(1, (max_l - min_l) / n_bins)
        bins = [0] * n_bins
        for l in latencies:
            bins[min(int((l - min_l) / bin_w), n_bins - 1)] += 1
        bar_max = max(bins)
        print(f"\n  Latency histogram (successful):")
        for b, count in enumerate(bins):
            lo, hi = min_l + b * bin_w, min_l + (b + 1) * bin_w
            bar = "█" * int(30 * count / bar_max) if bar_max else ""
            print(f"    {lo:5.1f}s–{hi:5.1f}s  {bar:<30}  {count}")


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pod_base    = args.url.rstrip("/").removesuffix("/layout-parsing")
    gateway_url = f"{pod_base}/layout-parsing"

    try:
        r = requests.get(f"{pod_base}/health", timeout=10)
        print(f"Health: {r.status_code} — {r.text[:80]}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    content  = download_pdf(args.pdf)
    reader   = PdfReader(io.BytesIO(content))
    n_pages  = min(args.pages, len(reader.pages))
    print(f"Building {args.users} unique PDF chunks ({n_pages} pages each)...")
    chunks     = make_chunks(content, n_pages, args.users)
    chunks_b64 = [base64.b64encode(c).decode() for c in chunks]

    print(f"\nGateway : {gateway_url}")
    print(f"Users   : {args.users}")
    print(f"Ramp    : {args.ramp}s" if args.ramp else "Ramp    : none (all at once)")
    print(f"Chunks  : {len(chunks)} unique ({n_pages} pages each, "
          f"~{sum(len(c) for c in chunks)//len(chunks):,} bytes avg)")
    print(f"Timeout : {args.timeout}s per request")

    results, wall_time = asyncio.run(
        run_load_test(gateway_url, chunks_b64, args.users, args.ramp, args.timeout)
    )

    print_stats(results, wall_time, args.users, n_pages, len(chunks))

    summary = [{k: v for k, v in r.items() if k != "resp_data"} for r in results]
    (OUT_DIR / "load_test_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    resp_dir = OUT_DIR / "load_test_responses"
    resp_dir.mkdir(exist_ok=True)
    saved = sum(
        1 for r in results
        if r["resp_data"] is not None
        and (resp_dir / f"user_{r['user_id']:04d}_http{r['status']}.json").write_text(
            json.dumps(r["resp_data"], ensure_ascii=False, indent=2), encoding="utf-8"
        ) is not None
    )
    print(f"\nSummary  → {OUT_DIR}/load_test_results.json")
    print(f"Responses → {resp_dir.name}/  ({saved} files)")


if __name__ == "__main__":
    main()
