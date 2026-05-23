"""Test the glmocr-vllmworker-serverless RunPod endpoint.

Usage:
    python test_endpoint.py document.pdf --endpoint-id <id> --api-key <key>
    python test_endpoint.py document.pdf --endpoint-id <id> --api-key <key> --concurrent 4
    python test_endpoint.py document.pdf --endpoint-id <id> --api-key <key> --repeat 3 --concurrent 2
    python test_endpoint.py document.pdf --endpoint-id <id> --api-key <key> --timeline
    python test_endpoint.py document.pdf --endpoint-id <id> --api-key <key> --save

Reads RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID from env if not passed as flags:
    RUNPOD_API_KEY=xxx RUNPOD_ENDPOINT_ID=yyy python test_endpoint.py document.pdf
"""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# Defaults read from environment so CI / shell aliases don't need explicit flags
_DEFAULT_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
_DEFAULT_API_KEY      = os.environ.get("RUNPOD_API_KEY", "")

# Cost estimate: RTX A4000 on RunPod (~$0.00013/s at spot pricing)
GPU_RATE = 0.00013   # $/s
IDR_RATE = 17_500    # IDR per USD


def _fmt(v, w=8) -> str:
    return f"{v:>{w}.2f}s" if isinstance(v, (int, float)) else f"{'?':>{w}}"


def _price(seconds: float) -> str:
    c = seconds * GPU_RATE
    if c >= 1:    return f"${c:.4f}"
    if c >= 0.01: return f"${c:.5f}"
    return f"${c:.6f}"


def _idr(seconds: float) -> str:
    rp = seconds * GPU_RATE * IDR_RATE
    if rp >= 1000: return f"Rp{rp:,.0f}"
    if rp >= 1:    return f"Rp{rp:.2f}"
    return f"Rp{rp:.4f}"


def _pc(seconds: float) -> str:
    return f"{_price(seconds)}  ({_idr(seconds)})"


# ── network call ───────────────────────────────────────────────────────────────

def send(endpoint_url: str, api_key: str, pdf_bytes: bytes,
         label: str, round_t0: float | None = None) -> tuple[str, float, dict]:
    t0 = time.time()
    t_sent_offset = round(t0 - round_t0, 3) if round_t0 is not None else 0.0
    b64 = base64.b64encode(pdf_bytes).decode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": {"images": [f"data:application/pdf;base64,{b64}"]}}

    try:
        with requests.post(
            endpoint_url, headers=headers, json=payload, timeout=600, stream=True
        ) as resp:
            status       = resp.status_code
            t_first_byte = time.time()
            raw          = resp.content
            t_last_byte  = time.time()

        wall = round(time.time() - t0, 3)
        client_meta = {
            "status":        status,
            "resp_bytes":    len(raw),
            "wall_s":        wall,
            "ttfb_s":        round(t_first_byte - t0, 3),
            "download_s":    round(t_last_byte - t_first_byte, 3),
            "t_sent_offset": t_sent_offset,
            "t_done_offset": round(time.time() - round_t0, 3) if round_t0 else wall,
        }

        try:
            envelope = json.loads(raw)
        except Exception:
            data = {"error": f"HTTP {status}: non-JSON ({len(raw)} bytes)"}
            envelope = {}
        else:
            # RunPod runsync wraps handler output in {"output": {...}}
            rp_status = envelope.get("status", "")
            if rp_status == "FAILED":
                data = {"error": envelope.get("error", "RunPod job FAILED")}
            elif "output" in envelope:
                data = envelope["output"]
                # Propagate any RunPod-level error fields
                if "error" in envelope and "error" not in data:
                    data["error"] = envelope["error"]
            else:
                # Unexpected shape — surface the raw envelope
                data = envelope

        if status >= 400 and "error" not in data:
            data["error"] = f"HTTP {status}"
        data["_client"] = client_meta
        return label, wall, data

    except requests.Timeout:
        wall = round(time.time() - t0, 3)
        return label, wall, {
            "error": f"CLIENT TIMEOUT after {wall:.1f}s",
            "_client": {"status": None, "resp_bytes": 0, "wall_s": wall,
                        "ttfb_s": None, "t_sent_offset": t_sent_offset,
                        "t_done_offset": round(time.time() - round_t0, 3) if round_t0 else wall},
        }
    except Exception as e:
        wall = round(time.time() - t0, 3)
        return label, wall, {
            "error": str(e),
            "_client": {"status": None, "resp_bytes": 0, "wall_s": wall,
                        "ttfb_s": None, "t_sent_offset": t_sent_offset,
                        "t_done_offset": round(time.time() - round_t0, 3) if round_t0 else wall},
        }


# ── helpers ────────────────────────────────────────────────────────────────────

def _count_pages(data: dict) -> int:
    jr = data.get("json_result", [])
    return len(jr) if isinstance(jr, list) else 0


def _count_regions(data: dict) -> int:
    jr = data.get("json_result", [])
    if isinstance(jr, list):
        return sum(len(p) for p in jr if isinstance(p, list))
    return 0


# ── single-request detail ──────────────────────────────────────────────────────

def print_result(label: str, wall: float, data: dict):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    cl      = data.get("_client", {})
    status  = cl.get("status")
    resp_kb = round(cl.get("resp_bytes", 0) / 1024, 1)

    if "error" in data:
        print(f"  HTTP {status or 'T/O'}  {resp_kb} KB")
        print(f"  ERROR: {data['error']}")
        return

    pages   = _count_pages(data)
    regions = _count_regions(data)
    markdown = data.get("markdown_result") or data.get("md_results") or ""

    print(f"  HTTP {status}  {resp_kb} KB")
    print(f"  Pages    : {pages}")
    if pages:
        print(f"  Regions  : {regions}  ({regions/pages:.1f} per page)")
    else:
        print(f"  Regions  : {regions}")

    print(f"\n  ── timing ──────────────────────────────────────────────")
    print(f"  ttfb_s   : {_fmt(cl.get('ttfb_s'))}")
    print(f"  wall_s   : {_fmt(wall)}   (client round-trip)")

    n_pages = pages or 1
    print(f"\n  ── pricing  (A4000 ${GPU_RATE}/s · Rp{IDR_RATE:,}/USD) ───────")
    print(f"  this request   : {_pc(wall)}")
    print(f"  per 1 000 req  : {_pc(wall * 1000)}")
    print(f"  per page       : {_pc(wall / n_pages)}")

    jr = data.get("json_result", [])
    if jr and isinstance(jr[0], list):
        print(f"\n  ── page 1 regions (first 5) ─────────────────────────")
        for region in jr[0][:5]:
            preview = (region.get("content") or "")[:60].replace("\n", " ")
            print(f"    [{region.get('label', '?'):<12}] {preview}")
        if len(jr[0]) > 5:
            print(f"    … and {len(jr[0]) - 5} more")

    print(f"\n  ── markdown (first 600 chars) ───────────────────────")
    print("  " + markdown[:600].replace("\n", "\n  "))
    if len(markdown) > 600:
        print(f"  … ({len(markdown) - 600} more chars)")


# ── concurrent timing table ────────────────────────────────────────────────────

def _print_immediate(label: str, wall: float, data: dict, round_t0: float) -> None:
    elapsed = round(time.time() - round_t0, 2)
    short   = label.split("#")[-1].strip() if "#" in label else label
    cl      = data.get("_client", {})
    status  = cl.get("status", "T/O")
    if "error" in data:
        print(f"  [+{elapsed:>6.2f}s]  req {short:<4}  {status}  ERROR: {data['error'][:50]}")
    else:
        pages   = _count_pages(data)
        regions = _count_regions(data)
        kb      = cl.get("resp_bytes", 0) / 1024
        print(f"  [+{elapsed:>6.2f}s]  req {short:<4}  {status}  {pages}p  {regions}r  {kb:.1f}KB  wall={wall:.2f}s")


def print_timing_table(results: list):
    print(f"\n  ── per-request timing ──────────────────────────────────────────────────")
    print(f"  {'req':<14}  {'st':>5}  {'size':>8}  {'pages':>5}  {'regions':>7}  {'ttfb':>8}  {'wall':>8}")
    print(f"  {'─'*14}  {'─'*5}  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}")

    walls = []
    ttfbs = []

    for label, wall, data in results:
        short    = label.split("#")[-1].strip() if "#" in label else label
        cl       = data.get("_client", {})
        status   = cl.get("status")
        kb       = cl.get("resp_bytes", 0) / 1024
        status_s = f"{status}" if status else "T/O"

        if "error" in data:
            print(f"  {short:<14}  {status_s:>5}  {kb:>7.1f}K  {data['error'][:35]}")
            continue

        pages   = _count_pages(data)
        regions = _count_regions(data)
        ttfb    = cl.get("ttfb_s")
        print(f"  {short:<14}  {status_s:>5}  {kb:>7.1f}K  {pages:>5}  {regions:>7}  {_fmt(ttfb)}  {_fmt(wall)}")
        walls.append(wall)
        if ttfb is not None:
            ttfbs.append(ttfb)

    print(f"  {'─'*14}  {'─'*5}  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}")
    if walls:
        def avg(lst): return sum(lst) / len(lst)
        for stat, fn in [("avg", avg), ("max", max)]:
            ttfb_s = _fmt(fn(ttfbs)) if ttfbs else f"{'?':>8}"
            print(f"  {stat:<14}  {'':>5}  {'':>8}  {'':>5}  {'':>7}  {ttfb_s}  {_fmt(fn(walls))}")


# ── ASCII timeline ─────────────────────────────────────────────────────────────

def print_ascii_timeline(results: list, round_wall: float):
    WIDTH = 50
    scale = WIDTH / max(round_wall, 1.0)
    unit  = round_wall / WIDTH
    print(f"\n  ── timeline  (1 char ≈ {unit:.1f}s) ─────────────────────────────────")
    for label, wall, data in results:
        cl     = data.get("_client", {})
        t_sent = cl.get("t_sent_offset", 0.0)
        t_done = cl.get("t_done_offset", wall)
        short  = label.split("#")[-1].strip() if "#" in label else label
        gap = int(t_sent * scale)
        bar = max(int((t_done - t_sent) * scale), 1)
        ch  = "█" if "error" not in data else "░"
        print(f"  #{short:<4}  {' '*gap}{ch*bar}  {t_done:.1f}s")
    step = max(1, int(round_wall / 8))
    axis = ""
    for tv in range(0, int(round_wall) + step, step):
        axis = axis.ljust(int(tv * scale) + 8) + f"{tv}s"
    print(f"        {axis}")


# ── distribution stats ─────────────────────────────────────────────────────────

def _dist(lst):
    if not lst:
        return None
    s = sorted(lst)
    n = len(s)
    def _p(pct): return s[min(int(n * pct / 100), n - 1)]
    return s[0], _p(50), sum(s) / n, _p(95), s[-1]


def print_summary(results: list, round_elapsed: float | None = None):
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    n_ok = sum(1 for _, _, d in results if "error" not in d)
    if round_elapsed:
        print(f"  {n_ok}/{len(results)} ok  |  {round_elapsed:.2f}s  |  {n_ok/round_elapsed:.2f} req/s")
    print(f"{'═' * 60}")

    walls = [w for _, w, d in results if "error" not in d]
    ttfbs = [d["_client"]["ttfb_s"] for _, _, d in results
             if "error" not in d and d.get("_client", {}).get("ttfb_s") is not None]

    W = 8
    def _f(v): return f"{v:>{W}.2f}s" if v is not None else f"{'─'*W}"
    def _row(name, lst):
        d = _dist(lst)
        if d is None: return
        mn, p50, avg, p95, mx = d
        print(f"  {name:<16}  {_f(mn)}  {_f(p50)}  {_f(avg)}  {_f(p95)}  {_f(mx)}")

    print(f"  {'metric':<16}  {'min':>{W}}  {'p50':>{W}}  {'avg':>{W}}  {'p95':>{W}}  {'max':>{W}}")
    print(f"  {'─'*16}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}")
    _row("ttfb_s",      ttfbs)
    _row("wall (e2e)",  walls)

    if walls:
        avg_wall = sum(walls) / len(walls)
        print(f"\n  ── pricing  (A4000 ${GPU_RATE}/s · Rp{IDR_RATE:,}/USD, avg wall) ──")
        print(f"  avg per request  : {_pc(avg_wall)}")
        print(f"  per 1 000 req    : {_pc(avg_wall * 1000)}")


# ── run logic ──────────────────────────────────────────────────────────────────

def run_round(label_prefix: str, n: int, endpoint_url: str, api_key: str,
              pdf_bytes: bytes, timeline: bool = False) -> tuple[list, float | None]:
    if n == 1:
        label = label_prefix
        print(f"\nSending {label} …")
        return [send(endpoint_url, api_key, pdf_bytes, label)], None

    print(f"\nSending {n} requests concurrently …")
    round_t0    = time.time()
    futures_map = {}
    results     = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            lbl = f"{label_prefix} #{i+1}"
            futures_map[pool.submit(send, endpoint_url, api_key, pdf_bytes, lbl, round_t0)] = lbl

        for fut in as_completed(futures_map):
            label, wall, data = fut.result()
            results.append((label, wall, data))
            _print_immediate(label, wall, data, round_t0)

    results.sort(key=lambda r: r[0])
    round_elapsed = time.time() - round_t0
    print(f"  Round finished in {round_elapsed:.2f}s")
    print_timing_table(results)
    if timeline:
        print_ascii_timeline(results, round_elapsed)
    return results, round_elapsed


def main():
    ap = argparse.ArgumentParser(description="Test glmocr-vllmworker-serverless RunPod endpoint")
    ap.add_argument("file",                                        help="PDF file to send")
    ap.add_argument("--endpoint-id",  default=_DEFAULT_ENDPOINT_ID, help="RunPod endpoint ID")
    ap.add_argument("--api-key",      default=_DEFAULT_API_KEY,      help="RunPod API key")
    ap.add_argument("--concurrent",   type=int,   default=1,         help="parallel requests per round")
    ap.add_argument("--repeat",       type=int,   default=1,         help="number of rounds")
    ap.add_argument("--delay",        type=float, default=0,         help="seconds between rounds")
    ap.add_argument("--timeline",     action="store_true",           help="ASCII timeline for concurrent rounds")
    ap.add_argument("--save",         action="store_true",           help="save JSON + markdown to disk")
    args = ap.parse_args()

    if not args.endpoint_id:
        ap.error("--endpoint-id required (or set RUNPOD_ENDPOINT_ID)")
    if not args.api_key:
        ap.error("--api-key required (or set RUNPOD_API_KEY)")

    endpoint_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/runsync"
    pdf_bytes    = Path(args.file).read_bytes()
    stem         = Path(args.file).stem

    print(f"Endpoint   : {endpoint_url}")
    print(f"File       : {args.file}  ({len(pdf_bytes):,} bytes)")
    print(f"Rounds     : {args.repeat}  ×  {args.concurrent} concurrent")

    all_results  = []
    last_elapsed = None

    for r in range(args.repeat):
        if r > 0 and args.delay > 0:
            print(f"\n  [waiting {args.delay}s …]")
            time.sleep(args.delay)

        prefix  = f"round {r+1}" if args.repeat > 1 else "request"
        results, elapsed = run_round(prefix, args.concurrent, endpoint_url,
                                     args.api_key, pdf_bytes, args.timeline)
        last_elapsed = elapsed
        single = args.repeat == 1 and args.concurrent == 1

        for label, wall, data in results:
            if args.concurrent == 1:
                print_result(label, wall, data)
            if (args.save or single) and "error" not in data:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"{stem}_{label.replace(' ', '_')}_{ts}"
                path = Path(name)
                path.mkdir(exist_ok=True)
                clean = {k: v for k, v in data.items() if k != "_client"}
                (path / "result.json").write_text(
                    json.dumps(clean, indent=2, ensure_ascii=False)
                )
                md = data.get("markdown_result") or data.get("md_results") or ""
                (path / "result.md").write_text(md)
                print(f"  Saved: {path}/")

        all_results.extend(results)

    if len(all_results) > 1:
        print_summary(all_results, last_elapsed if args.repeat == 1 else None)


if __name__ == "__main__":
    main()
