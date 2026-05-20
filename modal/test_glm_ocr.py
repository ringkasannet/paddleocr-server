"""Test script for the deployed GLM-OCR whole-page endpoint.

Usage:
    python modal/test_glm_ocr.py document.pdf
    python modal/test_glm_ocr.py document.pdf --page 2
    python modal/test_glm_ocr.py document.pdf --repeat 3
    python modal/test_glm_ocr.py document.pdf --concurrent 4
    python modal/test_glm_ocr.py document.pdf --repeat 2 --concurrent 4
    python modal/test_glm_ocr.py document.pdf --timeline
    python modal/test_glm_ocr.py document.pdf --save
    python modal/test_glm_ocr.py document.pdf --endpoint https://your-url.modal.run
"""

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

ENDPOINT         = "https://ringkasan-net--glm-ocr-ocrfrontend-process.modal.run"
COLD_THRESHOLD   = 3.0   # queued_s above this → likely cold start
GPU_SCALEDOWN_S  = 5     # scaledown_window on GLMOCRWorker
GPU_RATE         = 0.000222  # L4 $/s


def _fmt(v, w=7, unit="s") -> str:
    return f"{v:>{w}.2f}{unit}" if isinstance(v, (int, float)) else f"{'?':>{w}}"


def _container_label(queued_s) -> str:
    if queued_s is None:
        return "?"
    if queued_s > COLD_THRESHOLD:
        return f"COLD  {queued_s:.1f}s restore"
    return f"warm  {queued_s:.3f}s"


# ── network call ───────────────────────────────────────────────────────────────

def send(endpoint: str, pdf_bytes: bytes, page: int, dpi: int,
         label: str, round_t0: float | None = None) -> tuple[str, float, dict]:
    t0 = time.time()
    t_sent_offset = round(t0 - round_t0, 3) if round_t0 is not None else 0.0
    b64 = base64.b64encode(pdf_bytes).decode()
    try:
        with requests.post(
            endpoint,
            json={"file": b64, "page": page, "dpi": dpi},
            timeout=600,
            stream=True,
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
            data = json.loads(raw)
        except Exception:
            data = {"error": f"HTTP {status}: non-JSON response ({len(raw)} bytes)"}
        if status >= 400 and "error" not in data:
            data["error"] = f"HTTP {status}"
        data["_client"] = client_meta
        return label, wall, data

    except requests.Timeout:
        wall = round(time.time() - t0, 3)
        return label, wall, {
            "error": f"CLIENT TIMEOUT after {wall:.1f}s (server still running)",
            "_client": {"status": None, "resp_bytes": 0, "wall_s": wall,
                        "t_sent_offset": t_sent_offset,
                        "t_done_offset": round(time.time() - round_t0, 3) if round_t0 else wall},
        }
    except Exception as e:
        wall = round(time.time() - t0, 3)
        return label, wall, {
            "error": str(e),
            "_client": {"status": None, "resp_bytes": 0, "wall_s": wall,
                        "t_sent_offset": t_sent_offset,
                        "t_done_offset": round(time.time() - round_t0, 3) if round_t0 else wall},
        }


# ── per-request detail ─────────────────────────────────────────────────────────

def print_result(label: str, wall: float, data: dict):
    print(f"\n{'─' * 56}")
    print(f"  {label}")
    print(f"{'─' * 56}")

    client   = data.get("_client", {})
    status   = client.get("status")
    resp_kb  = round(client.get("resp_bytes", 0) / 1024, 1)
    status_s = f"HTTP {status}" if status else "TIMEOUT"

    if "error" in data:
        print(f"  {status_s}  {resp_kb} KB")
        print(f"  ERROR: {data['error']}")
        return

    meta   = data.get("meta", {})
    timing = meta.get("timing", {})
    text   = data.get("text", "")
    queued_s = timing.get("ocr_queued_s")

    print(f"  {status_s}  {resp_kb} KB")
    print(f"  Page     : {meta.get('page')}  ({meta.get('width_px')}×{meta.get('height_px')}px @ {meta.get('dpi')} dpi)")
    print(f"  Chars    : {len(text)}")
    print(f"\n  ── timing ───────────────────────────────────────────")
    print(f"  render_s      : {_fmt(timing.get('render_s'))}")
    print(f"  ocr_queued_s  : {_fmt(queued_s)}   {_container_label(queued_s)}")
    print(f"  ocr_exec_s    : {_fmt(timing.get('ocr_exec_s'))}   ← vLLM inference")
    print(f"  ocr_wall_s    : {_fmt(timing.get('ocr_wall_s'))}")
    print(f"  total_s       : {_fmt(timing.get('total_s'))}   (server)")
    print(f"  ttfb_s        : {_fmt(client.get('ttfb_s'))}   (client)")
    print(f"  wall_s        : {_fmt(wall)}   (client round-trip)")
    print(f"\n  ── text (first 400 chars) ───────────────────────────")
    print(f"  {text[:400].replace(chr(10), chr(10) + '  ')}")
    if len(text) > 400:
        print(f"  … ({len(text) - 400} more chars)")


# ── concurrent timing table ────────────────────────────────────────────────────

def print_timing_table(results: list):
    print(f"\n  ── per-request timing ─────────────────────────────────────────────────────────────")
    print(f"  {'req':<12}  {'st':>5}  {'size':>7}  {'render':>7}  {'queued':>7}  {'exec':>7}  {'total':>7}  {'ttfb':>7}  {'wall':>7}")
    print(f"  {'─'*12}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

    cols = {k: [] for k in ["render", "queued", "exec", "total", "ttfb", "wall"]}

    for label, wall, data in results:
        short  = label.split("#")[-1].strip() if "#" in label else label
        cl     = data.get("_client", {})
        status = cl.get("status")
        kb     = round(cl.get("resp_bytes", 0) / 1024, 1)
        status_s = f"{status}" if status else "T/O"
        if "error" in data:
            err_short = data["error"][:30]
            print(f"  {short:<12}  {status_s:>5}  {kb:>6.1f}KB  {err_short}")
            continue
        t  = data.get("meta", {}).get("timing", {})
        vs = {
            "render": t.get("render_s"),
            "queued": t.get("ocr_queued_s"),
            "exec":   t.get("ocr_exec_s"),
            "total":  t.get("total_s"),
            "ttfb":   cl.get("ttfb_s"),
            "wall":   wall,
        }
        print(f"  {short:<12}  {status_s:>5}  {kb:>6.1f}KB  " + "  ".join(_fmt(vs[k]) for k in ["render", "queued", "exec", "total", "ttfb", "wall"]))
        for k, v in vs.items():
            if isinstance(v, (int, float)):
                cols[k].append(v)

    print(f"  {'─'*12}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for stat, fn in [("avg", lambda lst: sum(lst)/len(lst)), ("max", max)]:
        row = "  ".join(_fmt(fn(cols[k])) if cols[k] else f"{'?':>7}" for k in ["render", "queued", "exec", "total", "ttfb", "wall"])
        print(f"  {stat:<12}  {'':>5}  {'':>7}  {row}")


# ── ASCII timeline ─────────────────────────────────────────────────────────────

def print_ascii_timeline(results: list, round_wall: float):
    WIDTH = 50
    scale = WIDTH / max(round_wall, 1.0)
    unit  = round_wall / WIDTH
    print(f"\n  ── timeline  (1 char ≈ {unit:.1f}s) ─────────────────────────────────────")
    for label, wall, data in results:
        cl       = data.get("_client", {})
        t_sent   = cl.get("t_sent_offset", 0.0)
        t_done   = cl.get("t_done_offset", wall)
        queued_s = data.get("meta", {}).get("timing", {}).get("ocr_queued_s") if "error" not in data else None
        cold     = queued_s is not None and queued_s > COLD_THRESHOLD
        gap = int(t_sent * scale)
        bar = max(int((t_done - t_sent) * scale), 1)
        ch  = "░" if cold else "█"
        short = label.split("#")[-1].strip() if "#" in label else label
        print(f"  #{short:<4}  {' '*gap}{ch*bar}  {t_done:.1f}s{' ← COLD' if cold else ''}")
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
    return s[0], _p(50), sum(s)/n, _p(95), s[-1]


def print_summary(results: list, round_elapsed: float | None = None):
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    n_ok = sum(1 for _, _, d in results if "error" not in d)
    if round_elapsed:
        print(f"  {n_ok} requests  |  {round_elapsed:.2f}s  |  {n_ok/round_elapsed:.2f} req/s")
    print(f"{'═' * 60}")

    cols = {k: [] for k in ["queued", "exec", "total", "wall"]}
    for _, wall, data in results:
        if "error" in data:
            continue
        t = data.get("meta", {}).get("timing", {})
        for k, src in [("queued", t.get("ocr_queued_s")), ("exec", t.get("ocr_exec_s")),
                       ("total", t.get("total_s")), ("wall", wall)]:
            if isinstance(src, (int, float)):
                cols[k].append(src)

    W = 7
    def _f(v): return f"{v:>{W}.2f}s" if v is not None else f"{'─'*W}"
    def _row(name, lst):
        d = _dist(lst)
        if d is None: return
        mn, p50, avg, p95, mx = d
        print(f"  {name:<16}  {_f(mn)}  {_f(p50)}  {_f(avg)}  {_f(p95)}  {_f(mx)}")

    print(f"  {'metric':<16}  {'min':>{W}}  {'p50':>{W}}  {'avg':>{W}}  {'p95':>{W}}  {'max':>{W}}")
    print(f"  {'─'*16}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}")
    _row("ocr_queued_s",  cols["queued"])
    _row("ocr_exec_s",    cols["exec"])
    _row("total_s",       cols["total"])
    _row("wall (e2e)",    cols["wall"])


# ── run logic ──────────────────────────────────────────────────────────────────

def run_round(label_prefix: str, n: int, endpoint: str, pdf_bytes: bytes,
              page: int, dpi: int, timeline: bool = False) -> tuple[list, float | None]:
    if n == 1:
        label = label_prefix
        print(f"\nSending {label} …")
        return [send(endpoint, pdf_bytes, page, dpi, label)], None

    print(f"\nSending {n} requests concurrently …")
    round_t0     = time.time()
    futures_map  = {}
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            lbl = f"{label_prefix} #{i+1}"
            futures_map[pool.submit(send, endpoint, pdf_bytes, page, dpi, lbl, round_t0)] = lbl

    results = [fut.result() for fut in as_completed(futures_map)]
    results.sort(key=lambda r: r[0])

    round_elapsed = time.time() - round_t0
    print(f"  Round finished in {round_elapsed:.2f}s")
    print_timing_table(results)
    if timeline:
        print_ascii_timeline(results, round_elapsed)
    return results, round_elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--page",       type=int,   default=0)
    ap.add_argument("--dpi",        type=int,   default=200)
    ap.add_argument("--repeat",     type=int,   default=1)
    ap.add_argument("--concurrent", type=int,   default=1)
    ap.add_argument("--delay",      type=float, default=0,  help="seconds between serial rounds")
    ap.add_argument("--timeline",   action="store_true")
    ap.add_argument("--endpoint",   default=ENDPOINT)
    ap.add_argument("--save",       action="store_true")
    args = ap.parse_args()

    pdf_bytes = Path(args.file).read_bytes()
    stem      = Path(args.file).stem

    print(f"Endpoint   : {args.endpoint}")
    print(f"File       : {args.file}  ({len(pdf_bytes):,} bytes)")
    print(f"Page       : {args.page}  DPI={args.dpi}")
    print(f"Rounds     : {args.repeat}  ×  {args.concurrent} concurrent")

    all_results = []
    last_elapsed = None

    for r in range(args.repeat):
        if r > 0 and args.delay > 0:
            print(f"\n  [waiting {args.delay}s …]")
            time.sleep(args.delay)

        prefix  = f"round {r+1}" if args.repeat > 1 else "request"
        results, elapsed = run_round(prefix, args.concurrent, args.endpoint,
                                     pdf_bytes, args.page, args.dpi, args.timeline)
        last_elapsed = elapsed

        single = args.repeat == 1 and args.concurrent == 1

        for label, wall, data in results:
            if args.concurrent == 1:
                print_result(label, wall, data)
            if (args.save or single) and "error" not in data:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"{stem}_p{args.page}_{label.replace(' ','_')}_{ts}.json"
                Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
                print(f"  Saved: {path}")

        all_results.extend(results)

    if len(all_results) > 1:
        print_summary(all_results, last_elapsed if args.repeat == 1 else None)


if __name__ == "__main__":
    main()
