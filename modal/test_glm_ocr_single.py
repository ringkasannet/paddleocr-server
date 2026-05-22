"""Test script for the deployed GLM-OCR single-document endpoint.

Usage:
    python modal/test_glm_ocr_single.py document.pdf
    python modal/test_glm_ocr_single.py document.pdf --pages 0 1 2
    python modal/test_glm_ocr_single.py document.pdf --concurrent 3
    python modal/test_glm_ocr_single.py document.pdf --repeat 2 --concurrent 3
    python modal/test_glm_ocr_single.py document.pdf --timeline
    python modal/test_glm_ocr_single.py document.pdf --save
    python modal/test_glm_ocr_single.py document.pdf --endpoint https://your-url.modal.run
"""

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

ENDPOINT = "https://ringkasan-net--glm-ocr-single-documentocrworker-process.modal.run"
COLD_THRESHOLD = 3.0
GPU_RATE  = 0.000222   # L4 $/s (Modal on-demand)
IDR_RATE  = 17_500     # IDR per USD


def _fmt(v, w=8) -> str:
    return f"{v:>{w}.2f}s" if isinstance(v, (int, float)) else f"{'?':>{w}}"


def _price(seconds: float) -> str:
    c = seconds * GPU_RATE
    if c >= 1:
        return f"${c:.4f}"
    if c >= 0.01:
        return f"${c:.5f}"
    return f"${c:.6f}"


def _idr(seconds: float) -> str:
    rp = seconds * GPU_RATE * IDR_RATE
    if rp >= 1000:
        return f"Rp{rp:,.0f}"
    if rp >= 1:
        return f"Rp{rp:.2f}"
    return f"Rp{rp:.4f}"


def _pc(seconds: float) -> str:
    return f"{_price(seconds)}  ({_idr(seconds)})"


def _pricing_block(total_s: float | None, wall_s: float,
                   n_pages: int | None = None) -> None:
    print(f"\n  ── pricing  (L4 ${GPU_RATE}/s · Rp{IDR_RATE:,}/USD, 1 req, no concurrency) ──")
    if total_s is not None:
        print(f"  lower bound  (server total_s)  : {_pc(total_s)}")
        print(f"  upper bound  (client wall_s)   : {_pc(wall_s)}")
        print(f"  per 1 000 req (lower)          : {_pc(total_s * 1000)}")
        print(f"  per 1 000 req (upper)          : {_pc(wall_s  * 1000)}")
        if n_pages and n_pages > 0:
            print(f"  per page  ({n_pages:>3} pages) lower  : {_pc(total_s / n_pages)}")
            print(f"  per page  ({n_pages:>3} pages) upper  : {_pc(wall_s  / n_pages)}")
    else:
        print(f"  upper bound  (client wall_s)   : {_pc(wall_s)}")


# ── network call ───────────────────────────────────────────────────────────────

def send(endpoint: str, pdf_bytes: bytes, pages: list | None, dpi: int,
         label: str, round_t0: float | None = None) -> tuple[str, float, dict]:
    t0 = time.time()
    t_sent_offset = round(t0 - round_t0, 3) if round_t0 is not None else 0.0
    b64 = base64.b64encode(pdf_bytes).decode()
    payload = {"file": b64, "dpi": dpi}
    if pages is not None:
        payload["pages"] = pages
    try:
        with requests.post(endpoint, json=payload, timeout=600, stream=True) as resp:
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
            "error": f"CLIENT TIMEOUT after {wall:.1f}s",
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
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    client   = data.get("_client", {})
    status   = client.get("status")
    resp_kb  = round(client.get("resp_bytes", 0) / 1024, 1)

    if "error" in data:
        print(f"  HTTP {status or 'T/O'}  {resp_kb} KB")
        print(f"  ERROR: {data['error']}")
        return

    meta     = data.get("meta", {})
    timing   = meta.get("timing", {})
    markdown = data.get("markdown", "")
    blocks   = data.get("blocks", [])

    status  = client.get("status")
    resp_kb = round(client.get("resp_bytes", 0) / 1024, 1)
    print(f"  HTTP {status}  {resp_kb} KB")
    print(f"  Pages processed  : {meta.get('pages')}")
    print(f"  Total regions    : {meta.get('total_regions')}")
    print(f"  OCR'd regions    : {meta.get('ocr_regions')}")
    print(f"  Skipped regions  : {meta.get('skip_regions')}")

    print(f"\n  ── timing ─────────────────────────────────────────────")
    if timing.get("decode_s") is not None:
        print(f"  decode_s          : {_fmt(timing.get('decode_s'))}")
    print(f"  render_s          : {_fmt(timing.get('render_s'))}")
    print(f"  layout_s          : {_fmt(timing.get('layout_s'))}")
    print(f"  crop_s            : {_fmt(timing.get('crop_s'))}")
    print(f"  ocr_wall_s        : {_fmt(timing.get('ocr_wall_s'))}")
    if timing.get("ocr_avg_queued_s") is not None:
        print(f"  ocr_avg_queued_s  : {_fmt(timing.get('ocr_avg_queued_s'))}   ← avg GPU slot wait")
    if timing.get("ocr_first_result_s") is not None:
        print(f"  ocr_first_result_s: {_fmt(timing.get('ocr_first_result_s'))}   ← vLLM first batch latency")
    print(f"  ocr_avg_exec_s    : {_fmt(timing.get('ocr_avg_exec_s'))}   ← avg per-region")
    if timing.get("ocr_min_exec_s") is not None:
        print(f"  ocr_min_exec_s    : {_fmt(timing.get('ocr_min_exec_s'))}   ← fastest region")
    print(f"  ocr_max_exec_s    : {_fmt(timing.get('ocr_max_exec_s') or timing.get('ocr_max_wall_s'))}   ← slowest region")
    if timing.get("assemble_s") is not None:
        print(f"  assemble_s        : {_fmt(timing.get('assemble_s'))}")
    print(f"  total_s           : {_fmt(timing.get('total_s'))}   (server)")
    print(f"  ttfb_s            : {_fmt(client.get('ttfb_s'))}   (client)")
    print(f"  wall_s            : {_fmt(wall)}   (client round-trip)")
    cs = timing.get("cold_start")
    if cs:
        print(f"\n  ── cold start ─────────────────────────────────────────")
        print(f"  wakeup_s          : {_fmt(cs.get('wakeup_s'))}   ← vLLM weight restore")
        print(f"  health_s          : {_fmt(cs.get('health_s'))}   ← ready check")
        layout_key = "layout_gpu_s" if "layout_gpu_s" in cs else "layout_load_s"
        print(f"  {layout_key:<17} : {_fmt(cs.get(layout_key))}   ← layout .to(cuda) + warmup")
        if "batch_warmup_s" in cs:
            print(f"  batch_warmup_s    : {_fmt(cs.get('batch_warmup_s'))}   ← 16-concurrent Triton JIT pre-compile")
        print(f"  total_wake_s      : {_fmt(cs.get('total_s'))}   ← total wake() time")
        ld = cs.get("layout_detail")
        if ld:
            print(f"\n  ── layout activation detail ───────────────────────────")
            for k in ("torch_import_s", "transformers_import_s", "processor_s",
                      "model_load_s", "model_cuda_s", "warmup_s"):
                if k in ld:
                    print(f"  {k:<24}: {_fmt(ld.get(k))}")

    n_pages = len(meta.get("pages") or [])
    _pricing_block(timing.get("total_s"), wall, n_pages if n_pages > 0 else None)

    print(f"\n  ── blocks ({len(blocks)}) ──────────────────────────────────────")
    for b in blocks[:10]:
        text_preview = (b.get("text") or "")[:60].replace("\n", " ")
        print(f"    p{b['page']} [{b['order']:3d}] {b['label']:<22s}  {text_preview}")
    if len(blocks) > 10:
        print(f"    … and {len(blocks) - 10} more")

    print(f"\n  ── markdown (first 600 chars) ──────────────────────────")
    print("  " + markdown[:600].replace("\n", "\n  "))
    if len(markdown) > 600:
        print(f"  … ({len(markdown) - 600} more chars)")


# ── concurrent timing table ────────────────────────────────────────────────────

def print_timing_table(results: list):
    print(f"\n  ── per-request timing ─────────────────────────────────────────────────────────────────────────────────────────────")
    print(f"  {'req':<12}  {'st':>5}  {'size':>7}  {'render':>8}  {'layout':>8}  {'ocr_wall':>8}  {'avg_q':>7}  {'avg_exec':>8}  {'total':>8}  {'wall':>8}")
    print(f"  {'─'*12}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")

    cols = {k: [] for k in ["render", "layout", "ocr_wall", "avg_q", "avg_exec", "total", "wall"]}

    for label, wall, data in results:
        short  = label.split("#")[-1].strip() if "#" in label else label
        cl     = data.get("_client", {})
        status = cl.get("status")
        kb     = round(cl.get("resp_bytes", 0) / 1024, 1)
        status_s = f"{status}" if status else "T/O"
        if "error" in data:
            print(f"  {short:<12}  {status_s:>5}  {kb:>6.1f}KB  {data['error'][:40]}")
            continue
        t  = data.get("meta", {}).get("timing", {})
        vs = {
            "render":   t.get("render_s"),
            "layout":   t.get("layout_s"),
            "ocr_wall": t.get("ocr_wall_s"),
            "avg_q":    t.get("ocr_avg_queued_s"),
            "avg_exec": t.get("ocr_avg_exec_s"),
            "total":    t.get("total_s"),
            "wall":     wall,
        }
        row = "  ".join(_fmt(vs[k], 8 if k != "avg_q" else 7) for k in ["render", "layout", "ocr_wall", "avg_q", "avg_exec", "total", "wall"])
        print(f"  {short:<12}  {status_s:>5}  {kb:>6.1f}KB  {row}")
        for k, v in vs.items():
            if isinstance(v, (int, float)):
                cols[k].append(v)

    print(f"  {'─'*12}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
    for stat, fn in [("avg", lambda lst: sum(lst)/len(lst)), ("max", max)]:
        row = "  ".join(
            _fmt(fn(cols[k]), 8 if k != "avg_q" else 7) if cols[k] else f"{'?':>{8 if k != 'avg_q' else 7}}"
            for k in ["render", "layout", "ocr_wall", "avg_q", "avg_exec", "total", "wall"]
        )
        print(f"  {stat:<12}  {'':>5}  {'':>7}  {row}")


# ── ASCII timeline ─────────────────────────────────────────────────────────────

def print_ascii_timeline(results: list, round_wall: float):
    WIDTH = 50
    scale = WIDTH / max(round_wall, 1.0)
    unit  = round_wall / WIDTH
    print(f"\n  ── timeline  (1 char ≈ {unit:.1f}s) ─────────────────────────────────────")
    for label, wall, data in results:
        cl     = data.get("_client", {})
        t_sent = cl.get("t_sent_offset", 0.0)
        t_done = cl.get("t_done_offset", wall)
        short  = label.split("#")[-1].strip() if "#" in label else label
        gap = int(t_sent * scale)
        bar = max(int((t_done - t_sent) * scale), 1)
        print(f"  #{short:<4}  {' '*gap}{'█'*bar}  {t_done:.1f}s")
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

    cols = {k: [] for k in ["layout", "ocr_wall", "avg_q", "avg_exec", "total", "wall"]}
    for _, wall, data in results:
        if "error" in data:
            continue
        t = data.get("meta", {}).get("timing", {})
        for k, v in [("layout", t.get("layout_s")), ("ocr_wall", t.get("ocr_wall_s")),
                     ("avg_q", t.get("ocr_avg_queued_s")), ("avg_exec", t.get("ocr_avg_exec_s")),
                     ("total", t.get("total_s")), ("wall", wall)]:
            if isinstance(v, (int, float)):
                cols[k].append(v)

    W = 8
    def _f(v): return f"{v:>{W}.2f}s" if v is not None else f"{'─'*W}"
    def _row(name, lst):
        d = _dist(lst)
        if d is None: return
        mn, p50, avg, p95, mx = d
        print(f"  {name:<18}  {_f(mn)}  {_f(p50)}  {_f(avg)}  {_f(p95)}  {_f(mx)}")

    print(f"  {'metric':<18}  {'min':>{W}}  {'p50':>{W}}  {'avg':>{W}}  {'p95':>{W}}  {'max':>{W}}")
    print(f"  {'─'*18}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}")
    _row("layout_s",         cols["layout"])
    _row("ocr_wall_s",       cols["ocr_wall"])
    _row("ocr_avg_queued_s", cols["avg_q"])
    _row("ocr_avg_exec_s",   cols["avg_exec"])
    _row("total_s",          cols["total"])
    _row("wall (e2e)",        cols["wall"])

    if cols["total"] and cols["wall"]:
        avg_total = sum(cols["total"]) / len(cols["total"])
        avg_wall  = sum(cols["wall"])  / len(cols["wall"])
        print(f"\n  ── pricing  (L4 ${GPU_RATE}/s · Rp{IDR_RATE:,}/USD, avg, no concurrency) ──")
        print(f"  avg lower (server total_s) : {_pc(avg_total)}")
        print(f"  avg upper (client wall_s)  : {_pc(avg_wall)}")
        print(f"  per 1 000 req (lower)      : {_pc(avg_total * 1000)}")
        print(f"  per 1 000 req (upper)      : {_pc(avg_wall  * 1000)}")


# ── run logic ──────────────────────────────────────────────────────────────────

def run_round(label_prefix: str, n: int, endpoint: str, pdf_bytes: bytes,
              pages: list | None, dpi: int, timeline: bool = False) -> tuple[list, float | None]:
    if n == 1:
        label = label_prefix
        print(f"\nSending {label} …")
        return [send(endpoint, pdf_bytes, pages, dpi, label)], None

    print(f"\nSending {n} requests concurrently …")
    round_t0    = time.time()
    futures_map = {}
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            lbl = f"{label_prefix} #{i+1}"
            futures_map[pool.submit(send, endpoint, pdf_bytes, pages, dpi, lbl, round_t0)] = lbl

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
    ap.add_argument("--pages",      type=int, nargs="+", default=None, help="explicit page indices (0-based)")
    ap.add_argument("--num-pages",  type=int, default=None,           help="parse first N pages (0 to N-1)")
    ap.add_argument("--dpi",        type=int, default=200)
    ap.add_argument("--repeat",     type=int, default=1)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--delay",      type=float, default=0)
    ap.add_argument("--timeline",   action="store_true")
    ap.add_argument("--endpoint",   default=ENDPOINT)
    ap.add_argument("--save",       action="store_true")
    args = ap.parse_args()

    pdf_bytes = Path(args.file).read_bytes()
    stem      = Path(args.file).stem

    pages = args.pages
    if pages is None and args.num_pages is not None:
        pages = list(range(args.num_pages))

    print(f"Endpoint   : {args.endpoint}")
    print(f"File       : {args.file}  ({len(pdf_bytes):,} bytes)")
    print(f"Pages      : {pages if pages is not None else 'all'}  DPI={args.dpi}")
    print(f"Rounds     : {args.repeat}  ×  {args.concurrent} concurrent")

    all_results = []
    last_elapsed = None

    for r in range(args.repeat):
        if r > 0 and args.delay > 0:
            print(f"\n  [waiting {args.delay}s …]")
            time.sleep(args.delay)

        prefix  = f"round {r+1}" if args.repeat > 1 else "request"
        results, elapsed = run_round(prefix, args.concurrent, args.endpoint,
                                     pdf_bytes, pages, args.dpi, args.timeline)
        last_elapsed = elapsed

        single = args.repeat == 1 and args.concurrent == 1

        for label, wall, data in results:
            if args.concurrent == 1:
                print_result(label, wall, data)
            if (args.save or single) and "error" not in data:
                ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
                path    = f"{stem}_single_{label.replace(' ','_')}_{ts}"
                out_dir = Path(path)
                out_dir.mkdir(exist_ok=True)
                data_clean = {k: v for k, v in data.items() if k != "blocks"}
                data_clean["blocks"] = [
                    {k: v for k, v in b.items() if k != "image_b64"}
                    for b in data.get("blocks", [])
                ]
                (out_dir / "result.json").write_text(json.dumps(data_clean, indent=2, ensure_ascii=False))
                (out_dir / "result.md").write_text(data.get("markdown", ""))
                img_dir = out_dir / "imgs"
                img_count = 0
                for b in data.get("blocks", []):
                    if b.get("image_b64"):
                        img_bytes = base64.b64decode(b["image_b64"])
                        fname = f"p{b['page']}_r{b['order']}_{b['label']}_{img_count}.jpg"
                        (img_dir := out_dir / "imgs").mkdir(exist_ok=True)
                        (img_dir / fname).write_bytes(img_bytes)
                        img_count += 1
                print(f"  Saved: {out_dir}/  ({img_count} images)")

        all_results.extend(results)

    if len(all_results) > 1:
        print_summary(all_results, last_elapsed if args.repeat == 1 else None)


if __name__ == "__main__":
    main()
