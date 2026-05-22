"""Test script for the deployed Modal layout worker.

Usage:
    python modal/test_layout.py document.pdf
    python modal/test_layout.py document.pdf --repeat 3
    python modal/test_layout.py document.pdf --concurrent 30
    python modal/test_layout.py document.pdf --repeat 2 --concurrent 10
    python modal/test_layout.py document.pdf --no-regions
    python modal/test_layout.py document.pdf --timeline     # show ASCII bars
    python modal/test_layout.py document.pdf --save
    python modal/test_layout.py image.jpg --image
    python modal/test_layout.py document.pdf --endpoint https://your-endpoint.modal.run
"""

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

IDR_RATE             = 17500
COLD_THRESHOLD       = 5.0    # queued_s above this → cold start
ENDPOINT             = "https://ringkasan-net--layout-worker-processor-process.modal.run"
DETECT_MAX_INPUTS    = 4      # @modal.concurrent max_inputs on LayoutDetector
GPU_SCALEDOWN_S      = 60     # scaledown_window on LayoutDetector (idle billed per container)
GPU_RATES            = {"T4": 0.000164, "L4": 0.000222}


def _fmt(v, unit="s", decimals=3) -> str:
    if not isinstance(v, (int, float)):
        return "  ?"
    return f"{v:.{decimals}f}{unit}"

def _container_label(queued_s) -> str:
    if queued_s is None:
        return "?"
    if queued_s > COLD_THRESHOLD:
        return f"COLD  {queued_s:.1f}s restore"
    return f"warm  {queued_s:.2f}s dispatch"


# ── network call ──────────────────────────────────────────────────────────────

def send(endpoint: str, file_bytes: bytes, file_type: int, dpi: int,
         label: str, round_t0: float | None = None) -> tuple[str, float, dict]:
    t0 = time.time()
    t_sent_offset = round(t0 - round_t0, 3) if round_t0 is not None else 0.0
    t_enc = time.time()
    b64 = base64.b64encode(file_bytes).decode()
    encode_s = round(time.time() - t_enc, 3)
    try:
        # stream=True: t_first_byte is when server finishes processing and starts sending
        with requests.post(
            endpoint, timeout=600,
            json={"file": b64, "fileType": file_type, "dpi": dpi},
            stream=True,
        ) as resp:
            resp.raise_for_status()
            t_first_byte = time.time()           # server is done, response headers received
            raw = resp.content                   # now download the body
            t_last_byte  = time.time()
            resp_bytes   = len(raw)

        data = json.loads(raw)
        wall = round(time.time() - t0, 3)
        t_done_offset  = round(time.time() - round_t0, 3) if round_t0 is not None else wall
        upload_s       = round(t_first_byte - t0 - encode_s, 3)   # encode already subtracted
        download_s     = round(t_last_byte - t_first_byte, 3)
        data["_client"] = {
            "encode_s":      encode_s,
            "upload_s":      upload_s,      # time from send to first response byte (upload + server)
            "download_s":    download_s,    # time to receive full response body
            "resp_bytes":    resp_bytes,    # response size in bytes
            "wall_s":        wall,
            "t_sent_offset": t_sent_offset,
            "t_done_offset": t_done_offset,
        }
        return label, wall, data
    except requests.HTTPError as e:
        wall = round(time.time() - t0, 3)
        t_done_offset = round(time.time() - round_t0, 3) if round_t0 is not None else wall
        return label, wall, {
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            "_client": {"wall_s": wall, "t_sent_offset": t_sent_offset, "t_done_offset": t_done_offset},
        }
    except Exception as e:
        wall = round(time.time() - t0, 3)
        t_done_offset = round(time.time() - round_t0, 3) if round_t0 is not None else wall
        return label, wall, {
            "error": str(e),
            "_client": {"wall_s": wall, "t_sent_offset": t_sent_offset, "t_done_offset": t_done_offset},
        }


# ── event timeline ────────────────────────────────────────────────────────────

def _print_timeline(t: dict, client: dict, wall: float):
    """Absolute elapsed-time timeline anchored to t=0 (client start)."""

    enc_s    = client.get("encode_s")    or 0.0
    upload_s = client.get("upload_s")    or 0.0   # = net + srv_wall
    dl_s     = client.get("download_s")  or 0.0
    wall_s   = t.get("wall_s")           or 0.0
    net_s    = max(round(upload_s - wall_s, 3), 0.0)

    render_s  = t.get("render_s")     or 0.0
    jpeg_s    = t.get("jpeg_encode_s") or 0.0
    jpeg_kb   = t.get("jpeg_kb")
    queued_s  = t.get("queued_s")
    detect_s  = t.get("detect_s")     or 0.0
    text_s    = t.get("text_s")       or 0.0
    post_s    = t.get("post_s")       or 0.0

    # All timestamps are elapsed seconds from t=0 (start of send())
    t_sent        = enc_s
    t_proc        = enc_s + net_s                               # Processor receives
    t_render      = t_proc + render_s                           # render done
    t_jpeg        = t_render + jpeg_s                           # JPEG encode done
    t_rpc         = t_jpeg                                      # detect.remote() called
    t_gpu_start   = t_rpc + queued_s if queued_s is not None else None
    t_gpu_done    = t_gpu_start + detect_s if t_gpu_start is not None else None
    t_text        = (t_gpu_done or t_rpc) + text_s
    t_post        = t_text + post_s
    t_srv_done    = t_proc + wall_s                             # processor sends response
    t_first_byte  = enc_s + upload_s                           # client receives first byte
    t_complete    = t_first_byte + dl_s                         # client receives last byte

    cold = queued_s is not None and queued_s > COLD_THRESHOLD

    def _ts(v):
        return f"t={v:>7.3f}s" if v is not None else "t=      ?s"

    def _bar(duration, char="─"):
        n = max(1, int(duration * 2))
        return char * min(n, 32)

    print(f"\n  ── event timeline ──────────────────────────────────────")
    print(f"  {_ts(0.0)}  [client]     encode start")
    print(f"  {_ts(t_sent)}  [client]     request sent  (+{enc_s:.3f}s encode)")
    print(f"             ╎  {_bar(net_s, '·')} network {net_s:.3f}s")
    print(f"  {_ts(t_proc)}  [processor]  received · render starts")
    print(f"  {_ts(t_render)}  [processor]  render done  (+{render_s:.3f}s)")
    if jpeg_s > 0:
        kb_note = f"  {jpeg_kb} KB" if jpeg_kb else ""
        print(f"  {_ts(t_jpeg)}  [processor]  JPEG encoded{kb_note}  (+{jpeg_s:.3f}s)")
    print(f"  {_ts(t_rpc)}  [processor]  detect.remote() called")
    if queued_s is not None:
        label = f"COLD {queued_s:.2f}s" if cold else f"warm {queued_s:.3f}s"
        print(f"             ╎  {_bar(queued_s, '░' if cold else '·')} dispatch {label}")
    if t_gpu_start is not None:
        print(f"  {_ts(t_gpu_start)}  [gpu]        detect() started")
    if t_gpu_done is not None:
        print(f"  {_ts(t_gpu_done)}  [gpu]        detect() done  (+{detect_s:.3f}s)")
        print(f"  {_ts(t_gpu_done)}  [processor]  result received · text+nms starts")
    print(f"  {_ts(t_text)}  [processor]  text+nms done  (+{text_s:.3f}s)")
    if post_s > 0.001:
        print(f"  {_ts(t_post)}  [processor]  post done  (+{post_s:.3f}s)")
    print(f"  {_ts(t_srv_done)}  [processor]  response sent  (wall {wall_s:.3f}s)")
    print(f"             ╎  {_bar(dl_s, '·')} download {dl_s:.3f}s")
    print(f"  {_ts(t_first_byte)}  [client]     first byte  (ttfb {upload_s:.3f}s)")
    print(f"  {_ts(t_complete)}  [client]     complete  (wall {wall:.3f}s)")


# ── per-request detail ────────────────────────────────────────────────────────

def print_result(label: str, wall: float, data: dict, show_regions: bool = True):
    print(f"\n{'─' * 54}")
    print(f"  {label}")
    print(f"{'─' * 54}")

    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return

    pages  = data.get("pages", [])
    meta   = data.get("meta", {})
    t      = meta.get("timing", {})
    cost   = meta.get("cost", {})
    client = data.get("_client", {})

    print(f"  Pages         : {len(pages)}")
    print(f"  Searchable    : {data.get('searchable')}")
    print(f"  Total regions : {meta.get('total_regions', '?')}")

    if show_regions:
        for page in pages:
            regions = page["regions"]
            print(f"\n  Page {page['page_num']}  ({page['width_px']}×{page['height_px']}px)"
                  f"  — {len(regions)} regions")
            for r in regions[:8]:
                x0, y0, x1, y1 = r["bbox"]
                text_preview = (r["text"][:40].replace("\n", " / ") + "…") if r["text"] else ""
                print(f"    [{r['order']:3d}] {r['type']:<22s}  score={r['score']:.3f}"
                      f"  bbox=({x0},{y0})-({x1},{y1})  {text_preview}")
            if len(regions) > 8:
                print(f"    … and {len(regions) - 8} more")

    render_s = t.get("render_s")
    queued_s = t.get("queued_s")
    detect_s = t.get("detect_s")
    text_s   = t.get("text_s")
    post_s   = t.get("post_s")
    wall_s   = t.get("wall_s")

    encode_s   = client.get("encode_s")
    upload_s   = client.get("upload_s")
    download_s = client.get("download_s")
    resp_bytes = client.get("resp_bytes")
    net_s      = round(upload_s - wall_s, 3) if isinstance(upload_s, (int, float)) and isinstance(wall_s, (int, float)) else None

    W = 8
    def _f(v, note=""): return f"{_fmt(v):>{W}}   {note}" if note else f"{_fmt(v):>{W}}"

    jpeg_encode_s = t.get("jpeg_encode_s")
    jpeg_kb       = t.get("jpeg_kb")
    rpc_wall_s    = t.get("rpc_wall_s")

    print(f"\n  ── timing ──────────────────────────────────────────────")
    print(f"  [client]")
    print(f"  encode          : {_f(encode_s)}")
    print(f"  upload_net      : {_f(net_s,  '← pure network (ttfb − srv_wall)')}")
    print(f"  download        : {_f(download_s, f'← {resp_bytes:,} bytes' if resp_bytes else '')}")
    print(f"")
    print(f"  [server — cpu]")
    print(f"  render          : {_f(render_s,      '← PDF → PIL pages')}")
    print(f"  jpeg_encode     : {_f(jpeg_encode_s, f'← {jpeg_kb} KB JPEG payload' if jpeg_kb else '← JPEG encode')}")
    print(f"  text + nms      : {_f(text_s,        '← NMS + reading order + text extract')}")
    print(f"  post            : {_f(post_s,        '← regulation rules + reclassify + merge')}")
    print(f"")
    print(f"  [server — rpc  cpu→gpu]")
    rpc_note = f"← {jpeg_kb} KB sent  ({_fmt(queued_s)} dispatch + {_fmt(detect_s)} detect)" if jpeg_kb else ""
    print(f"  rpc_wall        : {_f(rpc_wall_s, rpc_note)}")
    print(f"  dispatch        : {_f(queued_s)}   {_container_label(queued_s)}")
    print(f"  detect          : {_f(detect_s, '← DETR inference')}")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  server wall     : {_f(wall_s)}")
    print(f"  ttfb            : {_f(upload_s, '← upload_net + srv_wall')}")
    print(f"  client wall     : {_f(wall)}")

    # ── event timeline ────────────────────────────────────────────────────
    _print_timeline(t, client, wall)

    if cost:
        usd_q  = cost.get("estimated_usd_queued") or cost.get("estimated_usd_lower")
        ppu_q  = cost.get("per_page_usd_queued")  or cost.get("per_page_usd_lower")
        usd_w  = cost.get("estimated_usd_wall")
        ppu_w  = cost.get("per_page_usd_wall")
        gpu    = cost.get("gpu", "?")
        rate   = cost.get("rate_per_s", "?")
        print(f"\n  ── cost ({gpu} @ ${rate}/s) ──────────────")
        if usd_q is not None:
            print(f"  queued est : ${usd_q:.6f}   Rp{usd_q * IDR_RATE:.2f}  ({_fmt(ppu_q * IDR_RATE, 'Rp', 2) if ppu_q else '?'}/page)")
        print(f"  wall   est : ${usd_w:.6f}   Rp{usd_w * IDR_RATE:.2f}  ({_fmt(ppu_w * IDR_RATE, 'Rp', 2) if ppu_w else '?'}/page)")


# ── concurrent timing table ───────────────────────────────────────────────────

def print_timing_table(results: list[tuple[str, float, dict]]):
    """One row per request: render / dispatch / detect / text / walls."""
    print(f"\n  ── per-request timing ──────────────────────────────────────────────────────────────────────────────────────")
    print(f"  {'req':<10}  {'render':>7}  {'queue':>7}  {'detect':>7}  {'text':>6}  {'srv_wall':>8}  {'ttfb':>7}  {'net_s':>6}  {'download':>8}  {'resp_kb':>7}")
    print(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*7}")

    render_vals, queue_vals, detect_vals, text_vals = [], [], [], []
    srv_vals, ttfb_vals, net_vals, download_vals, size_vals = [], [], [], [], []

    for label, _, data in results:
        short = label.split("#")[-1].strip() if "#" in label else label
        if "error" in data:
            print(f"  {short:<10}  {'ERROR':>7}")
            continue
        t          = data.get("meta", {}).get("timing", {})
        client     = data.get("_client", {})
        render_s   = t.get("render_s")
        queued_s   = t.get("queued_s")
        detect_s   = t.get("detect_s")
        text_s     = t.get("text_s")
        wall_s     = t.get("wall_s")
        ttfb_s     = client.get("upload_s")
        download_s = client.get("download_s")
        resp_kb    = round(client.get("resp_bytes", 0) / 1024, 1) if client.get("resp_bytes") else None
        net_s      = round(ttfb_s - wall_s, 3) if isinstance(ttfb_s, (int, float)) and isinstance(wall_s, (int, float)) else None

        def _c(v, w=7): return f"{v:>{w}.2f}s" if isinstance(v, (int, float)) else f"{'?':>{w}}"
        def _kb(v):     return f"{v:>6.1f}k"   if isinstance(v, (int, float)) else f"{'?':>7}"

        print(f"  {short:<10}  {_c(render_s)}  {_c(queued_s)}  {_c(detect_s)}  {_c(text_s,6)}  {_c(wall_s,8)}  {_c(ttfb_s)}  {_c(net_s,6)}  {_c(download_s,8)}  {_kb(resp_kb)}")

        for lst, v in [(render_vals, render_s), (queue_vals, queued_s), (detect_vals, detect_s),
                       (text_vals, text_s), (srv_vals, wall_s), (ttfb_vals, ttfb_s),
                       (net_vals, net_s), (download_vals, download_s), (size_vals, resp_kb)]:
            if isinstance(v, (int, float)):
                lst.append(v)

    def _stat(lst, w=7):
        if not lst:
            return f"{'?':>{w}}", f"{'?':>{w}}"
        return f"{sum(lst)/len(lst):>{w}.2f}s", f"{max(lst):>{w}.2f}s"

    print(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*7}")
    rows = [("avg", [_stat(l,w)[0] for l,w in zip([render_vals,queue_vals,detect_vals,text_vals,srv_vals,ttfb_vals,net_vals,download_vals],[7,7,7,6,8,7,6,8])]),
            ("max", [_stat(l,w)[1] for l,w in zip([render_vals,queue_vals,detect_vals,text_vals,srv_vals,ttfb_vals,net_vals,download_vals],[7,7,7,6,8,7,6,8])])]
    for name, vals in rows:
        print(f"  {name:<10}  {'  '.join(vals)}")


# ── ASCII timeline (opt-in) ───────────────────────────────────────────────────

def print_ascii_timeline(results: list[tuple[str, float, dict]], round_wall: float):
    WIDTH = 48
    scale = WIDTH / max(round_wall, 1.0)
    unit  = round_wall / WIDTH
    print(f"\n  ── timeline  (1 char ≈ {unit:.1f}s) ─────────────────────────────────────")
    for label, wall, data in results:
        client   = data.get("_client", {})
        t_sent   = client.get("t_sent_offset", 0.0)
        t_done   = client.get("t_done_offset", wall)
        t        = data.get("meta", {}).get("timing", {}) if "error" not in data else {}
        queued_s = t.get("queued_s")
        gap  = int(t_sent * scale)
        bar  = max(int((t_done - t_sent) * scale), 1)
        cold = queued_s is not None and queued_s > COLD_THRESHOLD
        ch   = "░" if cold else "█"
        flag = " ← COLD" if cold else ""
        short = label.split("#")[-1].strip() if "#" in label else label
        print(f"  #{short:<3}  {' '*gap}{ch*bar}  {t_done:.1f}s{flag}")
    step = max(1, int(round_wall / 8))
    axis = ""
    for tv in range(0, int(round_wall) + step, step):
        pos = int(tv * scale)
        axis = axis.ljust(pos + 6) + f"{tv}s"
    print(f"        {axis}")


# ── summary table ─────────────────────────────────────────────────────────────

def _dist(lst: list[float]):
    """Return (min, p50, avg, p95, max) for a list, or None if empty."""
    if not lst:
        return None
    s = sorted(lst)
    n = len(s)
    def _p(pct): return s[min(int(n * pct / 100), n - 1)]
    return s[0], _p(50), sum(s) / n, _p(95), s[-1]


def print_summary(results: list[tuple[str, float, dict]], round_elapsed: float | None = None):
    print(f"\n{'═' * 72}")
    print("  SUMMARY")
    if round_elapsed is not None:
        n_ok = sum(1 for _, _, d in results if "error" not in d)
        print(f"  {n_ok} requests  |  round elapsed: {round_elapsed:.2f}s  |  avg throughput: {n_ok/round_elapsed:.2f} req/s")
    print(f"{'═' * 72}")
    print(f"  {'Label':<20}  {'srv_wall':>8}  {'ttfb':>7}  {'net_s':>6}  {'download':>8}  {'render':>6}  {'queue':>6}  {'detect':>7}  {'text':>5}  {'pages':>5}  {'cost(q)':>10}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*10}")

    total_srv = total_ttfb = total_net = total_download = 0.0
    total_queue = total_detect = total_render = total_text = 0.0
    total_pages = total_cost = 0.0
    errors = 0

    all_wall: list[float] = []
    all_srv: list[float]  = []
    all_ttfb: list[float] = []
    all_net: list[float]  = []
    all_dl: list[float]   = []
    all_q: list[float]    = []
    all_det: list[float]  = []
    all_rnd: list[float]  = []
    all_txt: list[float]  = []

    for label, _, data in results:
        if "error" in data:
            print(f"  {label:<20}  ERROR")
            errors += 1
            continue
        t          = data.get("meta", {}).get("timing", {})
        cost       = data.get("meta", {}).get("cost", {})
        client     = data.get("_client", {})
        pages      = data.get("meta", {}).get("page_count", 0)
        render_s   = t.get("render_s")  or 0
        queued_s   = t.get("queued_s")  or 0
        detect_s   = t.get("detect_s")  or 0
        text_s     = t.get("text_s")    or 0
        srv_wall   = t.get("wall_s")    or 0
        ttfb_s     = client.get("upload_s",   0) or 0
        download_s = client.get("download_s", 0) or 0
        wall_s     = client.get("wall_s",     0) or 0
        net_s      = round(ttfb_s - srv_wall, 3) if srv_wall else 0
        usd_q      = cost.get("estimated_usd_queued") or cost.get("estimated_usd_lower", 0) or 0

        total_srv      += srv_wall
        total_ttfb     += ttfb_s
        total_net      += net_s
        total_download += download_s
        total_render   += render_s
        total_queue    += queued_s
        total_detect   += detect_s
        total_text     += text_s
        total_pages    += pages if isinstance(pages, int) else 0
        total_cost     += usd_q if isinstance(usd_q, (int, float)) else 0

        if wall_s:   all_wall.append(wall_s)
        if srv_wall: all_srv.append(srv_wall)
        if ttfb_s:   all_ttfb.append(ttfb_s)
        if net_s:    all_net.append(net_s)
        if download_s: all_dl.append(download_s)
        if queued_s: all_q.append(queued_s)
        if detect_s: all_det.append(detect_s)
        if render_s: all_rnd.append(render_s)
        if text_s:   all_txt.append(text_s)

        def _s(v, w=6): return f"{v:>{w}.2f}s"
        print(f"  {label:<20}  {_s(srv_wall,8)}  {_s(ttfb_s,7)}  {_s(net_s,6)}  {_s(download_s,8)}  {_s(render_s)}  {_s(queued_s)}   {_s(detect_s)}  {_s(text_s,5)}  {str(pages):>5}  ${usd_q:>9.6f}")

    n = len(results) - errors
    if n > 1:
        ppu_avg = total_cost / total_pages if total_pages else 0
        print(f"  {'─'*20}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*10}")
        def _s(v, w=6): return f"{v:>{w}.2f}s"
        print(f"  {'avg':<20}  {_s(total_srv/n,8)}  {_s(total_ttfb/n,7)}  {_s(total_net/n,6)}  {_s(total_download/n,8)}  {_s(total_render/n)}  {_s(total_queue/n)}   {_s(total_detect/n)}  {_s(total_text/n,5)}"
              f"       ${total_cost/n:>9.6f}")
        print(f"  {'TOTAL':<20}  {_s(total_srv,8)}  {_s(total_ttfb,7)}  {_s(total_net,6)}  {_s(total_download,8)}  {_s(total_render)}  {_s(total_queue)}   {_s(total_detect)}  {_s(total_text,5)}"
              f"  {int(total_pages):>5}  ${total_cost:>9.6f}  Rp{ppu_avg * IDR_RATE:.2f}/pg")

        # ── distribution stats ────────────────────────────────────────────────
        W = 7
        def _f(v): return f"{v:>{W}.2f}s" if v is not None else f"{'─'*W}"
        def _row(label, lst):
            d = _dist(lst)
            if d is None:
                return
            mn, p50, avg, p95, mx = d
            print(f"  {label:<16}  {_f(mn)}  {_f(p50)}  {_f(avg)}  {_f(p95)}  {_f(mx)}")

        print(f"\n  {'─'*72}")
        print(f"  {'metric':<16}  {'min':>{W}}  {'p50':>{W}}  {'avg':>{W}}  {'p95':>{W}}  {'max':>{W}}")
        print(f"  {'─'*16}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}")
        _row("wall (e2e)",  all_wall)
        _row("srv_wall",    all_srv)
        _row("ttfb",        all_ttfb)
        _row("net_s",       all_net)
        _row("download",    all_dl)
        _row("detect",      all_det)
        _row("queue",       all_q)
        _row("render",      all_rnd)
        _row("text",        all_txt)

        # ── batch cost breakdown ──────────────────────────────────────────────
        import math
        rate = next(
            (d.get("meta", {}).get("cost", {}).get("rate_per_s")
             for _, _, d in results if "error" not in d),
            GPU_RATES["T4"],
        )
        gpu  = next(
            (d.get("meta", {}).get("cost", {}).get("gpu", "T4")
             for _, _, d in results if "error" not in d),
            "T4",
        )
        n_containers  = max(1, math.ceil(n / DETECT_MAX_INPUTS))
        compute_s     = sum(all_det)
        idle_s_total  = n_containers * GPU_SCALEDOWN_S
        cost_compute  = compute_s    * rate
        cost_idle     = idle_s_total * rate
        cost_batch    = cost_compute + cost_idle          # cold / one-off run
        cost_warm     = cost_compute                      # steady-state: containers stay warm
        ppu_batch     = cost_batch / total_pages if total_pages else 0
        ppu_warm      = cost_warm  / total_pages if total_pages else 0
        ppu_naive     = total_cost / total_pages if total_pages else 0

        print(f"\n  {'─'*72}")
        print(f"  GPU cost breakdown  ({gpu} @ ${rate}/s  |  {n} reqs  |  ~{n_containers} containers × max_inputs={DETECT_MAX_INPUTS})")
        print(f"  {'─'*72}")
        print(f"  {'component':<22}  {'seconds':>9}  {'cost':>10}  {'per-req':>10}  {'per-page':>10}")
        print(f"  {'─'*22}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*10}")
        idle_label = f"idle ({n_containers}×{GPU_SCALEDOWN_S}s scaledown)"
        if total_pages:
            print(f"  {'compute (detect)':<22}  {compute_s:>8.1f}s  ${cost_compute:>9.4f}  ${cost_compute/n:>9.4f}  Rp{cost_compute/total_pages*IDR_RATE:>7.2f}")
            print(f"  {idle_label:<22}  {idle_s_total:>8.1f}s  ${cost_idle:>9.4f}  ${cost_idle/n:>9.4f}  Rp{cost_idle/total_pages*IDR_RATE:>7.2f}")
        print(f"  {'─'*22}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*10}")
        print(f"  {'cold / one-off':<22}  {compute_s+idle_s_total:>8.1f}s  ${cost_batch:>9.4f}  ${cost_batch/n:>9.4f}  Rp{ppu_batch*IDR_RATE:>7.2f}")
        print(f"  {'warm / steady-state':<22}  {compute_s:>8.1f}s  ${cost_warm:>9.4f}  ${cost_warm/n:>9.4f}  Rp{ppu_warm*IDR_RATE:>7.2f}")
        print(f"  {'naive (per-req sum)':<22}             ${total_cost:>9.4f}  ${total_cost/n:>9.4f}  Rp{ppu_naive*IDR_RATE:>7.2f}  ← current per-req model")

    print(f"\n  ttfb = upload_net + srv_wall;  net_s = ttfb − srv_wall;  wall = full client round-trip")


# ── run logic ─────────────────────────────────────────────────────────────────

def run_round(label_prefix: str, n: int, endpoint: str,
              file_bytes: bytes, file_type: int, dpi: int,
              show_timeline: bool = False,
              gap: float = 0.0, gap_max: float | None = None,
              ) -> tuple[list[tuple[str, float, dict]], float | None]:
    import random

    if n == 1:
        label = label_prefix
        print(f"\nSending {label} …")
        return [send(endpoint, file_bytes, file_type, dpi, label)], None

    if gap > 0 or gap_max is not None:
        lo, hi = gap, (gap_max if gap_max is not None else gap)
        mode = f"random {lo:.2f}–{hi:.2f}s gap" if hi > lo else f"fixed {lo:.2f}s gap"
    else:
        mode = "burst"
    print(f"\nSending {n} requests concurrently  [{mode}] …")

    round_t0 = time.time()
    futures_map = {}
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            lbl = f"{label_prefix} #{i+1}"
            futures_map[pool.submit(send, endpoint, file_bytes, file_type, dpi, lbl, round_t0)] = lbl
            if i < n - 1 and (gap > 0 or gap_max is not None):
                lo, hi = gap, (gap_max if gap_max is not None else gap)
                time.sleep(random.uniform(lo, hi) if hi > lo else lo)

    results = [fut.result() for fut in as_completed(futures_map)]
    results.sort(key=lambda r: r[0])

    round_elapsed = time.time() - round_t0
    print(f"  Round finished in {round_elapsed:.2f}s")
    print_timing_table(results)
    if show_timeline:
        print_ascii_timeline(results, round_elapsed)
    return results, round_elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--dpi",        type=int, default=200)
    ap.add_argument("--image",      action="store_true")
    ap.add_argument("--endpoint",   default=ENDPOINT)
    ap.add_argument("--repeat",     type=int, default=1)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--gap",        type=float, default=0.0,  metavar="S", help="Fixed delay between request dispatches (seconds)")
    ap.add_argument("--gap-max",    type=float, default=None, metavar="S", help="If set, gap is random uniform between --gap and --gap-max")
    ap.add_argument("--detail",     action="store_true", help="Print full per-request detail block")
    ap.add_argument("--no-regions", action="store_true")
    ap.add_argument("--timeline",   action="store_true", help="Print ASCII bar timeline")
    ap.add_argument("--save",       action="store_true")
    args = ap.parse_args()

    file_type  = 1 if args.image else 0
    file_bytes = open(args.file, "rb").read()
    stem       = Path(args.file).stem

    print(f"Endpoint   : {args.endpoint}")
    print(f"File       : {args.file}  ({len(file_bytes):,} bytes)")
    print(f"Type       : {'image' if args.image else 'PDF'}  DPI={args.dpi}")
    if args.gap_max is not None:
        gap_desc = f"random {args.gap:.2f}–{args.gap_max:.2f}s gap"
    elif args.gap > 0:
        gap_desc = f"fixed {args.gap:.2f}s gap"
    else:
        gap_desc = "burst"
    print(f"Rounds     : {args.repeat}  ×  {args.concurrent} concurrent  [{gap_desc}]")

    all_results: list[tuple[str, float, dict]] = []
    last_round_elapsed: float | None = None

    for r in range(args.repeat):
        prefix  = f"round {r+1}" if args.repeat > 1 else "request"
        results, round_elapsed = run_round(prefix, args.concurrent, args.endpoint,
                                           file_bytes, file_type, args.dpi,
                                           show_timeline=args.timeline,
                                           gap=args.gap, gap_max=args.gap_max)
        last_round_elapsed = round_elapsed
        for label, wall, data in results:
            if args.detail or len(results) == 1:
                print_result(label, wall, data, show_regions=not args.no_regions)
            if args.save:
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe     = label.replace(" ", "_").replace("#", "")
                out_path = f"{stem}_{safe}_{ts}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  Saved: {out_path}")
        all_results.extend(results)

    if len(all_results) > 1:
        elapsed = last_round_elapsed if args.repeat == 1 else None
        print_summary(all_results, round_elapsed=elapsed)

    if len(all_results) == 1 and not args.save:
        _, _, data = all_results[0]
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{stem}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
