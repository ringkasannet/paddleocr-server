"""Test script for the deployed Modal layout worker.

Usage:
    python modal/test_layout.py document.pdf
    python modal/test_layout.py document.pdf --repeat 3          # 3 sequential sends
    python modal/test_layout.py document.pdf --concurrent 4      # 4 parallel sends
    python modal/test_layout.py document.pdf --repeat 2 --concurrent 3  # 2 rounds of 3
    python modal/test_layout.py document.pdf --dpi 200
    python modal/test_layout.py document.pdf --endpoint https://your-org--layout-worker-process.modal.run
"""

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


IDR_RATE = 17500

def _usd(val) -> str:
    if not isinstance(val, (int, float)):
        return str(val)
    return f"{val:.6f}"

def _idr(val) -> str:
    if not isinstance(val, (int, float)):
        return str(val)
    return f"Rp{val * IDR_RATE:.2f}"

ENDPOINT = "https://ringkasan-net--layout-worker-process.modal.run"


def send(endpoint: str, file_bytes: bytes, file_type: int, dpi: int, label: str) -> tuple[str, float, dict]:
    t_encode = time.time()
    b64 = base64.b64encode(file_bytes).decode()
    encode_s = round(time.time() - t_encode, 3)
    t0 = time.time()
    try:
        resp = requests.post(
            endpoint,
            json={"file": b64, "fileType": file_type, "dpi": dpi},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        data["_client"] = {"encode_s": encode_s, "wall_s": round(time.time() - t0, 3)}
        return label, round(time.time() - t0, 3), data
    except requests.HTTPError as e:
        return label, round(time.time() - t0, 3), {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return label, round(time.time() - t0, 3), {"error": str(e)}


def print_result(label: str, wall: float, data: dict, show_regions: bool = True):
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")

    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return

    pages = data.get("pages", [])
    meta  = data.get("meta", {})
    t     = meta.get("timing", {})
    cost  = meta.get("cost", {})

    client = data.get("_client", {})
    print(f"  Pages          : {len(pages)}")
    print(f"  Searchable     : {data.get('searchable')}")
    print(f"  Total regions  : {meta.get('total_regions', '?')}")

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

    if t or client:
        print(f"\n  ── timing ──")
        if client:
            print(f"  encode (client): {client.get('encode_s', '?')}s")
        print(f"  wall (client)  : {wall:.3f}s")
        if t:
            print(f"  queued         : {t.get('queued_s', '?')}s  (Modal dispatch latency)")
            print(f"  render         : {t.get('render_s', '?')}s")
            print(f"  detect         : {t.get('detect_s', '?')}s")
            print(f"  text extract   : {t.get('text_s', '?')}s")
            print(f"  execution      : {t.get('execution_s', '?')}s  (render+detect+text)")
            print(f"  wall (server)  : {t.get('wall_s', '?')}s")
    if cost:
        print(f"\n  ── cost ({cost.get('gpu','?')} @ ${cost.get('rate_per_s','?')}/s) ──")
        est_lo  = cost.get('estimated_usd_lower')
        ppu_lo  = cost.get('per_page_usd_lower')
        est_q   = cost.get('estimated_usd_queued')
        ppu_q   = cost.get('per_page_usd_queued')
        est_wl  = cost.get('estimated_usd_wall')
        ppu_wl  = cost.get('per_page_usd_wall')
        print(f"  lower  (exec+idle      = {cost.get('billed_s_lower','?'):>7}s):  ${_usd(est_lo)}  {_idr(ppu_lo)}/page")
        if est_q is not None:
            print(f"  queued (queue+exec+idle= {cost.get('billed_s_queued','?'):>7}s):  ${_usd(est_q)}  {_idr(ppu_q)}/page  ← best estimate")
        print(f"  wall   (wall+idle      = {cost.get('billed_s_wall','?'):>7}s):  ${_usd(est_wl)}  {_idr(ppu_wl)}/page")
        print(f"  note   : {cost.get('note', '')}")


def print_summary(results: list[tuple[str, float, dict]]):
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    print(f"  {'Label':<20s}  {'Wall':>6s}  {'Exec':>7s}  {'Pages':>5s}  {'Cost(queue)':>13s}  {'Rp/pg(queue)':>13s}  {'Cost(wall)':>12s}  {'Rp/pg(wall)':>13s}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*13}  {'─'*13}  {'─'*12}  {'─'*13}")

    total_wall       = 0.0
    total_exec       = 0.0
    total_pages      = 0
    total_cost_queue = 0.0
    total_cost_wall  = 0.0
    errors = 0

    for label, wall, data in results:
        if "error" in data:
            print(f"  {label:<20s}  {wall:>6.2f}s  ERROR")
            errors += 1
            continue
        t      = data.get("meta", {}).get("timing", {})
        cost   = data.get("meta", {}).get("cost", {})
        pages  = data.get("meta", {}).get("page_count", 0)
        exec_s    = t.get("execution_s", 0)
        usd_queue = cost.get('estimated_usd_queued') or cost.get('estimated_usd_lower', 0)
        ppu_queue = cost.get('per_page_usd_queued') or cost.get('per_page_usd_lower')
        usd_wall  = cost.get('estimated_usd_wall', 0)
        ppu_wall  = cost.get('per_page_usd_wall')

        total_wall       += wall
        total_exec       += exec_s if isinstance(exec_s, (int, float)) else 0
        total_pages      += pages if isinstance(pages, int) else 0
        total_cost_queue += usd_queue if isinstance(usd_queue, (int, float)) else 0
        total_cost_wall  += usd_wall if isinstance(usd_wall, (int, float)) else 0
        print(f"  {label:<20s}  {wall:>6.2f}s"
              f"  {str(exec_s):>6s}s"
              f"  {str(pages):>5s}"
              f"  ${_usd(usd_queue):>12s}"
              f"  {_idr(ppu_queue):>13s}"
              f"  ${_usd(usd_wall):>11s}"
              f"  {_idr(ppu_wall):>13s}")

    successful = len(results) - errors
    if successful > 1:
        ppu_queue_total = round(total_cost_queue / total_pages, 6) if total_pages else None
        ppu_wall_total  = round(total_cost_wall  / total_pages, 6) if total_pages else None
        print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*13}  {'─'*13}  {'─'*12}  {'─'*13}")
        print(f"  {'TOTAL':<20s}  {total_wall:>6.2f}s"
              f"  {round(total_exec,3):>6.3f}s"
              f"  {total_pages:>5d}"
              f"  ${_usd(total_cost_queue):>12s}"
              f"  {_idr(ppu_queue_total) if ppu_queue_total else '?':>13s}"
              f"  ${_usd(total_cost_wall):>11s}"
              f"  {_idr(ppu_wall_total) if ppu_wall_total else '?':>13s}")

    print(f"\n  * queue estimate is most accurate; wall is upper bound; both exclude the first cold start after deploy")


def run_round(label_prefix: str, n: int, endpoint: str,
              file_bytes: bytes, file_type: int, dpi: int) -> list[tuple[str, float, dict]]:
    if n == 1:
        label = label_prefix
        print(f"\nSending {label} …")
        result = send(endpoint, file_bytes, file_type, dpi, label)
        return [result]

    print(f"\nSending {n} requests concurrently …")
    t_round = time.time()
    futures_map = {}
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            label = f"{label_prefix} #{i+1}"
            fut = pool.submit(send, endpoint, file_bytes, file_type, dpi, label)
            futures_map[fut] = label

    results = []
    for fut in as_completed(futures_map):
        results.append(fut.result())
    results.sort(key=lambda r: r[0])  # sort by label

    print(f"  Round finished in {time.time() - t_round:.2f}s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file",          help="Path to PDF or image file")
    ap.add_argument("--dpi",         type=int, default=150)
    ap.add_argument("--image",       action="store_true", help="Treat input as image (fileType=1)")
    ap.add_argument("--endpoint",    default=ENDPOINT)
    ap.add_argument("--repeat",      type=int, default=1,
                    help="Number of sequential rounds (default 1)")
    ap.add_argument("--concurrent",  type=int, default=1,
                    help="Number of parallel requests per round (default 1)")
    ap.add_argument("--no-regions",  action="store_true", help="Skip printing region detail")
    ap.add_argument("--save",        action="store_true", help="Save each response as JSON")
    args = ap.parse_args()

    file_type  = 1 if args.image else 0
    file_bytes = open(args.file, "rb").read()
    stem       = Path(args.file).stem

    print(f"Endpoint   : {args.endpoint}")
    print(f"File       : {args.file}  ({len(file_bytes):,} bytes)")
    print(f"Type       : {'image' if args.image else 'PDF'}  DPI={args.dpi}")
    print(f"Rounds     : {args.repeat}  ×  {args.concurrent} concurrent")

    all_results: list[tuple[str, float, dict]] = []

    for r in range(args.repeat):
        prefix = f"round {r+1}" if args.repeat > 1 else "request"
        results = run_round(prefix, args.concurrent, args.endpoint,
                            file_bytes, file_type, args.dpi)
        for label, wall, data in results:
            print_result(label, wall, data, show_regions=not args.no_regions)
            if args.save:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe = label.replace(" ", "_").replace("#", "")
                out_path = f"{stem}_{safe}_{ts}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  Saved: {out_path}")
        all_results.extend(results)

    if len(all_results) > 1:
        print_summary(all_results)

    if len(all_results) == 1 and not args.save:
        _, _, data = all_results[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{stem}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
