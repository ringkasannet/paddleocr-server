#!/usr/bin/env python3
"""
Test script for the PP-DocLayoutV3 RunPod serverless endpoint.

Usage:
    export RUNPOD_API_KEY=your_key_here

    python test_layout.py <endpoint_id> document.pdf
    python test_layout.py <endpoint_id> document.pdf --dpi 200
    python test_layout.py <endpoint_id> document.pdf --pages 2
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

POLL_INTERVAL = 2  # seconds


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        sys.exit("ERROR: set RUNPOD_API_KEY environment variable")
    return key


def headers() -> dict:
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def submit(base_url: str, pdf_bytes: bytes, dpi: int) -> str:
    b64 = base64.b64encode(pdf_bytes).decode()
    resp = requests.post(
        f"{base_url}/run",
        headers=headers(),
        json={"input": {"file": b64, "fileType": 0, "dpi": dpi}},
        timeout=30,
    )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"Job submitted: {job_id}")
    return job_id


def poll(base_url: str, job_id: str, timeout: int = 300) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(
            f"{base_url}/status/{job_id}",
            headers=headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data
        if status in ("FAILED", "CANCELLED"):
            sys.exit(f"Job {status}: {data.get('error', '')}")
        print(f"  [{status}] waiting…")
        time.sleep(POLL_INTERVAL)
    sys.exit(f"Timeout after {timeout}s")


def print_results(data: dict):
    delay_ms  = data.get("delayTime", 0)   # cold start + queue time (ms)
    exec_ms   = data.get("executionTime", 0)  # handler runtime (ms)

    print()
    print(f"  Cold start + queue : {delay_ms / 1000:.2f}s")
    print(f"  Execution time     : {exec_ms / 1000:.2f}s")
    print(f"  Total              : {(delay_ms + exec_ms) / 1000:.2f}s")

    output = data.get("output", {})
    if "error" in output:
        print(f"\nHandler error: {output['error']}")
        return

    pages = output.get("pages", [])
    print(f"\n  Pages processed: {len(pages)}")
    for page in pages:
        regions = page["regions"]
        print(f"\n  Page {page['page_num']}  ({page['width_px']}×{page['height_px']}px)"
              f"  — {len(regions)} regions")
        for r in regions[:8]:  # show first 8 regions
            x0, y0, x1, y1 = r["bbox"]
            print(f"    [{r['order']:3d}] {r['type']:<20s}  score={r['score']:.3f}"
                  f"  bbox=({x0},{y0})-({x1},{y1})")
        if len(regions) > 8:
            print(f"    … and {len(regions) - 8} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint_id", help="RunPod endpoint ID (e.g. olwc542lxnubyo)")
    ap.add_argument("pdf", help="Path to PDF file")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--pages", type=int, default=None,
                    help="Only send first N pages (requires pypdf: pip install pypdf)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", metavar="OUT.json", help="Save full output to JSON file")
    args = ap.parse_args()

    base_url = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    print(f"Endpoint: {base_url}")

    pdf_bytes = open(args.pdf, "rb").read()

    if args.pages:
        try:
            from pypdf import PdfReader, PdfWriter
            import io
            reader = PdfReader(io.BytesIO(pdf_bytes))
            writer = PdfWriter()
            for i in range(min(args.pages, len(reader.pages))):
                writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()
            print(f"Trimmed to {args.pages} page(s)")
        except ImportError:
            print("pypdf not installed — sending full PDF")

    print(f"Sending {len(pdf_bytes):,} bytes at {args.dpi} DPI …")
    t0 = time.time()
    job_id = submit(base_url, pdf_bytes, args.dpi)
    data = poll(base_url, job_id, timeout=args.timeout)
    print(f"\nWall clock: {time.time() - t0:.2f}s")
    print_results(data)

    out_path = args.json or (
        f"{Path(args.pdf).stem}_{args.endpoint_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
