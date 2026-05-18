#!/usr/bin/env python3
"""Local test runner for handler.py — no RunPod account needed.

One-time setup
--------------
1. Install CUDA PyTorch (replace cu121 with cu118 if your CUDA is 11.8):
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

2. Install remaining deps:
       pip install pypdfium2 pypdf

3. First run downloads ~400 MB model from HuggingFace automatically.
   Set HF_HOME to redirect the cache, e.g.:
       set HF_HOME=D:\\models\\huggingface

Usage
-----
    python test_local.py collab.pdf
    python test_local.py collab.pdf --dpi 200
    python test_local.py collab.pdf --pages 4
    python test_local.py collab.pdf --out result.json
"""

import argparse
import base64
import io
import json
import sys
import time
import types
from pathlib import Path

# ── Mock runpod before handler.py is imported ─────────────────────────────────
# handler.py calls runpod.serverless.start() at module level; that blocks
# indefinitely waiting for jobs. The mock turns it into a no-op so the
# module loads cleanly for local use.
_runpod = types.ModuleType("runpod")
_runpod.serverless = types.SimpleNamespace(start=lambda cfg: None)
sys.modules["runpod"] = _runpod

# Add handler directory to path and import — this triggers the model load.
sys.path.insert(0, str(Path(__file__).parent))
print("[init] Loading model (downloads from HuggingFace on first run ~400 MB)…")
t_load = time.time()
import handler
print(f"[init] Model ready in {time.time() - t_load:.1f}s")


def _run(pdf_bytes: bytes, dpi: int = 150) -> dict:
    b64 = base64.b64encode(pdf_bytes).decode()
    return handler.handler({"input": {"file": b64, "fileType": 0, "dpi": dpi}})


def _print_results(result: dict) -> None:
    pages = result.get("pages", [])
    searchable = result.get("searchable", False)
    print(f"\n  Pages      : {len(pages)}")
    print(f"  Searchable : {searchable}")
    for page in pages:
        regions = page["regions"]
        print(
            f"\n  Page {page['page_num']}  "
            f"({page['width_px']}×{page['height_px']} px)  "
            f"— {len(regions)} regions:"
        )
        for r in regions[:15]:
            txt = repr(r["text"][:70]) if r["text"] else '""'
            print(
                f"    [{r['order']:3d}] {r['type']:<22s} "
                f"score={r['score']:.3f}  {txt}"
            )
        if len(regions) > 15:
            print(f"    … and {len(regions) - 15} more")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local handler test")
    ap.add_argument("pdf", help="PDF file to process")
    ap.add_argument("--dpi", type=int, default=150, help="Render DPI (default 150)")
    ap.add_argument("--pages", type=int, default=None, help="Only first N pages")
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    pdf_bytes = Path(args.pdf).read_bytes()

    if args.pages:
        try:
            from pypdf import PdfReader, PdfWriter
            reader = PdfReader(io.BytesIO(pdf_bytes))
            writer = PdfWriter()
            for i in range(min(args.pages, len(reader.pages))):
                writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()
            print(f"Trimmed to {args.pages} page(s)  ({len(pdf_bytes):,} bytes)")
        except ImportError:
            print("pypdf not installed, sending full PDF  (pip install pypdf to slice)")

    print(f"\nProcessing: {Path(args.pdf).name}  ({len(pdf_bytes):,} bytes)  dpi={args.dpi}")
    t0 = time.time()
    result = _run(pdf_bytes, args.dpi)
    elapsed = time.time() - t0

    if "error" in result:
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    print(f"Done in {elapsed:.2f}s")
    _print_results(result)

    out_path = Path(args.out or f"{Path(args.pdf).stem}_local.json")
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
