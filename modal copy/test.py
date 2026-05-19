"""Test script for the deployed Modal layout extractor.

Usage:
    python modal/test.py                          # sends 2 chunks back-to-back (warm test)
    python modal/test.py --cold                   # waits 6 min between chunks (cold-start test)
    python modal/test.py --chunk-size 4           # pages per request (default 4)
    python modal/test.py --chunk 0 --chunk 2      # specific chunk indices only
    python modal/test.py --all                    # send all chunks
"""

import argparse
import base64
import io
import json
import sys
import time

import requests

ENDPOINT   = "https://ringkasan-net--paddleocr-hybrid-process-pdf.modal.run"
PDF_PATH   = "modal/pmk.pdf"
CHUNK_SIZE = 4

SCALEDOWN_S = 300
COLD_WAIT_S = SCALEDOWN_S + 30


def split_pdf(pdf_path: str, chunk_size: int) -> list[bytes]:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

    reader = PdfReader(pdf_path)
    total  = len(reader.pages)
    chunks = []
    for start in range(0, total, chunk_size):
        writer = PdfWriter()
        for i in range(start, min(start + chunk_size, total)):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
        print(f"  Chunk {len(chunks) - 1}: pages {start + 1}–{min(start + chunk_size, total)} of {total}")
    return chunks


def send_chunk(chunk_bytes: bytes, label: str, show_text: bool = False) -> dict:
    pdf_b64 = base64.b64encode(chunk_bytes).decode()
    print(f"\n-> Sending {label} ({len(chunk_bytes) / 1024:.0f} KB) ...")
    t0 = time.time()

    resp = requests.post(ENDPOINT, json={"pdf_b64": pdf_b64}, timeout=600)
    wall = time.time() - t0

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text[:300]}")
        return {}

    data = resp.json()
    m    = data.get("meta", {})
    t    = m.get("timing", {})

    print(f"  Wall clock     : {wall:.2f}s  (Modal reports {m.get('elapsed_s', '?')}s)")
    print(f"  Pages          : {len(data.get('pages', []))}")
    print(f"  Total regions  : {m.get('total_regions', '?')}")
    print(f"  Text extracted : {m.get('text_extracted', '?')}")
    if t:
        print(f"  ── inside detect() ──")
        print(f"    load (model) : {t.get('load_s', 'snapshot')}s")
        print(f"    render       : {t.get('render_s', '?')}s")
        print(f"    detect       : {t.get('detect_s', '?')}s")
        print(f"    build dicts  : {t.get('build_s', '?')}s")
        print(f"    pdfplumber   : {t.get('plumber_s', '?')}s")
        print(f"    total        : {t.get('total_s', '?')}s")

    if show_text:
        print(f"\n  ── region text ──")
        for page in data.get("pages", []):
            print(f"  [page {page['page_index']}]")
            for r in page["regions"]:
                text = r.get("text") or ""
                preview = text[:80].replace("\n", " / ") if text else "(no text)"
                print(f"    {r['region_id']:10s}  {r['native_label']:20s}  {preview}")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold",       action="store_true", help=f"Wait {COLD_WAIT_S}s between requests")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk",      type=int, action="append", dest="chunks")
    parser.add_argument("--all",        action="store_true", help="Send all chunks")
    parser.add_argument("--show-text",  action="store_true", help="Print text content of each region")
    parser.add_argument("--save",       action="store_true", help="Save each response as modal/output_chunk_N.json")
    parser.add_argument("--pdf",        default=PDF_PATH)
    args = parser.parse_args()

    print(f"PDF   : {args.pdf}")
    print(f"Chunk : {args.chunk_size} pages each")
    print(f"Mode  : {'cold-start' if args.cold else 'warm (back-to-back)'}")

    print(f"\nSplitting {args.pdf} ...")
    all_chunks = split_pdf(args.pdf, args.chunk_size)

    if args.chunks:
        indices = args.chunks
    elif args.all:
        indices = list(range(len(all_chunks)))
    else:
        indices = list(range(min(2, len(all_chunks))))

    results = []
    for n, idx in enumerate(indices):
        if idx >= len(all_chunks):
            print(f"Chunk index {idx} out of range")
            continue
        label = f"chunk {idx}"
        result = send_chunk(all_chunks[idx], label, show_text=args.show_text)
        if args.save and result:
            out_path = f"modal/output_chunk_{idx}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  Saved -> {out_path}")
        results.append((label, result))

        if args.cold and n < len(indices) - 1:
            print(f"\n⏳ Waiting {COLD_WAIT_S}s for scale-down ...")
            for remaining in range(COLD_WAIT_S, 0, -10):
                print(f"   {remaining}s remaining ...", end="\r")
                time.sleep(10)
            print()

    print("\n── Summary ──────────────────────────────────────────")
    for label, data in results:
        m = data.get("meta", {})
        t = m.get("timing", {})
        print(f"  {label}: wall={m.get('elapsed_s','?')}s  "
              f"render={t.get('render_s','?')}s  "
              f"detect={t.get('detect_s','?')}s  "
              f"plumber={t.get('plumber_s','?')}s  "
              f"text={m.get('text_extracted','?')}")


if __name__ == "__main__":
    main()
