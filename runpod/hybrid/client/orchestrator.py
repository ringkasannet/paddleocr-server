"""Reference orchestrator for the split layout + VLM endpoint pipeline.

Usage:
    from orchestrator import process_pdf
    result = process_pdf(pdf_bytes, layout_endpoint_id, vlm_endpoint_id)

Environment variables:
    RUNPOD_API_KEY      - Required for RunPod API calls
    LAYOUT_ENDPOINT_ID  - RunPod endpoint ID for layout detection
    VLM_ENDPOINT_ID     - RunPod endpoint ID for VLM inference
"""

from __future__ import annotations

import base64
import os
import time
from typing import Optional

import requests

from title_level import assign_levels_to_parsing_res

# ── RunPod client ──────────────────────────────────────────────────────────────

RUNPOD_API_KEY     = os.environ.get("RUNPOD_API_KEY", "")
LAYOUT_ENDPOINT_ID = os.environ.get("LAYOUT_ENDPOINT_ID", "")
VLM_ENDPOINT_ID    = os.environ.get("VLM_ENDPOINT_ID", "")


def _runpod_run_sync(endpoint_id: str, payload: dict, timeout: int = 600) -> dict:
    """Submit a RunPod job and poll until complete. Returns output dict."""
    base = f"https://api.runpod.io/v2/{endpoint_id}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(f"{base}/runsync", json={"input": payload}, headers=headers, timeout=30)
    r.raise_for_status()
    resp = r.json()

    # runsync can return immediately with output, or return a job ID to poll
    if resp.get("status") == "COMPLETED":
        return resp["output"]

    job_id = resp["id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        r = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=15)
        r.raise_for_status()
        resp = r.json()
        status = resp.get("status")
        if status == "COMPLETED":
            return resp["output"]
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"RunPod job {job_id} {status}: {resp.get('error', '')}")

    raise TimeoutError(f"RunPod job {job_id} did not complete within {timeout}s")


# ── Markdown rendering ─────────────────────────────────────────────────────────

_DISPLAY_FORMULA_LABELS = {"display_formula"}
_INLINE_FORMULA_LABELS  = {"inline_formula"}

# title_level integer → markdown heading prefix (level 0 = doc_title = #)
_LEVEL_TO_PREFIX = {0: "# ", 1: "# ", 2: "## ", 3: "### ", 4: "#### "}


class _Block:
    """Thin attribute wrapper so title_level.py can work with our region dicts."""
    __slots__ = ("label", "bbox", "content", "title_level", "_region")

    def __init__(self, region: dict) -> None:
        self.label   = region["native_label"]
        self.bbox    = region["bbox_px"]          # [x1, y1, x2, y2] pixels
        self.content = region.get("text") or ""
        self.title_level: int | None = None
        self._region = region


def _relevel_titles(pages: list[dict]) -> None:
    """Run assign_levels_to_parsing_res across all pages and write title_level
    back into each region dict."""
    blocks_by_page: list[list[_Block]] = []
    for page in pages:
        page_blocks = [
            _Block(r)
            for r in page["regions"]
            if r["native_label"] == "paragraph_title"
        ]
        blocks_by_page.append(page_blocks)

    assign_levels_to_parsing_res(blocks_by_page)

    for page_blocks in blocks_by_page:
        for block in page_blocks:
            block._region["title_level"] = block.title_level


def _to_markdown_block(region: dict) -> Optional[str]:
    """Convert a single region to a markdown string. Returns None to skip."""
    label  = region.get("label", "text")   # task_type
    native = region.get("native_label", "")
    content = region.get("text") or ""

    if label == "skip" or not content:
        return None

    if native in _DISPLAY_FORMULA_LABELS:
        return f"$$\n{content}\n$$"
    if native in _INLINE_FORMULA_LABELS:
        return f"${content}$"

    if native == "doc_title":
        return f"# {content}"

    if native == "paragraph_title":
        level = region.get("title_level") or 2
        prefix = _LEVEL_TO_PREFIX.get(level, "## ")
        return f"{prefix}{content}"

    # table — content already in markdown table format from VLM
    return content


def _assemble_markdown(layout_result: dict, vlm_map: dict[str, str]) -> str:
    """Build markdown from layout regions, merging pdfplumber + VLM text."""
    pages = layout_result.get("pages", [])

    # First pass: backfill all text so _relevel_titles sees complete content.
    for page in pages:
        for region in page["regions"]:
            region_id = region["region_id"]
            content = region.get("text") or vlm_map.get(region_id)
            region["text"] = content

    # Assign heading depths using font-size clustering + numbering patterns.
    _relevel_titles(pages)

    # Second pass: render markdown in reading order.
    blocks: list[str] = []
    for page in pages:
        for region in sorted(page["regions"], key=lambda r: r["order"]):
            block = _to_markdown_block(region)
            if block:
                blocks.append(block)

    return "\n\n".join(blocks)


# ── Main orchestration ─────────────────────────────────────────────────────────

def process_pdf(
    pdf_bytes: bytes,
    layout_endpoint_id: str = LAYOUT_ENDPOINT_ID,
    vlm_endpoint_id: str = VLM_ENDPOINT_ID,
) -> dict:
    """End-to-end OCR pipeline: layout detection + selective VLM inference.

    Args:
        pdf_bytes: Raw PDF file content.
        layout_endpoint_id: RunPod endpoint ID for layout detection.
        vlm_endpoint_id: RunPod endpoint ID for VLM inference.

    Returns:
        {
            "pages": [...],         # regions with text filled in
            "markdown": "...",      # assembled markdown document
            "meta": {
                "layout_meta": {...},
                "vlm_called": bool,
                "vlm_regions": int,
                "elapsed_s": float,
            }
        }
    """
    t0 = time.time()
    pdf_b64 = base64.b64encode(pdf_bytes).decode()

    # Step 1 — Layout detection + pdfplumber
    print(f"[orchestrator] Calling layout endpoint {layout_endpoint_id} ...")
    layout_result = _runpod_run_sync(
        layout_endpoint_id,
        {"images": [f"data:application/pdf;base64,{pdf_b64}"]},
    )
    if "error" in layout_result:
        raise RuntimeError(f"Layout endpoint error: {layout_result['error']}")
    t_layout = time.time()
    print(f"[orchestrator] Layout done in {t_layout - t0:.1f}s  "
          f"({layout_result['meta']['total_regions']} regions, "
          f"{layout_result['meta']['text_extracted']} extracted, "
          f"{layout_result['meta']['vlm_needed']} VLM needed)")

    # Step 2 — Collect regions needing VLM (text == null, not skip)
    vlm_regions = [
        {
            "region_id": r["region_id"],
            "crop":      r["crop"],
            "label":     r["native_label"],
        }
        for page in layout_result["pages"]
        for r in page["regions"]
        if r["text"] is None and r["label"] != "skip"
    ]

    # Step 3 — VLM inference (skipped for fully searchable PDFs)
    vlm_map: dict[str, str] = {}
    if vlm_regions:
        print(f"[orchestrator] Calling VLM endpoint {vlm_endpoint_id} "
              f"with {len(vlm_regions)} crops ...")
        vlm_response = _runpod_run_sync(
            vlm_endpoint_id,
            {"regions": vlm_regions},
        )
        if "error" in vlm_response:
            raise RuntimeError(f"VLM endpoint error: {vlm_response['error']}")
        vlm_map = {
            r["region_id"]: r["content"]
            for r in vlm_response.get("results", [])
            if r.get("content") is not None
        }
        print(f"[orchestrator] VLM done in {time.time() - t_layout:.1f}s")

    # Step 4 — Assemble markdown and backfill region text
    markdown = _assemble_markdown(layout_result, vlm_map)

    elapsed = time.time() - t0
    print(f"[orchestrator] Total elapsed: {elapsed:.1f}s")

    return {
        "pages":    layout_result["pages"],
        "markdown": markdown,
        "meta": {
            "layout_meta": layout_result["meta"],
            "vlm_called":  bool(vlm_regions),
            "vlm_regions": len(vlm_regions),
            "elapsed_s":   round(elapsed, 2),
        },
    }


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <path/to/document.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = process_pdf(pdf_bytes)
    print("\n--- MARKDOWN OUTPUT ---")
    print(result["markdown"][:3000])
    if len(result["markdown"]) > 3000:
        print(f"... [{len(result['markdown']) - 3000} chars truncated]")
    print("\n--- META ---")
    import json
    print(json.dumps(result["meta"], indent=2))
