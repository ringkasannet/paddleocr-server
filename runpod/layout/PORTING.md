# PP-DocLayoutV3 RunPod Handler — Porting Notes

Handler file: `handler.py`  
Model: `PaddlePaddle/PP-DocLayoutV3_safetensors` (HuggingFace)  
PDF text layer: `pypdfium2`

---

## What was ported from PaddleX

The PP-DocLayoutV3 model ships as a PaddlePaddle/PaddleX pipeline. We run it through
HuggingFace Transformers (`AutoModelForObjectDetection`) to avoid the PaddlePaddle runtime
dependency. That means the PaddleX pipeline's post-processing — reading order, NMS, text
extraction — had to be re-implemented from scratch in pure Python/NumPy.

### 1. Containment-based cross-class NMS — `_nms_regions()`

**Source:** PaddleX internal NMS logic, containment threshold 0.65.

**Why needed:** HuggingFace's `post_process_object_detection` runs per-class NMS only.
The same bounding box can survive detection as two different label types (e.g. a paragraph
detected as both `text` and `content`). PaddleX applies a second cross-class NMS pass that
removes the lower-score duplicate if the boxes overlap by more than 65% of the smaller area.

**Implementation:**
```python
def _nms_regions(regions: list) -> list:
    by_score = sorted(regions, key=lambda r: r["score"], reverse=True)
    kept = []
    for cand in by_score:
        cx0, cy0, cx1, cy1 = cand["bbox"]
        ca = max(0, cx1-cx0) * max(0, cy1-cy0)
        dominated = False
        for k in kept:
            kx0, ky0, kx1, ky1 = k["bbox"]
            ka = max(0, kx1-kx0) * max(0, ky1-ky0)
            ix0,iy0 = max(cx0,kx0), max(cy0,ky0)
            ix1,iy1 = min(cx1,kx1), min(cy1,ky1)
            if ix1>ix0 and iy1>iy0:
                if (ix1-ix0)*(iy1-iy0)/min(ca,ka) > 0.65:
                    dominated = True; break
        if not dominated:
            kept.append(cand)
    return kept
```

### 2. Projection-histogram XY-cut — `_projection_by_bboxes`, `_split_projection_profile`, `_recursive_xy_cut`

**Source:** `paddlex/repo_apis/PaddleDetection_api/xycut_enhanced/utils.py:recursive_xy_cut`

**Why needed:** Multi-column documents require column-aware reading order. A naive y-sort
places right-column content interleaved with left-column content whenever both columns have
content at the same vertical band.

The XY-cut algorithm builds a 1-D projection histogram of bbox extents along each axis, finds
gaps (zero-count positions), and recursively splits the page into column and row segments
using those natural gaps rather than fixed pixel thresholds.

**Implementation:** Pure NumPy, no GPU required. The three helper functions mirror the
PaddleX originals exactly except for Python 3 type annotations.

### 3. Column-aware layout sort — `_sorted_layout_boxes`

**Source:** `paddlex/inference/pipelines/layout_parsing/utils.py:sorted_layout_boxes`

**Why needed:** Fallback for the rare case where `_recursive_xy_cut` returns a mismatched
index count (degenerate bbox layouts). Also provides the Phase 1 threshold logic used to
classify regions as left-column, right-column, or full-width.

PaddleX thresholds (fraction of page width):
- Left column: `x0 < w/4` and `x1 < w*3/5`
- Right column: `x0 > w*2/5`
- Full-width (divider): neither

---

## Two-phase reading order design

The final `_reading_order()` function combines both PaddleX approaches:

**Phase 1** — Use PaddleX x-thresholds to classify every region and pull out the full-width
ones. Full-width regions (title banners, abstract headers, horizontal rules, section breaks)
act as natural dividers that flush the accumulated column buffer.

**Phase 2** — Within each column segment (the content between two full-width dividers),
run `_recursive_xy_cut` to find the natural column gap via projection. This avoids the
threshold gap-filling problem that arises when a full-width element sits in the middle of a
two-column band and causes PaddleX's `sorted_layout_boxes` to misorder everything below it.

```
y-sorted regions
      │
      ▼
  _is_column?
  ├── yes → accumulate into seg[]
  └── no  → flush seg via _xycut_segment(), emit divider, continue
      │
      ▼
  _xycut_segment → _recursive_xy_cut (or fallback _sorted_layout_boxes)
```

---

## Problems found and solutions

### Problem 1 — Soft hyphen artifacts in extracted text

**Symptom:** Compound words appeared joined without a hyphen (`humanAI`, `humancentred`,
`modelbuilding`) or with raw control characters.

**Root cause:** pypdfium2 encodes discretionary (line-break) hyphens placed by the PDF
typesetter as `\x02` (ASCII STX). Unicode soft hyphens (`U+00AD`, `\xad`) also appear.
When the line-break hyphen falls at a semantic compound-word boundary (e.g. `human‑AI`),
deleting it silently merges the words.

**Solution:** Two-step cleanup in `_extract_text_at_bbox`:
```python
# Restore hyphen at lowercase→UPPERCASE boundary (compound words like human-AI)
text = re.sub(r"([a-z])\x02([A-Z])", r"\1-\2", text)
# Strip remaining discretionary and soft hyphens (syllable breaks like de-vel-op-ment)
text = text.replace("\x02", "").replace("\xad", "")
```

**Limitation:** `[a-z]\x02[a-z]` boundaries (e.g. `human‑centred`) cannot be
resolved safely without a compound-word dictionary — the same pattern covers both
compound breaks and syllable breaks.

---

### Problem 2 — Adjacent-column glyph bleed into extracted text

**Symptom:** Reference entries had spurious leading characters (`p\n[38]…`, `pp\n[39]…`,
`y\n[41]…`) from the adjacent PDF column.

**Root cause:** The model outputs integer pixel bounding boxes. Converting back to PDF points
(`x_px / scale`) truncates to floats that can land exactly on the edge of a neighbouring
column's glyph. pypdfium2's `get_text_bounded` includes any character whose bounding box
intersects the query rectangle, so glyphs just outside the intended region creep in.

**Solution:** 2-point inward margin applied before calling `get_text_bounded`, inside
`_extract_text_at_bbox`:
```python
m = 2  # PDF points ≈ 1 px at 150 dpi
text = textpage.get_text_bounded(
    left   = x0 + m,
    bottom = h - y1 + m,
    right  = x1 - m,
    top    = h - y0 - m,
)
```

This is document-type-agnostic. The model's bbox precision is far larger than 2 pt so no
real content is clipped.

---

### Problem 3 — Document-level searchability check too coarse

**Symptom:** A scanned report with a native-text cover page returned `searchable=True`,
causing the handler to call `_extract_text_at_bbox` on every scanned page, returning empty
or garbage strings.

**Root cause:** The original `_is_searchable` sampled the first 3 pages and used a 50-char
average threshold — a single text-bearing cover page was enough to pass.

**Solution:** Replaced with per-page `_searchable_pages` returning `set[int]`:
```python
def _searchable_pages(pdf, min_chars=10) -> set[int]:
    result = set()
    for i in range(len(pdf)):
        page = pdf[i]
        textpage = page.get_textpage()
        if len(textpage.get_text_range()) > min_chars:
            result.add(i)
        textpage.close(); page.close()
    return result
```

- Threshold lowered to 10 chars — handles sparse forms/invoices with short field labels.
- Per-page: text is only extracted on pages confirmed to have a text layer.
- The output `"searchable"` field remains a bool (`True` = at least one page searchable).

---

### Problem 4 — Missing region types in `TEXT_REGION_TYPES`

Several region types detected by PP-DocLayoutV3 were not in the extraction set, causing
their `text` field to remain `""` silently.

| Type | Missing until | Affected content |
|------|--------------|-----------------|
| `figure_title` | fix | Figure captions |
| `table_title` | fix | Table captions |
| `chart_title` | fix | Chart captions |
| `number` | fix | Page numbers |
| `vision_footnote` | fix | Author photo bios, image annotations |

**Solution:** Added all five to `TEXT_REGION_TYPES`. The complete set:
```python
TEXT_REGION_TYPES = {
    "text", "title", "paragraph_title", "abstract",
    "content", "reference", "footnote", "doc_title",
    "header", "footer", "reference_content", "aside_text",
    "figure_title", "table_title", "chart_title",
    "number", "vision_footnote",
}
```

Excluded (no meaningful text layer — sent to VLM downstream): `table`, `figure`,
`figure_formula`, `seal`, `chart`, `formula`.

---

## Known limitations

| Issue | Root cause | Status |
|-------|-----------|--------|
| `humancentred`, `modelbuilding` missing hyphen | `[a-z]\x02[a-z]` indistinguishable from syllable break | No fix without word list |
| Reading order thresholds assume ≤2 columns | `_is_column` uses fixed `w/4`, `w*3/5` fractions | Would need layout-adaptive thresholds for 3-column |
| Last reference block merged as single region | Model detection decision | Not fixable in post-processing |
| Garbled text where bboxes overlap PDF structure | pypdfium2 extracts all chars in rect | 2-pt margin helps but can't fully prevent |
| PDF typos pass through unmodified | Errors in original PDF text layer | Expected behaviour |
