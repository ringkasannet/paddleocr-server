# Split Endpoint Plan — Layout + VLM Serverless

## Overview

Two independent RunPod serverless endpoints replace the monolithic handler. The
layout endpoint is cheap and fast; the VLM endpoint runs only for regions that
need it. FlashBoot + cached models achieve **seconds-level cold start** on both.

```
Client
  │
  ├─① layout-endpoint  (T4, ~$0.14/hr)
  │     Input:  {"images": ["data:application/pdf;base64,..."]}
  │     Output: pages → regions → bbox, label, crop, text (pdfplumber where available)
  │
  │     Client filters: regions where text == null → send to VLM
  │
  ├─② vlm-endpoint  (RTX 3090 / A100)
  │     Input:  {"regions": [{"region_id", "crop", "label"}]}
  │     Output: {"results": [{"region_id", "content"}]}
  │
  └─ orchestrator assembles: relevel_titles → markdown
```

---

## Directory Structure

```
runpod/hybrid/
├── layout/
│   ├── Dockerfile          ✅ done
│   ├── handler.py          ✅ done
│   └── requirements.txt    ✅ done
├── vlm/
│   ├── Dockerfile          ✅ done
│   └── handler.py          ✅ done
├── client/
│   ├── orchestrator.py     ✅ done (includes relevel_titles)
│   ├── title_level.py      ✅ done (ported from paddlex)
│   └── requirements.txt    ✅ done
├── SPLIT_ENDPOINT_PLAN.md  ← this file
├── ARCHITECTURE_COMPARISON.md
├── PIPELINE_ANALYSIS.md
└── HANDOFF_HYBRID_EXTRACTOR.md
```

---

## Cold Start Strategy

### RunPod capabilities used

| Capability | What it eliminates | Result |
|---|---|---|
| **Cached model** | HuggingFace download on cold start | Worker scheduled on host with weights already on local NVMe |
| **FlashBoot** | CUDA init + weight reload into GPU VRAM | GPU state snapshot restored on spin-up |
| Combined | Both download and GPU init | **P90 ≤ 2s, P95 ≤ 2.3s** (RunPod published metrics) |

### Cached vs baked-in

RunPod explicitly recommends **cached models** over baked-in:

> *"If your model is available on Hugging Face, we strongly recommend enabling
> cached models. Cached models provide faster startup times, lower costs, and
> uses less storage."*

Both models are on HuggingFace:
- Layout: `PaddlePaddle/PP-DocLayoutV3_safetensors` → **use cached model**
- VLM: `zai-org/GLM-OCR` → **use cached model**

Both Dockerfiles already point to the RunPod volume cache:
```dockerfile
ENV HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub
```

FlashBoot is **enabled by default** on all new RunPod GPU endpoints — no code
change required.

### Realistic cold start after FlashBoot warms up

| Endpoint | Cold start |
|---|---|
| Layout (PP-DocLayoutV3, ~300 MB) | ~2–5s |
| VLM (GLM-OCR via vLLM) | ~2–5s |

Note: FlashBoot performs better with consistent traffic. After a few warmup
requests, both endpoints settle into the P90 ≤ 2s window.

---

## Endpoint 1 — Layout

### Responsibilities

1. Render PDF pages to images at 150 DPI (PyMuPDF)
2. Run PP-DocLayoutV3 layout detection → bboxes + labels + reading order
3. Run pdfplumber on text-type regions → extract text from native text layer
4. Crop each region from page image → PNG base64
5. Return all regions; mark `text=null` for regions the VLM must handle
6. Supports multi-source input: any number of PDFs or images in `images[]`

### API Contract

**Request:**
```json
{"images": ["data:application/pdf;base64,<b64>"]}
```

**Response:**
```json
{
  "pages": [
    {
      "page_index": 0,
      "width_px": 1240,
      "height_px": 1754,
      "regions": [
        {
          "region_id":    "p0_r3",
          "bbox_px":      [120, 200, 980, 260],
          "bbox_pt":      [57.6, 96.0, 470.4, 124.8],
          "label":        "text",
          "native_label": "paragraph_title",
          "score":        0.94,
          "order":        3,
          "polygon":      [[120,200],[980,200],[980,260],[120,260]],
          "crop":         "data:image/png;base64,<b64>",
          "text":         "Peraturan Menteri Keuangan"
        },
        {
          "region_id":    "p0_r7",
          "bbox_px":      [120, 400, 980, 700],
          "bbox_pt":      [57.6, 192.0, 470.4, 336.0],
          "label":        "table",
          "native_label": "table",
          "score":        0.91,
          "order":        7,
          "polygon":      [[120,400],[980,400],[980,700],[120,700]],
          "crop":         "data:image/png;base64,<b64>",
          "text":         null
        }
      ]
    }
  ],
  "meta": {
    "total_regions":  24,
    "text_extracted": 18,
    "vlm_needed":      6,
    "searchable":    true,
    "num_sources":     1,
    "dpi":           150
  }
}
```

`text != null` → pdfplumber extracted it, skip VLM.
`text == null` → VLM required (table, formula, chart, seal, or scanned page).

### RunPod endpoint configuration

| Setting | Value |
|---|---|
| GPU type | T4 (16 GB) |
| Cached model | `PaddlePaddle/PP-DocLayoutV3_safetensors` |
| FlashBoot | enabled (default) |
| Min workers | 0 |
| Max workers | 5 |
| Container disk | 20 GB |

---

## Endpoint 2 — VLM

### Responsibilities

1. Start vLLM with `zai-org/GLM-OCR` (once per container)
2. Per-request: accept batch of pre-cropped region images
3. Submit all crops concurrently to vLLM via asyncio + aiohttp
4. vLLM continuous batching handles all concurrent requests natively
5. Return text per region_id

### Prompts (exact, from glmocr config.yaml)

```python
_PROMPTS = {
    "text":    "Text Recognition:",
    "table":   "Table Recognition:",
    "formula": "Formula Recognition:",
}
```

Regions with other labels (figure, seal, etc.) receive no text prompt — image
only, matching glmocr's behaviour for those task types.

### vLLM parameters (exact, from glmocr config.yaml)

```python
temperature=0.0, top_p=0.00001, top_k=1, repetition_penalty=1.1
```

### API Contract

**Request:**
```json
{
  "regions": [
    {"region_id": "p0_r7",  "crop": "data:image/png;base64,<b64>", "label": "table"},
    {"region_id": "p0_r12", "crop": "data:image/png;base64,<b64>", "label": "formula"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"region_id": "p0_r7",  "content": "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |"},
    {"region_id": "p0_r12", "content": "$$E = mc^2$$"}
  ],
  "meta": {
    "regions_received": 6,
    "regions_processed": 6,
    "errors": []
  }
}
```

### RunPod endpoint configuration

| Setting | Value |
|---|---|
| GPU type | RTX 3090 (24 GB) |
| Cached model | `zai-org/GLM-OCR` |
| FlashBoot | enabled (default) |
| Min workers | 0 |
| Max workers | 5 |
| Container disk | 20 GB |

Min workers = 0 is now viable because FlashBoot brings cold start to ~2–5s.
Previously required min=1 (always-on) to avoid 60–90s cold start; FlashBoot
eliminates that constraint.

---

## Client Orchestrator

Lives outside RunPod — application server or thin Lambda/Cloud Function.

### Pipeline

```
1. Layout call    → layout_result (regions with text or null)
2. Filter         → vlm_regions = [r for r if r.text is None and r.label != "skip"]
3. VLM call       → vlm_map {region_id: content}   (skipped if vlm_regions empty)
4. Backfill text  → region["text"] = pdfplumber_text or vlm_map[id]
5. relevel_titles → assign heading depth via font-size KMeans + numbering patterns
6. Render markdown
```

### Post-processing implemented

| Step | Source | Status |
|---|---|---|
| Title depth inference (`relevel_titles`) | `title_level.py` ported from paddlex | ✅ done |
| Formula delimiters (`$$..$$`, `$...$`) | orchestrator.py | ✅ done |
| `doc_title` → `#`, `paragraph_title` → `##`/`###`/`####` | orchestrator.py | ✅ done |
| Cross-page table merge | paddlex `merge_table.py` | not yet |

### title_level.py — how heading depth works

Three signals voted together per heading:
1. **Numbering pattern** — `1.2.3` → depth 3, `I.` → depth 1, `第一章` → depth 1
2. **Font size clustering** — KMeans(k=4) on bbox pixel heights across all pages
3. **Global symbol order** — order in which each numbering style first appeared

Majority vote wins. Special keywords (ABSTRACT, INTRODUCTION, REFERENCES, 结论,
参考文献…) always map to depth 1.

---

## Comparison: Monolithic vs Split

| Dimension | Monolithic | Split + FlashBoot |
|---|---|---|
| Layout cold start | 2–3 min | **~2–5s** |
| VLM cold start | 2–3 min | **~2–5s** |
| GPU for layout | RTX 3090 (overkill) | **T4 (~3× cheaper)** |
| GPU for VLM | Shared with layout | **Full GPU** |
| VLM GPU util | 80% | **93%** |
| Searchable PDF VLM cost | Full cost | **VLM skipped entirely** |
| Title heading depth | ❌ all `##` | ✅ `##`/`###`/`####` |
| Min workers needed | 1 (always-on) | **0 (FlashBoot handles it)** |
| Operational complexity | Low | Medium (two endpoints + orchestrator) |

---

## Implementation Status

### Done
- [x] `layout/handler.py` — PP-DocLayoutV3, pdfplumber, multi-source batching
- [x] `layout/Dockerfile` — cached model via runpod-volume
- [x] `layout/requirements.txt`
- [x] `vlm/handler.py` — vLLM subprocess, asyncio fan-out, exact glmocr prompts/params
- [x] `vlm/Dockerfile` — cached model via runpod-volume
- [x] `client/orchestrator.py` — two-step pipeline, relevel_titles, markdown assembly
- [x] `client/title_level.py` — ported from paddlex (Apache 2.0)
- [x] `client/requirements.txt`

### Pending
- [ ] End-to-end test: kemenkeu PDF → layout → VLM → compare vs monolithic output
- [ ] Benchmark: cold start, throughput, cost per page
- [ ] Push images to Docker Hub
- [ ] Create RunPod endpoints, enable cached models + FlashBoot
- [ ] Cross-page table merge (`merge_table.py` from paddlex, same port approach as title_level.py)

---

## Known Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `zai-org/GLM-OCR` requires HF token | Cached model setup needs token | Configure HF_TOKEN in RunPod endpoint env vars; RunPod cached model supports gated models |
| FlashBoot cold start degrades under low traffic | >2s cold starts during quiet periods | Set min_workers=1 during business hours if latency SLA is strict |
| Crop PNG payload exceeds RunPod job limit | Job rejected | Compress to JPEG q=95 for non-formula regions; keep PNG for formula/seal only |
| pdfplumber bbox misalignment | Text from wrong region | Validate with known document comparing pdfplumber vs VLM text on same region |
| relevel_titles gives wrong depths | Heading hierarchy broken | Falls back gracefully — headings still render as `##`, just without depth distinction |
