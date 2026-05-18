# Handoff: Hybrid PDF Extractor (pdfplumber + layout detection)

## What this is

A proposed optimization to the current document processing pipeline. Many PDFs are digitally created (Word → PDF, typeset) and already have an embedded text layer. For those, running a 0.9B vision-language model to "read" what is already machine-readable text is wasteful. The layout model still runs — it is non-negotiable for correct bboxes and semantic structure. Only the VLM text-generation step is replaced with direct PDF extraction via pdfplumber (MIT license, SaaS-safe).

---

## What each component provides

This is the most important thing to understand before designing anything:

| Component | Provides | Does NOT provide |
|-----------|----------|-----------------|
| Layout model (PP-DocLayoutV3) | bbox positions + semantic type (title/body/table/figure) + reading order | text content |
| VLM / GLM-OCR (vLLM) | text content by reading image crops | structure/type labels |
| pdfplumber | text content + word-level positions from PDF text layer | semantic structure |

**The layout model is non-negotiable.** pdfplumber cannot tell you whether a region is a title or a body paragraph — it gives you words scattered across the page with no semantic meaning. The layout model is the only component that produces correct region bboxes with semantic labels.

**What the hybrid replaces:** only the VLM text-generation step, and only for text/title regions in searchable PDFs. Layout detection always runs.

```
Layout model (always) → bboxes + semantic types
                              │
              ┌───────────────┴───────────────┐
         type=text/title               type=table/figure
              │                               │
    PDF searchable? ──yes──► pdfplumber     VLM always
              │
            ──no──► VLM fallback
```

---

## Two implementation paths

There are two existing stacks to choose from as the layout detection source:

| | Path A: GLM-OCR stack | Path B: HPS/Triton stack |
|--|----------------------|--------------------------|
| Layout model | PP-DocLayoutV3 (PyTorch/HuggingFace, in-process) | PP-DocLayoutV3 (PaddleX/Triton, isolated process) |
| VLM | GLM-OCR 0.9B via vLLM | PaddleOCR-VL-1.5-0.9B via vLLM |
| VLM cut point | After glmocr receives regions | Inside `layout-parsing` Triton model via config flag |
| VRAM pressure | High — layout (4 GB) + vLLM (19 GB) on same pool | Low — Triton isolates GPU allocation |
| Infrastructure | supervisord, single container | Docker Compose, 3 containers |
| Post-processing | glmocr server handles internally | `restructure-pages` Triton model (cross-page merging) |

**Path B (HPS/Triton) is recommended** — better layout quality isolation, dynamic batching, no OOM contention, and the VLM can be fully dropped from the deployment with a single config change.

---

## Current pipelines (what exists today)

### GLM-OCR stack (vast.ai / RunPod)

```
PDF → page_loader renders pages as images
    → PPDocLayoutV3 (PyTorch, in-process, ~4 GB VRAM)
        → bboxes + labels
    → for each region: crop → vLLM (GLM-OCR 0.9B, ~19 GB VRAM)
        → text tokens
    → glmocr server assembles JSON response
```

**Bottleneck:** vLLM inference. Each 4-page chunk takes ~15–25s on RTX 3090.
**Throughput:** ~1.44 pages/s at concurrency=8.
**OOM issue:** layout model (4 GB) and vLLM (19 GB) fight for 23.56 GB GPU, causing periodic 276 MiB allocation failures on layout batches. Retried automatically, always returns HTTP 200 but adds latency.

Deployed infra:
- vLLM on port 8000, `--gpu-memory-utilization 0.80`, `--max-model-len 4096`, `--max-num-seqs 32`, MTP `num_speculative_tokens:3`
- glmocr server on port 5002, `python -m glmocr.server --config /etc/glmocr_config.yaml`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on glmocr process
- API: `POST /glmocr/parse` with `{"images": ["data:application/pdf;base64,<b64>"]}`
- RunPod image: `ringkasannet/glm-ocr-worker:v1.2`

Known critical bug (already patched): glmocr's `page_loader.py` does not handle `data:application/pdf;base64,...` URIs — silently returns HTTP 200 with 0 regions. Fix is a post-install patch in `start.sh` and `handler.py`. See `DEPLOYMENT_NOTES.md` entry 6.

### HPS stack (Docker Compose, 3 containers)

```
Client → FastAPI gateway :8080
    → Triton gRPC :8001
        → "layout-parsing" model (GPU)
              ├─ PP-DocLayoutV3 → bboxes + labels
              └─ VLRecognition → vLLM at localhost:8118 → text
    → "restructure-pages" model (CPU)
        → cross-page table merge + title hierarchy
```

Three containers with strict startup ordering (vlm-server must be healthy before Triton, Triton before gateway):
- **paddleocr-vlm-server**: `paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --port 8080 --backend vllm`
- **paddleocr-vl-tritonserver**: Triton, port 8001 (gRPC), runs both `layout-parsing` and `restructure-pages` models
- **paddleocr-vl-api**: FastAPI gateway, port 8080

Gateway concurrency: 16 semaphore slots for `/layout-parsing` (inference), 64 for `/restructure-pages` (CPU post-processing).

Response structure from `layout-parsing`:
```json
{
  "result": {
    "layoutParsingResults": [
      {
        "prunedResult": {
          "parsing_res_list": [
            {"type": "text", "bbox": [x1,y1,x2,y2], "text": "...", "page_num": 0}
          ]
        }
      }
    ]
  }
}
```

---

## How layout detection works in each stack

### PaddleOCR (HPS) — PP-DocLayoutV3 via PaddleX/Triton

Model: PP-DocLayoutV3, DETR-based object detector, 25 region classes.
Config file: `runpod/hps/pipeline_config_local.yaml`.

```yaml
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayoutV3
    batch_size: 8
    threshold: 0.3
    layout_nms: True
    layout_unclip_ratio: [1.0, 1.0]
    layout_merge_bboxes_mode:
      0: "union"   # per-class merge strategy
      3: "large"   # e.g. tables use "large" (keep enclosing box)
      ...
  VLRecognition:
    module_name: vl_recognition
    model_name: PaddleOCR-VL-1.5-0.9B
    batch_size: 4096
    genai_config:
      backend: vllm-server
      server_url: http://localhost:8118/v1   # vLLM inside Triton container
```

Post-processing (`apply_layout_postprocess`):
- Soft NMS: `iou_same=0.6`, `iou_diff=0.98`
- Filters boxes covering >82–93% of image (page-level noise)
- Merge nested bboxes by class: `union` / `large` / `small`
- `unclip_ratio`: expands boxes outward to avoid edge clipping
- Output includes `polygon_points` (rotated/curved regions) and `order` (reading order index)

25 label classes: `text`, `title`, `table`, `figure`, `figure_title`, `formula`, `formula_number`, `header`, `footer`, `footnote`, `reference`, `abstract`, `algorithm`, `chart`, `aside_text`, `doc_title`, `content`, `display_formula`, `inline_formula`, `seal`, `vertical_text`, `number`, `paragraph_title`, `reference_content`, `vision_footnote`

### GLM-OCR — PP-DocLayoutV3 via HuggingFace PyTorch

Same model (PP-DocLayoutV3), different inference backend. Runs in-process alongside vLLM.

```python
class PPDocLayoutDetector:
    def start(self):
        self._image_processor = PPDocLayoutV3ImageProcessor.from_pretrained(model_dir)
        self._model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_dir)
        self._model = self._model.to("cuda:0")

    def _run_detection_single_image(self, image, threshold):
        inputs = self._image_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return self._image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[image.size[::-1]])[0]
```

Output per region:
```python
{
    "cls_id": 2,
    "label": "text",
    "score": 0.94,
    "coordinate": [x1, y1, x2, y2],   # pixel coords
    "order": 3,
    "polygon_points": np.array(...)
}
```

VRAM cost: ~4 GB. Source of OOM contention with vLLM on RTX 3090.

### Triton vs in-process comparison

| | GLM-OCR (in-process) | HPS (Triton) |
|--|---------------------|-------------|
| VRAM isolation | Shares pool with vLLM → OOM retries | Isolated → no contention |
| Batching | Manual `batch_size` config | Automatic dynamic batching |
| OOM risk | High (4 GB + 19 GB = 23 GB on 23.56 GB GPU) | Low |
| Deployment | Simple (supervisord) | 3 containers (Docker Compose) |
| Throughput ceiling | In-process GIL + VRAM contention | Scales with Triton instances |

---

## Path B implementation: HPS/Triton with VLM cut

The VLM call happens **inside** the `layout-parsing` Triton model, configured in `pipeline_config_local.yaml`. The gateway is a pure proxy — it does not call the VLM directly.

### Three files to change

**1. `runpod/hps/pipeline_config_local.yaml`** — disable VLRecognition:

```yaml
pipeline_name: PaddleOCR-VL-1.5

use_doc_preprocessor: False
use_layout_detection: True
use_chart_recognition: False
use_seal_recognition: False
use_vl_recognition: False     # ← add this line
format_block_content: False
merge_layout_blocks: True
# ... rest unchanged
```

This flag follows the same `use_*` convention as all other pipeline stages. If the flag is not recognised by the SDK version in use, fallback: set `server_url` to a non-existent endpoint so VLRecognition calls fail silently and return empty text, while layout results still come through.

**2. `official-hps/compose.yaml`** — drop the vLLM container:

```yaml
# Remove the entire paddleocr-vlm-server service block
# Remove depends_on: paddleocr-vlm-server from paddleocr-vl-tritonserver

services:
  paddleocr-vl-api:
    # unchanged
    depends_on:
      paddleocr-vl-tritonserver:
        condition: service_healthy

  paddleocr-vl-tritonserver:
    # unchanged, but remove:
    # depends_on:
    #   paddleocr-vlm-server:
    #     condition: service_healthy
```

Dropping `paddleocr-vlm-server` frees the ~19 GB VRAM it holds and removes the 300s startup wait it requires.

**3. `official-hps/gateway/app.py`** — make VLM health check non-blocking:

```python
# In the ready() endpoint, remove or guard the VLM check:
# vlm_ready = await _check_vlm_ready()
# if not vlm_ready:
#     return JSONResponse(status_code=503, ...)

# Replace with:
# (simply omit the block — VLM is no longer part of the stack)
```

Also remove `VLM_URL` from the env vars in compose.yaml if desired, or leave it and just stop checking it.

### New hybrid layer service

Sits between your application and the HPS gateway. The gateway and Triton stack are unchanged — this is a new wrapper service.

```python
import io, base64, requests, pdfplumber

GATEWAY_URL = "http://localhost:8080"

TEXT_REGION_TYPES = {
    "text", "title", "paragraph_title", "abstract",
    "content", "reference", "footnote", "doc_title",
}

def has_text_layer(pdf_bytes: bytes, sample_pages: int = 3) -> bool:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        chars = sum(len(p.extract_text() or "") for p in pdf.pages[:sample_pages])
    return (chars / max(sample_pages, 1)) > 50

def extract_text_at_bbox(pdf: pdfplumber.PDF, page_num: int,
                          bbox_px: list, render_dpi: int = 150) -> str:
    page = pdf.pages[page_num]
    scale = 72 / render_dpi   # px → PDF points
    x0, y0, x1, y1 = [c * scale for c in bbox_px]
    cropped = page.within_bbox((x0, y0, x1, y1))
    return (cropped.extract_text(x_tolerance=3, y_tolerance=3) or "").strip()

def process_document(pdf_bytes: bytes) -> dict:
    b64 = base64.b64encode(pdf_bytes).decode()

    # Step 1: layout detection (VLRecognition disabled → no VLM call)
    resp = requests.post(
        f"{GATEWAY_URL}/layout-parsing",
        json={"file": b64, "fileType": 0, "visualize": False},
        timeout=120,
    )
    resp.raise_for_status()
    layout = resp.json()

    # Step 2: fill text using pdfplumber for text regions
    searchable = has_text_layer(pdf_bytes)
    if searchable:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_result in layout["result"]["layoutParsingResults"]:
                for block in page_result["prunedResult"]["parsing_res_list"]:
                    if block.get("type") in TEXT_REGION_TYPES:
                        block["text"] = extract_text_at_bbox(
                            pdf,
                            block["page_num"],
                            block["bbox"],
                        )
                    # Non-text regions (figure, table, chart):
                    # block["text"] is empty here because VLRecognition is off.
                    # Call GLM-OCR here if you need figure/table text.

    # Step 3: restructure across pages (title hierarchy, cross-page table merge)
    restructured = requests.post(
        f"{GATEWAY_URL}/restructure-pages",
        json=layout["result"],
        timeout=60,
    ).json()

    return restructured
```

---

## Coordinate system — critical detail

The layout model returns bboxes in **pixel coordinates** of the rendered page image. pdfplumber works in **PDF points** (1 pt = 1/72 inch). You must scale before intersecting.

```python
RENDER_DPI = 150         # DPI used to render the page image
SCALE = 72 / RENDER_DPI  # 72/150 = 0.48 pt/px

# bbox_px from layout model: [x0, y0, x1, y1] in pixels
x0_pt, y0_pt, x1_pt, y1_pt = [c * SCALE for c in bbox_px]
```

pdfplumber `top` is measured from the **top of the page** (same as most renderers). If your renderer uses bottom-up y-coordinates, flip: `y_pt = page_height_pt - (y_px * SCALE)`.

---

## What still needs VLM even for searchable PDFs

| Region type | Why VLM is still needed |
|-------------|------------------------|
| `figure`, `chart` | Image-only, no text layer |
| `table` | Text layer exists but row/column structure requires model inference |
| `formula`, `display_formula` | Math notation, not readable as plain text |
| `seal` | Rotated/distorted text |
| Any region in scanned pages | No text layer at all |

Strategy: after pdfplumber fills text regions, call GLM-OCR (or any VLM) only for the remaining block types. If no VLM is available, these blocks return empty text.

---

## Verification steps before building

**1. Confirm `use_vl_recognition: False` is a valid flag** — restart Triton with the flag added and check whether `parsing_res_list` blocks have empty `text` fields (flag works) or whether Triton errors at startup (flag not recognised).

**2. Confirm `restructure-pages` handles empty text blocks** — it performs structural operations (table merge, title hierarchy by font size/position). These should work regardless of text content. Send it a result with empty text blocks and verify the output structure is sane.

**3. Verify target documents have a text layer:**

```bash
pip install pdfplumber
python - <<'EOF'
import pdfplumber, sys, io
with pdfplumber.open(sys.argv[1]) as pdf:
    for i, page in enumerate(pdf.pages[:5]):
        t = page.extract_text() or ""
        print(f"page {i+1}: {len(t):4d} chars  {'[TEXT]' if len(t)>50 else '[SCAN]'}")
EOF your_document.pdf
```

If most pages are `[SCAN]`, the hybrid adds no value — run GLM-OCR on everything as today.

---

## Licensing

| Library | License | SaaS-safe? |
|---------|---------|------------|
| pdfplumber | MIT | ✓ Yes |
| pdfminer.six (pdfplumber dep) | MIT | ✓ Yes |
| PyMuPDF / fitz | AGPL v3 or commercial (Artifex) | ✗ AGPL requires open-sourcing your app |
| pypdf | BSD-3 | ✓ Yes (less spatial precision) |

**Do not use PyMuPDF** without a commercial license from Artifex Software.

---

## Files in the repos

### GLM-OCR stack (`runpod/glm-ocr/` in PaddleOCR-main repo)

```
start.sh              # vast.ai provisioning (vLLM + glmocr via supervisord)
handler.py            # RunPod serverless handler
Dockerfile            # Docker image (ringkasannet/glm-ocr-worker)
build_push.sh         # Build + push to Docker Hub
test_glmocr.py        # Benchmark (concurrency ladder, non-overlapping chunks)
glmocr_config.yaml    # Placeholder — written at runtime by handler.py
DEPLOYMENT_NOTES.md   # 13 critical findings (bugs, perf, config)
```

`DEPLOYMENT_NOTES.md` is essential reading — covers the silent 0-regions bug, supervisor xmlrpc conflict, HF_TOKEN inheritance, OOM retry behaviour, and GPU utilization findings.

### HPS stack (`paddleocr-server` repo)

```
official-hps/
├── compose.yaml              # 3-container deployment (gateway + Triton + vLLM)
├── gateway/app.py            # FastAPI proxy to Triton gRPC
├── gateway.Dockerfile
└── tritonserver.Dockerfile   # Triton + PaddleX HPS SDK

runpod/hps/
├── pipeline_config_local.yaml   # PaddleX pipeline config (LayoutDetection + VLRecognition)
└── volume_prep/                 # Triton binaries
```
