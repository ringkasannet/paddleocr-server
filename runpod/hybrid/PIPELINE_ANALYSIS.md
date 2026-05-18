# GLM-OCR Pipeline — Comprehensive Analysis

## 1. High-Level Architecture

The stack is a two-process, single-GPU document OCR system. The two processes share one GPU and communicate over localhost HTTP.

```
┌─────────────────────────────────────────────────────────┐
│                     Container / VM                       │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  glmocr server  :5002                           │    │
│  │                                                  │    │
│  │  1. page_loader   – PDF → page images           │    │
│  │  2. PPDocLayoutV3 – images → bboxes + labels    │    │
│  │  3. region crops  – cut each bbox from image    │    │
│  │  4. prompt build  – wrap crop in VLM prompt     │    │
│  │  5. HTTP → vLLM   – generate text tokens        │    │
│  │  6. JSON assembly – regions + text → response   │    │
│  └──────────────────┬──────────────────────────────┘    │
│                     │ HTTP :8000                         │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │  vLLM server  :8000                             │    │
│  │  GLM-OCR 0.9B VLM (OpenAI-compatible API)       │    │
│  │  MTP speculative decoding                       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Entry point:** `POST /glmocr/parse`  
**Input:** `{"images": ["data:application/pdf;base64,<b64>"]}`  
**Output:** `{"json_result": [[region, ...]], "markdown_result": "...", ...}`

---

## 2. Pipeline Stages — Step by Step

### Stage 0: Worker Initialisation (runs once per container)

`handler.py` lines 186–190 execute four steps before accepting any request:

1. **Write config** — copies the package default `glmocr/config.yaml` to `/tmp/glmocr_config.yaml`, patches port numbers, GPU device, batch size, and max token count via regex substitution. The copy-then-patch strategy is critical: a minimal YAML write performs a shallow merge that silently drops required nested keys like `threshold` and `id2label`, crashing the layout model at startup (DEPLOYMENT_NOTES.md §5).

2. **Patch page_loader** — modifies the installed `glmocr/dataloader/page_loader.py` in-place to add a `data:application/pdf;base64,...` branch. Without this patch the server returns HTTP 200 with 0 regions and no error, because the exception from `_load_image()` is caught and logged as a warning (DEPLOYMENT_NOTES.md §6). This is the single most dangerous silent failure in the system.

3. **Start vLLM subprocess** — launches `python -m vllm.entrypoints.openai.api_server` with these flags:
   - `--model zai-org/GLM-OCR` — the 0.9B vision-language model
   - `--gpu-memory-utilization 0.80` — pre-allocates 80% of VRAM as the KV cache (18.8 GB on RTX 3090)
   - `--max-model-len 4096` — context window cap
   - `--max-num-seqs 32` — maximum parallel sequences in flight
   - `--speculative-config {"method":"mtp","num_speculative_tokens":3}` — MTP speculative decoding
   - `--trust-remote-code` — required for custom model code in the GLM-OCR repo
   - Polls `GET /health` every 3s, times out at 300s

4. **Start glmocr subprocess** — launches `python -m glmocr.server --config /tmp/glmocr_config.yaml` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce CUDA allocator fragmentation (important because glmocr and vLLM fight for the remaining ~4.7 GB after vLLM pre-allocates). Polls the root endpoint every 2s, times out at 60s.

---

### Stage 1: PDF Ingestion — `page_loader.py`

**What it does:** Converts the raw PDF bytes into a list of PIL Image objects, one per page.

**Input formats accepted:**
- `data:application/pdf;base64,...` — base64-encoded PDF as a data URI (requires the post-install patch)
- `file:///path/to/doc.pdf` — local file path
- Raw bytes starting with `b"%PDF-"` — directly passed to `_load_pdf_bytes()`
- `data:image/...;base64,...` — single-image input

**PDF rendering:** Uses an internal `_load_pdf_bytes()` / `_iter_pdf_bytes()` method that renders each page as a raster image. The default DPI is 150. This DPI value is important: it defines the pixel coordinate space that the layout model operates in, and you must apply a `72/150 = 0.48` scale factor when mapping bboxes back to PDF point space (e.g., for pdfplumber overlap queries — see HANDOFF_HYBRID_EXTRACTOR.md).

**Output:** A list of `(page_image, unit_index)` tuples where `unit_index` identifies which source document the page came from. Multiple source images/PDFs can be batched in a single request.

---

### Stage 2: Layout Detection — PP-DocLayoutV3

**Model:** `PaddlePaddle/PP-DocLayoutV3_safetensors` (~600 MB, loaded to `cuda:0`)  
**Architecture:** DETR-based object detector (Detection Transformer)  
**VRAM cost:** ~3.7–4.0 GB at batch_size=2

**What it does:** Runs a full-image object detection pass on each page image to locate every content region and classify it into one of 25 semantic types.

**25 label classes:**

| Category | Labels |
|----------|--------|
| Body text | `text`, `paragraph_title`, `abstract`, `content` |
| Headings | `title`, `doc_title` |
| Tables | `table` |
| Figures | `figure`, `figure_title`, `chart` |
| Math | `formula`, `formula_number`, `display_formula`, `inline_formula` |
| Navigation | `header`, `footer`, `footnote`, `reference`, `reference_content`, `vision_footnote` |
| Special | `algorithm`, `aside_text`, `seal`, `vertical_text`, `number` |

**Detection process:**

```python
class PPDocLayoutDetector:
    def _run_detection_single_image(self, image, threshold):
        inputs = self._image_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return self._image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[image.size[::-1]])[0]
```

**Post-processing (apply_layout_postprocess):**
- Soft NMS with `iou_same=0.6` (same-class suppression) and `iou_diff=0.98` (cross-class suppression)
- Filters bounding boxes that cover >82–93% of the image (page-spanning noise)
- Merges nested bboxes by class using three strategies: `union` (merge by class), `large` (keep enclosing box), `small` (keep smaller box)
- `unclip_ratio`: expands bboxes outward by a configurable ratio to avoid edge clipping
- Assigns a `reading_order` index to each region

**Output per region:**
```python
{
    "cls_id": 2,
    "label": "text",
    "score": 0.94,
    "coordinate": [x0, y0, x1, y1],  # pixel coords at 150 DPI
    "order": 3,                        # reading order index
    "polygon_points": [[x,y], ...]     # polygon (rotated/curved regions)
}
```

**Critical performance constraint:** This model loads at startup and stays resident on the GPU. At concurrency > 4, parallel requests trigger parallel layout detection batches. Each batch needs ~276 MB of headroom. On RTX 3090 with vLLM at 80% util, only ~1.0 GB is available for layout batches — causing periodic OOM errors at concurrency 8. These are retried automatically (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces retry frequency), so the server always returns HTTP 200, but latency spikes. Despite this, 80% util outperforms 65% (which has no OOM retries) because the larger KV cache reduces vLLM evictions more than the OOM retries hurt (DEPLOYMENT_NOTES.md §12).

---

### Stage 3: Region Cropping

After layout detection, glmocr loops over the detected regions and crops each one from the page image using the `coordinate` (bounding box) returned by the layout model. The crop is a PIL Image subimage.

The crop is used as input to the VLM in Stage 4. For regions like `figure`, `table`, `formula`, and `seal` the VLM must read the image directly. For `text` and `title` regions the crop is largely redundant if a text layer exists (which is the basis of the hybrid optimization in Stage 7).

---

### Stage 4: VLM Text Extraction — GLM-OCR via vLLM

**Model:** `zai-org/GLM-OCR` (0.9B parameter vision-language model)  
**Inference backend:** vLLM with OpenAI-compatible `/v1/chat/completions`  
**VRAM cost:** ~18.8 GB (pre-allocated KV cache at 80% GPU util on RTX 3090)

**What it does:** For each cropped region image, glmocr builds a multimodal chat prompt and submits it to vLLM. vLLM decodes text tokens that represent the OCR output for that region.

**Prompt structure (conceptual):**
```
[system] You are an OCR assistant. Read the text in this image.
[user] <image: base64-encoded crop>
        Extract all text, preserving layout.
[assistant] → decoded tokens
```

**vLLM inference settings:**
- `max_tokens: 2048` — hard cap per region (vLLM rejects `max_tokens > max_model_len=4096`; the default glmocr config of 8192 causes a startup crash — DEPLOYMENT_NOTES.md §7)
- `max_num_seqs: 32` — up to 32 crops processed in parallel within vLLM's continuous batching scheduler
- MTP speculative decoding with `num_speculative_tokens: 3` — draft tokens generated in parallel to accelerate autoregressive decoding

**Throughput bottleneck:** VLM inference is the dominant cost. Each 4-page chunk takes 15–25s on RTX 3090. At concurrency=8, throughput is ~1.44–1.77 pages/s. Larger batch size (more pages per chunk) amortizes per-request overhead.

---

### Stage 5: JSON Assembly

After all regions for a request are processed, glmocr assembles the response:

```json
{
  "id": "chatcmpl-...",
  "created": 1779015185,
  "json_result": [
    [
      {
        "bbox_2d":     [x0, y0, x1, y1],
        "content":     "extracted text from VLM",
        "index":       0,
        "label":       "text",
        "native_label":"doc_title",
        "polygon":     [[x,y], [x,y], [x,y], [x,y]]
      }
    ]
  ],
  "markdown_result": "# Title\n\nBody text...",
  "data_info": {"pages": [...]}
}
```

`json_result` is a list-of-lists: outer list = pages, inner list = regions per page. Each region carries the layout model's bbox/label alongside the VLM's extracted text content.

`markdown_result` is assembled by rendering regions in reading order, using `native_label` to pick Markdown syntax (e.g., `doc_title` → `#`, `text` → paragraph).

---

### Stage 6: RunPod Handler (request relay)

`handler.py`'s `handler(job)` function is a thin HTTP relay:

```python
def handler(job):
    r = requests.post(
        f"http://127.0.0.1:{GLMOCR_PORT}/glmocr/parse",
        json=job["input"],
        timeout=300,
    )
    return r.json() if r.ok else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
```

The RunPod serverless framework handles worker lifecycle, job queuing, and scaling. The handler is stateless per request; all state lives in the two persistent subprocesses.

---

### Stage 7: Hybrid Extraction (proposed optimization — not yet deployed)

Described in `HANDOFF_HYBRID_EXTRACTOR.md`. The key insight: layout detection is always required (it provides semantic structure), but VLM text generation is only necessary when no text layer exists.

```
Input PDF
    │
    ├── has_text_layer()? ──no──→ VLM for all regions (current path)
    │
    └──yes──→ Layout detection (always)
                   │
                   ├── type in {text, title, paragraph_title, ...}
                   │        └──→ pdfplumber bbox extraction (fast, free)
                   │
                   └── type in {table, figure, formula, seal}
                            └──→ VLM (still required, image-only content)
```

This can reduce VLM calls by 70–90% for typical business/government PDFs where most content is machine-readable text. The table/figure/formula regions still need VLM because their visual structure cannot be reconstructed from text-layer coordinates alone.

**Coordinate mapping:**
```python
RENDER_DPI = 150
SCALE = 72 / RENDER_DPI  # = 0.48 pt/px

x0_pt, y0_pt, x1_pt, y1_pt = [c * SCALE for c in bbox_px]
cropped = pdf_page.within_bbox((x0_pt, y0_pt, x1_pt, y1_pt))
text = cropped.extract_text(x_tolerance=3, y_tolerance=3)
```

---

## 3. Data Flow Summary

```
POST /glmocr/parse
{"images": ["data:application/pdf;base64,<b64>"]}
    │
    ▼
page_loader.py
  _load_pdf_bytes() → renders at 150 DPI
  → [PIL.Image page0, PIL.Image page1, ...]
    │
    ▼
PPDocLayoutV3 (cuda:0, ~4 GB VRAM)
  per-page detection pass
  → [{label, bbox, score, order, polygon}, ...]
    │
    ▼
region cropper
  image.crop(bbox) for each region
  → [PIL.Image crop0, PIL.Image crop1, ...]
    │
    ▼
glmocr HTTP client
  POST http://localhost:8000/v1/chat/completions
  multimodal prompt: system + image crop
  max_tokens=2048, model="glm-ocr"
    │
    ▼
vLLM (cuda:0, ~19 GB VRAM, continuous batching)
  up to 32 concurrent sequences
  MTP speculative decoding (3 draft tokens)
  → decoded text per region
    │
    ▼
JSON assembler
  merge layout bboxes + VLM text + reading order
  render markdown_result
    │
    ▼
{"json_result": [[...]], "markdown_result": "...", "data_info": {...}}
```

---

## 4. Deployment Options

### 4.1 RunPod Serverless (current — `handler.py` + `Dockerfile`)

**How it works:**  
A Docker image with both model weights baked in (~2.4 GB of model files) is deployed as a RunPod serverless worker. RunPod manages instance lifecycle: scales to zero when idle, warms up a container on first request, routes jobs through its queue.

**Cold start sequence:**  
Container starts → `handler.py` module-level code runs → vLLM starts (60–120s for model load) → glmocr starts (5–10s) → RunPod marks worker ready → requests accepted

**Pros:**
- Zero infra management — no servers to maintain
- Pay-per-second billing: $0 when idle
- Automatic horizontal scaling (RunPod spins more workers under load)
- Model weights baked in: no HuggingFace download on cold start

**Cons:**
- Cold starts are expensive: 2–3 minutes before the first request is served
- RunPod queues requests during cold start — downstream timeouts if caller doesn't retry
- Max 1 GPU per worker in standard serverless mode (no multi-GPU tensor parallelism)
- No persistent state across invocations (fine for this stateless pipeline)

**Best for:** Intermittent/bursty workloads where you can tolerate cold-start latency. A job queue with retry handles cold starts transparently.

**Docker image:** `ringkasannet/glm-ocr-worker:v1.2`  
**Key env vars:** `MODEL`, `GPU_MEM_UTIL`, `MAX_MODEL_LEN`, `MAX_TOKENS`, `HF_TOKEN`

---

### 4.2 Vast.ai Persistent Instance (`start.sh`)

**How it works:**  
Rent a GPU VM by the hour. `start.sh` provisions it: installs vLLM and glmocr, patches `page_loader.py`, writes the glmocr config, creates supervisord configs, and registers everything with supervisor. vLLM and glmocr run as persistent supervised processes.

**Pros:**
- No cold starts after initial setup (~5–10 minute provisioning)
- Full control: SSH access, file system, process inspection
- Cheapest per-GPU-hour for high-utilization workloads
- `INSTANCES=2` mode: two vLLM instances behind nginx round-robin load balancer, each at 38% GPU util

**Cons:**
- Manual provisioning (run `start.sh` after instance boot)
- You pay while idle (no scale-to-zero)
- No automatic horizontal scaling — you manage instance count
- `INSTANCES=2` is only useful if you have a GPU with > 40 GB VRAM; on RTX 3090 (24 GB) it is not viable — each instance needs ~9 GB at 38% util, which is feasible, but then glmocr's layout model (4 GB) has only ~5.5 GB remaining for both layout and KV cache for both vLLM instances — tight.

**Known pitfalls:**
- Supervisor xmlrpc conflict: `deactivate` before `supervisorctl` (§2)
- HF_TOKEN not inherited: inject into supervisor conf env (§4)
- Port 5002 must be declared in vast.ai template, or use SSH tunnel (§10)
- `supervisorctl restart` fails on FATAL state; use `start` (§8)

**Best for:** Sustained high-throughput batch workloads running for hours or days.

---

### 4.3 HPS/Triton Stack (described in `HANDOFF_HYBRID_EXTRACTOR.md`, lives in `paddleocr-server` repo)

**How it works:**  
Three Docker Compose containers with strict health-check ordering:
1. `paddleocr-vlm-server` — vLLM serving `PaddleOCR-VL-1.5-0.9B` on port 8118
2. `paddleocr-vl-tritonserver` — Triton Inference Server exposing `layout-parsing` (GPU) and `restructure-pages` (CPU) models
3. `paddleocr-vl-api` — FastAPI gateway on port 8080 that translates REST to Triton gRPC

**Pros:**
- GPU memory isolation: Triton manages its own allocations; vLLM does not share the same allocator pool as the layout model → no OOM contention
- Dynamic batching: Triton queues layout detection requests and batches them automatically
- VLM can be disabled in one config line (`use_vl_recognition: False` in `pipeline_config_local.yaml`) — the stack becomes layout-only, freeing 19 GB VRAM
- `restructure-pages` handles cross-page operations (table continuation, title hierarchy) — not available in the GLM-OCR stack
- Scales horizontally by adding Triton replicas

**Cons:**
- More operational complexity: 3 containers, health-check ordering, Triton model repository
- Startup order is strict: vLLM must be healthy before Triton, Triton before gateway
- `layout-parsing` gateway semaphore: 16 concurrent inference slots, 64 CPU slots — must tune for your concurrency target

**Best for:** Production deployments that need layout-only mode, better VRAM isolation, or cross-page structural post-processing.

---

### 4.4 Comparison Table

| Dimension | RunPod Serverless | Vast.ai Persistent | HPS/Triton |
|-----------|-------------------|--------------------|------------|
| Cold start | 2–3 min | None (after provision) | ~5 min (compose up) |
| Idle cost | $0 | Full GPU hourly rate | Full GPU hourly rate |
| GPU memory isolation | None | None | Yes (Triton) |
| OOM retries | Yes | Yes | No |
| Cross-page post-processing | No | No | Yes |
| Scale to zero | Yes | No | No |
| Horizontal scale | Auto (RunPod) | Manual | Docker replicas |
| VLM-off mode | Manual (env var) | Manual (config) | Single config flag |
| Hybrid extraction support | Requires extra service | Requires extra service | Native (Path B) |
| Operational complexity | Low | Medium | High |

---

## 5. Optimization Analysis

### 5.1 GPU Memory — The Central Constraint

On RTX 3090 (23.56 GB):

| Component | VRAM at 80% util |
|-----------|-----------------|
| vLLM KV cache | ~18.8 GB |
| Layout model (PP-DocLayoutV3) | ~3.7 GB |
| Headroom for layout batches | ~1.0 GB |
| Each layout batch needs | ~276 MB |
| Max concurrent layout batches | ~3–4 |

The two models cannot share GPU memory without contention. The sweet spot found empirically is **80% GPU util** — counterintuitively better than 65% because the larger KV cache reduces vLLM sequence evictions more than OOM retries hurt (DEPLOYMENT_NOTES.md §12):

| GPU util | Concurrency | Pages/s |
|----------|-------------|---------|
| 65% | 8 | 1.11 |
| 75% | 8 | 1.46 |
| 80% | 8 | 1.44 |
| 83% | 8 | **1.77** |

Adding `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to the glmocr process environment reduces fragmentation in the remaining ~4.7 GB, cutting OOM retry frequency.

**On a 40 GB GPU (A100/A6000):** Both models fit with room to spare. Set `--gpu-memory-utilization 0.75` to give the layout model ~10 GB headroom — OOM retries effectively disappear.

**On a 48 GB GPU (RTX A6000/RTX 6000 Ada):** Can run two separate vLLM instances (each at 45% util = ~21.6 GB each = 43.2 GB total). The `INSTANCES=2` mode in `start.sh` is designed for this.

---

### 5.2 Chunking Strategy

The layout detection overhead is per-request, not per-page. This makes chunk size a critical throughput lever (DEPLOYMENT_NOTES.md §13):

| Chunk size | Concurrency | Pages/s | Notes |
|------------|-------------|---------|-------|
| 1 page | 8 | 0.66 | Layout overhead paid per page |
| 1 page | 16 | 0.87 | More concurrent layout → more OOM |
| 1 page | 32 | 0.60 | OOM cascade → silent empty responses |
| 4 pages | 8 | **1.46** | Optimal: amortizes layout overhead |

**Rule:** Use 4-page chunks at concurrency 8. Do not go above concurrency 12 on RTX 3090 — concurrent layout batches exhaust the 1 GB headroom.

**Silent failure trap:** At high concurrency, concurrent layout OOMs return HTTP 200 with `json_result: []` (0 KB response). The caller sees a success with no content. Always validate `len(response["json_result"]) > 0` before treating a response as complete.

---

### 5.3 Speculative Decoding — MTP

MTP (Medusa-style token prediction) generates `n` draft tokens in one forward pass, then verifies them with the base model. Accepted drafts skip `n` autoregressive steps.

**Current config:** `num_speculative_tokens: 3`  
**Official vLLM recipe recommendation:** `num_speculative_tokens: 1`

Using 3 is untested against the official recommendation (DEPLOYMENT_NOTES.md §11). The optimal value depends on the draft acceptance rate, which varies with document complexity. For short OCR outputs (a few words per region), 1 draft token is likely optimal. For longer outputs (dense text paragraphs), 3 may help. Profile with the benchmark script if tuning.

---

### 5.4 max_tokens Alignment

glmocr default config: `max_tokens: 8192`  
vLLM launched with: `--max-model-len 4096`  
Result: vLLM startup crash — `max_tokens=8192 cannot be greater than max_model_len=4096`

**Correct values:**
- `--max-model-len 4096` (vLLM flag)
- `max_tokens: 2048` (glmocr config — half of max_model_len to leave room for prompt tokens)

The distinction matters: `max_model_len` sets the total context window (prompt + response). `max_tokens` caps only the response length. A typical OCR region prompt (image encoding + system prompt) consumes ~1500–2000 tokens, leaving 2048–2500 tokens for the response — consistent with the `max_tokens: 2048` setting.

---

### 5.5 Hybrid Extraction Optimization (Path B)

For searchable PDFs (Word → PDF, typeset documents), the VLM adds latency and VRAM pressure with no accuracy benefit for text regions. pdfplumber reads the embedded text layer with sub-millisecond latency per page vs. 15–25s per 4-page chunk for VLM inference.

**Expected speedup for typical business/government PDFs:**
- Text layer coverage: ~80% of regions are text/title
- VLM calls reduced to: ~20% of regions (table, figure, formula, seal)
- Throughput estimate: 8–15x faster for text-dominant PDFs

**What still needs VLM after hybrid:**

| Region type | Why |
|-------------|-----|
| `table` | Row/column structure requires image understanding |
| `figure`, `chart` | Image-only content, no text layer |
| `formula`, `display_formula` | Math notation |
| `seal` | Rotated, distorted text |
| Any region on scanned pages | No text layer |

**Implementation path (2–3 days):**
1. Add pdfplumber to dependencies
2. Wrap the `/glmocr/parse` call with a `has_text_layer()` check
3. After layout JSON is received, iterate regions and call `extract_text_at_bbox()` for text/title types
4. Apply the `72/150` coordinate scale factor

**Licensing:** pdfplumber (MIT), pypdf (BSD-3). Do not use PyMuPDF — AGPL requires open-sourcing your application.

---

### 5.6 Batch Size Tuning

glmocr config: `batch_size: 2`  
vast.ai DEPLOYMENT_NOTES config: `batch_size: 4` (from sed patch)

Increasing batch size processes more page images simultaneously through the layout model, improving GPU utilization for the detection pass. However, each image in the batch consumes ~276 MB of the layout model's allocation, so `batch_size: 4` needs ~1.1 GB of headroom — cutting into the already tight ~1.0 GB available on RTX 3090.

**Recommendation:**
- RTX 3090 (24 GB): `batch_size: 2`
- A100 (40 GB) or better: `batch_size: 4–8`

---

### 5.7 Docker Layer Ordering

The Dockerfile is correctly optimized:

```dockerfile
# Layer 1: base image (never changes)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Layer 2: Python packages (changes when deps change)
RUN pip install uv && uv pip install vllm transformers glmocr runpod

# Layer 3: model weights (changes only when model version changes)
ARG HF_TOKEN=""
RUN huggingface-cli download zai-org/GLM-OCR && \
    huggingface-cli download PaddlePaddle/PP-DocLayoutV3_safetensors

# Layer 4: application code (changes most often — at the top)
COPY handler.py /app/handler.py
```

This ordering means editing `handler.py` only invalidates Layer 4 — the ~2.4 GB model download layer is cached. Swapping layers 3 and 4 would force a model re-download on every code change, which would be extremely wasteful.

---

### 5.8 Configuration Stability

The glmocr config must be derived by copying the package default and patching specific values, never by writing a minimal YAML from scratch. The package's `config.yaml` contains deeply nested keys required by the layout model (e.g., `id2label`, `label_task_mapping`, per-class `merge_bboxes_mode`) that are not present in a minimal config. Writing a minimal YAML and expecting the application to merge it with defaults causes a `KeyError: 'threshold'` crash at startup (DEPLOYMENT_NOTES.md §5).

The correct approach, as implemented in both `handler.py` (`_write_config()`) and `start.sh`:
1. `shutil.copy(src_package_config, dest)` — start from the full default
2. Apply targeted regex substitutions for only the fields that need to change

---

## 6. What You May Not Know About This Pipeline

### 6.1 The page_loader Patch Is Load-Bearing

The `data:application/pdf;base64,...` patch in both `handler.py` and `start.sh` is not optional. Without it, every request returns `{"json_result": [], "markdown_result": ""}` — a silent, successful-looking empty result. This is because:
- `_load_source()` falls through to `_load_image()` for unknown data URI schemes
- `_load_image()` raises `ValueError("Invalid image source: ...")`
- The outer loop catches RuntimeError and logs `"Skipping source (unit 0): ..."` as a warning
- The server returns HTTP 200 with 0 regions

The symptom is indistinguishable from a valid document with no detectable content. There is no HTTP error code. You must check `len(json_result[0]) > 0` to detect this failure.

### 6.2 MTP Speculative Decoding Caveat

GLM-OCR uses Multi-Token Prediction (MTP), a speculative decoding method where a draft head predicts `n` tokens at once. When drafts are accepted, throughput increases proportionally. When rejected, the engine falls back to normal autoregressive decoding, adding minor overhead.

For OCR specifically, MTP acceptance rates are high when the model is confident (dense printed text, standard fonts) and low when the model is uncertain (handwriting, unusual symbols, low-resolution scans). The net effect depends heavily on your document corpus.

### 6.3 reading_order Is Not Free

The layout model assigns a `reading_order` index to each region. This ordering is used to assemble `markdown_result`. The order is determined by the model's understanding of document flow — left-to-right, top-to-bottom in most Western documents, but column-aware for multi-column layouts and footnote-aware for academic papers. Getting `markdown_result` to flow correctly depends on this ordering being accurate, which it usually is for standard layouts and less so for complex newspaper or academic paper formats.

### 6.4 vLLM `max_num_seqs` vs Concurrency

`--max-num-seqs 32` tells vLLM to process up to 32 sequences simultaneously in its continuous batching scheduler. A single 4-page document may generate 20–40 region crops. If 8 concurrent requests arrive simultaneously, vLLM sees 160–320 pending sequences. With `max_num_seqs: 32`, vLLM processes them in batches of 32, queuing the rest. Increasing `max_num_seqs` may improve throughput at very high concurrency but increases peak VRAM usage.

### 6.5 The `HF_HUB_OFFLINE=1` Flag

The Dockerfile sets `HF_HUB_OFFLINE=1`, which prevents HuggingFace Hub from making any network calls. This guarantees model weights are served from the baked-in image cache, eliminating the risk of a network failure or HuggingFace outage causing container startup failures. The tradeoff is that updating model weights requires rebuilding the Docker image — there is no way to hot-update weights at runtime.

### 6.6 Polygon Points vs Bounding Box

Each region has both `bbox_2d` (axis-aligned rectangle) and `polygon` (arbitrary quadrilateral or polygon). The polygon is more accurate for skewed, rotated, or curved text regions. For standard documents the bbox is sufficient. For scanned/photographed documents with perspective distortion, using the polygon for cropping gives the VLM a better-aligned image crop, potentially improving OCR accuracy.

### 6.7 native_label vs label

`label` is the normalized label used in `json_result` (e.g., `"text"`, `"title"`). `native_label` is the original class name from the layout model (e.g., `"doc_title"`, `"paragraph_title"`). They differ because glmocr normalizes some fine-grained classes into broader categories for output simplicity. If you need the full 25-class taxonomy, read `native_label`. If you need a stable label for downstream logic, read `label`.

---

## 7. Quick Reference: Known Failure Modes

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `json_result: []`, HTTP 200 | page_loader not patched for data:application/pdf | Apply `_patch_page_loader()` at startup |
| `KeyError: 'threshold'` crash | Minimal YAML missing layout config keys | Copy package default, then sed-patch |
| `max_tokens > max_model_len` crash | glmocr default 8192 > vLLM max 4096 | Set `max_tokens: 2048` in config |
| `401 Unauthorized` in vLLM logs | HF_TOKEN not passed to subprocess | Set via env var or supervisor conf |
| `ImportError: cannot import xmlrpc` | venv supervisor package conflicts with system supervisord | `deactivate` before `supervisorctl` |
| Port 5002 unreachable | Vast.ai template doesn't expose the port | Add to template or SSH tunnel |
| Silent empty response at high concurrency | Layout detection OOM, exception swallowed | Check `len(json_result[0]) > 0`; use `expandable_segments:True` |
| vLLM `spec config` error | Wrong MTP JSON syntax | Use `{"method":"mtp","num_speculative_tokens":N}` |

---

## 8. Recommended Next Steps

1. **Switch to `num_speculative_tokens: 1`** — align with official vLLM GLM-OCR recipe (DEPLOYMENT_NOTES.md §11).

2. **Implement hybrid extraction** — for searchable PDFs, replace VLM calls on text/title regions with pdfplumber. 8–15x throughput improvement on typical business documents. Full plan in `HANDOFF_HYBRID_EXTRACTOR.md`.

3. **Add response validation** — always check `len(json_result[0]) > 0` before downstream processing. Silent empty responses from layout OOM are the hardest failure to catch.

4. **Upgrade GPU for OOM elimination** — on A100 (40 GB) or A6000 (48 GB), the layout model and vLLM fit without contention. Use `--gpu-memory-utilization 0.75` to give layout 10 GB headroom.

5. **Consider HPS/Triton for production** — if you need cross-page table merging, better VRAM isolation, or layout-only mode (VLM completely off), Path B from `HANDOFF_HYBRID_EXTRACTOR.md` is the right architecture.
