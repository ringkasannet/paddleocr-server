# Architecture Comparison — Layout + VLM Approaches

## Systems Covered

| ID | Name | Stack |
|---|---|---|
| **A** | GLM-OCR | PP-DocLayoutV3 (HF transformers) + GLM-OCR 0.9B (vLLM) |
| **B** | PaddleOCR Simple | PP-DocLayoutV3 (PaddleX serving) + PaddleOCR-VL-1.5-0.9B (vLLM) |
| **C** | HPS/Triton | PP-DocLayoutV3 (Triton Python backend) + PaddleOCR-VL-1.5-0.9B (vLLM-server) |
| **D** | Our Proposed | PP-DocLayoutV3 (direct transformers, RunPod serverless) + GLM-OCR (native vLLM worker) |

---

## Part 1 — Layout Detection

### How each system runs the layout model

#### A — GLM-OCR (`layout_detector.py`)

Loads `PaddlePaddle/PP-DocLayoutV3_safetensors` directly via HuggingFace transformers.
No subprocess, no serving layer — model lives in the same Python process as the pipeline.

```python
# layout_detector.py:320 — chunked batch inference
for chunk_start in range(0, num_images, self.batch_size):   # batch_size=2 (config)
    chunk_pil = pil_images[chunk_start:chunk_end]

    inputs = self._image_processor(images=chunk_pil, return_tensors="pt")
    inputs = {k: v.to(self._device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = self._model(**inputs)          # single GPU forward pass

    raw_results = self._post_process_chunk_with_fallback(...)
    paddle_format_results = apply_layout_postprocess(
        raw_results, id2label, img_sizes,
        layout_nms, layout_unclip_ratio, layout_merge_bboxes_mode
    )
    torch.cuda.empty_cache()                     # free between chunks
```

**Post-processing** (`apply_layout_postprocess`):
- Soft NMS: same-class IoU threshold 0.6, cross-class 0.98
- Filters bboxes covering >82–93% of page area (page-spanning noise)
- Per-class merge strategies: `union` / `large` / `small`
- `unclip_ratio` expansion per axis
- Assigns `reading_order` index

**Batch accumulation** (`_workers.py:layout_worker`):
```python
# Accumulates pages until batch_size reached, then flushes
if len(batch_images) >= layout_detector.batch_size:
    _flush_layout_batch(...)
# Also flushes on UNIT_DONE sentinel (last pages of a document)
elif identifier == IDENTIFIER_UNIT_DONE:
    if batch_images:
        _flush_layout_batch(...)
```

**Config values (glmocr/config.yaml):**
- `batch_size: 2`
- `threshold: 0.3`
- `layout_nms: true`
- 25 label classes

---

#### B — PaddleOCR Simple (`handler.py` + `paddlex --serve`)

Runs PaddleX pipeline server as a subprocess on port 8080. The Python handler calls it
via HTTP. Layout detection is inside the PaddleX pipeline — not directly accessible.

```python
# handler.py — starts PaddleX as subprocess
_paddle_proc = subprocess.Popen([
    "/workspace/.paddleocr/bin/paddlex",
    "--serve", "--pipeline", PIPELINE_CFG,
    "--host", "0.0.0.0", "--port", str(PADDLE_PORT),
])
# Per request: POST http://localhost:8080/ocr-doc-parser
```

Layout model same (PP-DocLayoutV3) but accessed through PaddleX serving abstraction.
No direct control over batch size or post-processing parameters at request time.

---

#### C — HPS/Triton (`model_repo/layout-parsing/1/model.py`)

PP-DocLayoutV3 runs **inside** a Triton Python backend. The entire pipeline
(layout + crop + VLM) is one `self.pipeline()` call within the backend:

```python
# layout-parsing/1/model.py:run_batch()
preds = list(
    self.pipeline(
        images,
        use_layout_detection=inputs_g[0].useLayoutDetection,
        layout_threshold=inputs_g[0].layoutThreshold,
        layout_nms=inputs_g[0].layoutNms,
        layout_unclip_ratio=inputs_g[0].layoutUnclipRatio,
        layout_merge_bboxes_mode=inputs_g[0].layoutMergeBboxesMode,
    )
)
```

**Triton config** (`config.pbtxt`):
```
max_batch_size: 16
dynamic_batching { }          # no max_queue_delay_microseconds set
instance_group [{ count: 2, kind: KIND_GPU, gpus: [0] }]
```

> **⚠️ Dynamic batching is effectively disabled in the actual deployment.**
> `dynamic_batching {}` with no `max_queue_delay_microseconds` defaults to **zero**
> delay — Triton dispatches each request to a free stub immediately rather than
> waiting to accumulate a batch. `max_batch_size=16` is never reached in practice.
> Each `execute()` call receives exactly one request.

**Real parallelism = 2 stub processes.** The gateway's 16-slot inference semaphore
does not create 16 parallel executions — it creates **2 executing + 14 queued** at
Triton. Maximum throughput: `2 × (1 / 20s) = 6 requests/min`.

**Pipeline config** (`pipeline_config.yaml`):
```yaml
SubModules:
  LayoutDetection:
    model_name: PP-DocLayoutV3
    batch_size: 8             # images within a single request — not cross-request
    threshold: 0.3
    layout_nms: True
```

**3-thread pipeline overlap** (`use_queues: true`): within one stub execution, three
threads run concurrently — `thread_input` decodes pages, `thread_cv` runs layout
detection, `thread_vlm` fires VLM calls. Layout for page N+1 overlaps with VLM for
page N. This is a genuine latency win on multi-page documents.

**Pixel-key VLM batching**: crops are grouped by `(min_pixels, max_pixels)` before
submission to vLLM, so all crops with the same resolution constraints arrive together
— maximises vLLM's batching efficiency.

**Per-request parameter override**: `layoutThreshold`, `layoutNms`,
`layoutUnclipRatio` and 20+ other fields are overridable per call.

---

#### D — Our Proposed (layout endpoint)

Same model as A (PP-DocLayoutV3 via HF transformers), same direct in-process loading.
Key differences vs A:

- No vLLM sharing the GPU → `batch_size=4` safe on T4 (vs 2 on RTX 3090)
- pdfplumber runs alongside layout for text-layer PDFs
- Returns crops + pdfplumber text in one response — client decides what needs VLM
- RunPod cached model (`PaddlePaddle/PP-DocLayoutV3_safetensors`) — no baked weights

```python
# Proposed layout handler — startup (module level, runs once)
_processor = PPDocLayoutV3ImageProcessor.from_pretrained(LAYOUT_MODEL)
_model     = PPDocLayoutV3ForObjectDetection.from_pretrained(LAYOUT_MODEL).to("cuda:0")
_model.eval()

# Per request
def handler(job):
    pdf_bytes  = decode_pdf(job["input"]["images"][0])
    pages      = render_pdf(pdf_bytes, dpi=150)          # PIL Images
    layout     = detect_layout_batched(pages, batch_size=4)
    text_map   = pdfplumber_extract(pdf_bytes, layout)   # fast, CPU
    crops      = crop_and_encode(pages, layout)          # base64 PNG per region
    return build_response(layout, text_map, crops)
```

---

### Layout Detection — Side-by-Side

| Dimension | A GLM-OCR | B PaddleOCR Simple | C HPS/Triton | D Our Proposed |
|---|---|---|---|---|
| Model | PP-DocLayoutV3 (HF safetensors) | PP-DocLayoutV3 (PaddleX) | PP-DocLayoutV3 (PaddleX via Triton) | PP-DocLayoutV3 (HF safetensors) |
| Inference layer | Direct `model(**inputs)` | PaddleX HTTP serving | Triton Python backend | Direct `model(**inputs)` |
| Batch size | 2 (VRAM constrained) | Unknown (PaddleX default) | **8** (within one request) | **4** (T4, no vLLM) |
| Cross-request batching | None — per-request only | None | **Configured but disabled** (queue delay=0) | None — per-request only |
| Post-processing | `apply_layout_postprocess` in-process | PaddleX internal | PaddleX internal (same) | `apply_layout_postprocess` in-process |
| Per-request param override | No | No | **Yes** (threshold, nms, unclip per call) | No |
| VRAM used | ~4 GB (shares with vLLM) | ~4 GB (shares with vLLM) | ~4 GB (Triton owns it) | **~4 GB (T4, no sharing)** |
| GPU isolation | None — fights vLLM allocator | None | **Yes — Triton owns GPU process** | **Yes — dedicated endpoint** |
| OOM risk | **High** at concurrency >4 | Medium | **None** (Triton manages) | **None** (no vLLM competing) |
| Cold start | Part of 2–3 min combined | Part of 2–3 min combined | Part of 5 min compose-up | **~15–20s standalone** |
| Hybrid text extraction | Not present | Not present | Not present | **Yes — pdfplumber for text regions** |

---

## Part 2 — VLM Text Recognition

### How each system calls the VLM

#### A — GLM-OCR (`_workers.py:recognition_worker`)

`ThreadPoolExecutor` — each crop is one concurrent HTTP POST to vLLM's
`/v1/chat/completions`. Standard OpenAI vision format.

```python
# _workers.py:318 — recognition_worker
executor = ThreadPoolExecutor(max_workers=concurrency)  # min(max_workers, 128)
futures: Dict[Any, Dict] = {}

# For each region crop from region_queue:
req = page_loader.build_request_from_image(
    msg["cropped_image"],
    msg["region"]["task_type"],   # "text", "table", "formula", etc.
)
future = executor.submit(ocr_client.process, req)   # non-blocking submit
futures[future] = msg

# Backpressure: wait if concurrency limit reached
if len(futures) >= concurrency:
    _wait_for_any(futures)
```

**Exact vLLM request** (`ocr_client.py` + `page_loader.py:build_request_from_image`):
```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url",
       "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
      {"type": "text", "text": "<task-specific prompt>"}
    ]
  }],
  "max_tokens": 2048,
  "temperature": 0.0,
  "top_p": 0.00001,
  "top_k": 1,
  "repetition_penalty": 1.1,
  "model": "glm-ocr"
}
```

**OCRClient** uses a connection-pooled `requests.Session` (pool_size=128).
Each thread holds one connection from the pool for the duration of its request.
vLLM receives N concurrent POST requests and batches them via continuous batching
(up to `max_num_seqs=32`).

**vLLM flags (handler.py):**
```
--model zai-org/GLM-OCR
--gpu-memory-utilization 0.80   → ~18.8 GB KV cache (RTX 3090)
--max-model-len 4096
--max-num-seqs 32
--speculative-config {"method":"mtp","num_speculative_tokens":3}
--trust-remote-code
```

---

#### B — PaddleOCR Simple

PaddleX pipeline calls vLLM via its `genai_config.backend: vllm-server` setting.
Same HTTP POST pattern internally, but mediated through PaddleX abstractions.
The pipeline is a black box from the handler's perspective.

```yaml
# pipeline config
VLRecognition:
  model_name: PaddleOCR-VL-1.5-0.9B
  genai_config:
    backend: native          # uses PaddleX's built-in serving, not external vLLM
```

---

#### C — HPS/Triton

VLM is also hidden inside `self.pipeline()`. The pipeline config wires it:

```yaml
# pipeline_config_local.yaml (our local override)
VLRecognition:
  model_name: PaddleOCR-VL-1.5-0.9B
  batch_size: 4096             # effectively unlimited — vLLM controls batching
  genai_config:
    backend: vllm-server
    server_url: http://localhost:8118/v1   # separate vllm process
```

The gateway enforces concurrency with a semaphore, but this is not the actual
parallelism limit — the 2 Triton stubs are:
```python
# gateway/app.py
app.state.inference_semaphore = asyncio.Semaphore(16)
# Semaphore = 16, but real parallelism = 2 stub processes.
# 14 of those 16 slots queue at Triton waiting for a free stub.
```

**vLLM config (actual defaults from start_hps.sh):**
```yaml
gpu-memory-utilization: 0.50   # ← 50%, not ~85%
enable-prefix-caching: false
mm-processor-cache-gb: 0
```

On an A4000 (16 GB): 50% = **8 GB for KV cache**. Layout model (~200 MB) and
PaddlePaddle runtime take ~2 GB of the remaining 8 GB.

> **⚠️ vLLM runs at only 50% GPU utilisation by default.** This is extremely
> conservative — the rationale is that layout detection peaks can cause OOM spikes
> since both processes share the same physical GPU. The fix is `LAYOUT_DEVICE=cpu`
> which frees the entire 16 GB for vLLM (`GPU_MEMORY_UTILIZATION` can then be 0.70+).

**VLM concurrency**: `max_concurrency=32` per stub × 2 stubs = **64 total concurrent**
HTTP calls to vLLM (not "unlimited").

**To disable VLM entirely** (layout-only mode):
```yaml
# pipeline_config.yaml — add one line:
use_vl_recognition: False
# Frees all VRAM allocated to vLLM
```

---

#### D — Our Proposed (RunPod native vLLM worker)

No custom VLM handler. The RunPod native vLLM worker serves `zai-org/GLM-OCR`
and exposes the standard OpenAI API — exactly what A's `OCRClient` calls.

**Elixir orchestrator replicates GLM-OCR's ThreadPoolExecutor pattern:**
```elixir
# After layout endpoint returns regions
vlm_regions = Enum.filter(regions, &is_nil(&1["text"]))

results =
  vlm_regions
  |> Task.async_stream(
       fn region ->
         call_vllm_endpoint(region["crop"], region["label"])
       end,
       max_concurrency: 32,    # matches vLLM max_num_seqs
       timeout: 30_000
     )
  |> Enum.map(fn {:ok, result} -> result end)
```

Each `call_vllm_endpoint/2` is an HTTP POST to:
```
POST https://api.runpod.ai/v2/{VLM_ENDPOINT_ID}/openai/v1/chat/completions
Authorization: Bearer {RUNPOD_API_KEY}
```

Same JSON payload as A. vLLM receives 32 concurrent requests and batches them.

**vLLM config (RunPod native worker env vars):**
```
MODEL_NAME=zai-org/GLM-OCR
GPU_MEMORY_UTILIZATION=0.93      # no layout model sharing → +3 GB vs A's 0.80
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=64                  # double A's 32 — no VRAM pressure from layout
TRUST_REMOTE_CODE=true
NUM_SPECULATIVE_TOKENS=3
SPECULATIVE_MODEL=zai-org/GLM-OCR  # MTP
```

---

### VLM — Side-by-Side

| Dimension | A GLM-OCR | B PaddleOCR Simple | C HPS/Triton | D Our Proposed |
|---|---|---|---|---|
| Model | GLM-OCR 0.9B | PaddleOCR-VL-1.5-0.9B | PaddleOCR-VL-1.5-0.9B | GLM-OCR 0.9B |
| Inference backend | vLLM subprocess | PaddleX native genai | vLLM-server subprocess | RunPod native vLLM worker |
| Request format | OpenAI `/v1/chat/completions` | PaddleX internal | OpenAI `/v1/chat/completions` | OpenAI `/v1/chat/completions` |
| Concurrency model | ThreadPoolExecutor (128 threads) | PaddleX internal | asyncio sem (32/stub × 2 stubs = 64) | Elixir `Task.async_stream` (32) |
| GPU util % | 80% (layout shares GPU) | ~85% | **50%** default (layout on same GPU) | **93%** (dedicated GPU) |
| KV cache (A4000 16GB at 50%) | — | — | **~8 GB** | **~21.9 GB** (RTX 3090 at 93%) |
| max_num_seqs | 32 | unknown | vLLM default (~256) | **64** |
| Speculative decoding | MTP 3 tokens | None documented | None documented | MTP 3 tokens |
| VLM-off mode | Manual (remove from handler) | Manual | **One config line** | Not start VLM endpoint |
| VLM cold start | 60–90s (part of full 2–3 min) | 60–90s (part of full 2–3 min) | 300s start_period (compose) | **60–90s independent** |
| Custom handler code | Yes — glmocr server manages | Yes — handler.py | **No** — Triton manages | **No** — native worker |

---

## Part 3 — Optimization Approaches

### A — GLM-OCR Optimizations

| Optimization | Implementation | Effect |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | glmocr process env | Reduces CUDA allocator fragmentation — fewer OOM retries on layout batches |
| MTP speculative decoding | `{"method":"mtp","num_speculative_tokens":3}` | Parallelizes draft token generation — ~20–30% throughput gain on dense text |
| `max-model-len 4096` | vLLM flag | Prevents 8192 default from doubling KV cache allocation |
| Soft NMS in postprocess | `iou_same=0.6, iou_diff=0.98` | Keeps more overlapping boxes vs hard NMS — better recall on dense layouts |
| `--max-num-seqs 32` | vLLM flag | Limits concurrent sequences — prevents VRAM spike from too many parallel crops |
| `batch_size: 2` | glmocr config | Conservative — safe given vLLM occupies ~19 GB |
| ThreadPoolExecutor backpressure | `_workers.py:338` | Prevents unbounded queue buildup when vLLM is saturated |
| Per-chunk `torch.cuda.empty_cache()` | `layout_detector.py:359` | Frees GPU memory between layout batches — reduces OOM pressure |

**Known limitations:**
- Layout model and vLLM fight the same CUDA allocator → OOM retries at concurrency >4
- 80% GPU util is the empirical sweet spot: lower reduces KV cache too much, higher triggers OOM cascade
- Benchmark shows 83% util achieves 1.77 pages/s vs 1.44 at 80% — not currently set

---

### B — PaddleOCR Simple Optimizations

Limited — the pipeline is a black box behind PaddleX serving. Key config:
- Isolated Python venv (`/workspace/.paddleocr`) separates PaddleX deps from vLLM
- `GPU_MEMORY_UTILIZATION=0.85` (slightly higher than GLM-OCR default)
- `MODEL_NAME`, `VLLM_PORT`, `PADDLE_PORT` configurable via env vars
- No speculative decoding

---

### C — HPS/Triton Optimizations

| Optimization | Implementation | Effect |
|---|---|---|
| Triton dynamic batching | `dynamic_batching {}` in config.pbtxt | Accumulates requests from multiple clients before GPU forward pass — better utilization |
| Separate process isolation | Triton owns layout GPU memory; vLLM is separate process | Eliminates CUDA allocator contention — no OOM retries |
| `batch_size: 8` for layout | `pipeline_config.yaml` | 4× larger batches vs GLM-OCR — amortizes fixed GPU launch overhead |
| `batch_size: 4096` for VLM | `pipeline_config_local.yaml` | Effectively unlimited — vLLM handles its own batching |
| async gRPC client | `tritonclient.grpc.aio` | Gateway never blocks on Triton calls — `await triton_request_async()` |
| Two semaphores | inference=16, non-inference=64 | Separates GPU-bound from CPU-bound work — CPU restructure never starves GPU |
| `uvicorn --workers 4` | gateway.Dockerfile CMD | 4 uvicorn processes handle concurrent HTTP — ASGI throughput |
| Cross-page restructure-pages | CPU Triton model | Table continuation / title hierarchy fixed post-hoc — no VLM re-run |
| Per-request parameter override | `layoutThreshold`, `layoutNms` etc. in request body | Callers tune detection sensitivity without redeploying |
| `use_vl_recognition: False` | pipeline_config_local.yaml | Disables VLM entirely — layout-only mode in one line |

**Key advantage over A/B**: Triton's dynamic batching means a burst of 8 concurrent
requests gets batched into one layout model forward pass. In A/B, 8 concurrent
requests = 8 serial layout forward passes (layout worker is single-threaded).

---

### D — Our Proposed Optimizations

| Optimization | Implementation | Effect |
|---|---|---|
| **Dedicated GPU per endpoint** | Layout on T4; VLM on RTX 3090 | No VRAM contention — layout gets full 16 GB, VLM gets full 24 GB |
| **pdfplumber hybrid** | Layout endpoint extracts text-layer content | 70–80% fewer VLM calls for searchable PDFs → proportional cost and latency reduction |
| `GPU_MEMORY_UTILIZATION=0.93` | VLM endpoint env | +3.1 GB KV cache vs GLM-OCR's 0.80 — fewer sequence evictions at high concurrency |
| `max_num_seqs=64` | VLM endpoint env | Double GLM-OCR's 32 — safe without layout model competing |
| RunPod cached model | `HF_HOME=/runpod-volume` | No model download on cold start — weights on local NVMe |
| `HF_HUB_OFFLINE=1` | VLM Dockerfile ENV | Prevents accidental HF network calls at runtime |
| `batch_size=4` for layout | Layout handler config | Safe on T4 — 4× GLM-OCR without vLLM competing |
| Layout cold start ~20s | No vLLM in layout container | Layout accepts requests 5–6× faster than current monolith |
| Keep 1 active VLM worker | RunPod endpoint config | Eliminates VLM cold start for steady traffic — pays for itself at >50 docs/day |
| Elixir `Task.async_stream(max_concurrency: 32)` | Orchestrator | Replicates GLM-OCR's ThreadPoolExecutor — same vLLM batching behavior |
| PNG encoding for crops | Layout handler | Lossless — preserves fine text detail for VLM (vs JPEG artifacts) |
| `concurrencyBatch=32` on VLM endpoint | RunPod config | All concurrent crop requests land on same worker → vLLM continuous batching |

---

## Part 4 — Batching Strategy

Batching exists at three independent levels. Each system satisfies them differently.

| Level | Definition |
|---|---|
| **L1 — Layout intra-request** | Pages from ONE document processed together in one GPU forward pass |
| **L2 — VLM intra-request** | Crops from ONE document submitted to vLLM simultaneously (one job) |
| **L3 — Layout cross-request** | Pages from MULTIPLE concurrent clients batched into a single GPU forward pass |

(VLM cross-request batching happens automatically via vLLM's continuous batching scheduler for all four systems — it is not configurable by the caller.)

---

### L1 — Layout intra-request batching

All four systems slice pages into chunks and run the layout model on each chunk:

| System | Batch size | Chunk loop | VRAM per batch |
|---|---|---|---|
| A GLM-OCR | **2** | `for chunk_start in range(0, N, 2)` | ~2 × 276 MB |
| B PaddleOCR Simple | Unknown (PaddleX default) | Hidden inside `self.pipeline()` | ~? |
| C HPS/Triton | **8** | `self.pipeline(all_images)` inside Triton backend | ~8 × 276 MB |
| D Our Proposed | **4** | `PPDocLayoutDetector.process(pages_pil)` | ~4 × 276 MB |

A uses `batch_size=2` because vLLM occupies ~19 GB of the same RTX 3090.
D uses `batch_size=4` because the layout endpoint has the entire T4 (16 GB) to itself.
C uses `batch_size=8` because Triton manages GPU memory and layout is the only model loaded.

```python
# A — GLM-OCR layout_detector.py:320
for chunk_start in range(0, num_images, self.batch_size):   # batch_size=2
    inputs = self._image_processor(images=chunk_pil, return_tensors="pt")
    outputs = self._model(**inputs)
    torch.cuda.empty_cache()    # free between chunks

# D — Our handler.py (via PPDocLayoutDetector internally, batch_size=4)
all_results, _ = _detector.process(pages_pil)   # batches internally
```

---

### L2 — VLM intra-request batching

How crops from a single document reach vLLM:

| System | Mechanism | Concurrency |
|---|---|---|
| A GLM-OCR | `ThreadPoolExecutor` — one thread per crop | up to `max_workers=128` |
| B PaddleOCR Simple | PaddleX internal (unknown) | Unknown |
| C HPS/Triton | `batch_size: 4096` — PaddleX sends all crops together | Effectively unlimited |
| **D Our Proposed** | `asyncio.gather` — all crops submitted simultaneously | All N crops of the document |

```python
# A — _workers.py:318  (one submit per crop, backpressure at max_workers)
future = executor.submit(ocr_client.process, req)

# D — vlm/handler.py  (all crops from one job gathered concurrently)
async with aiohttp.ClientSession() as session:
    tasks = [_infer_one(session, r["region_id"], r["crop"], r["label"]) for r in regions]
    results = await asyncio.gather(*tasks)
```

Both A and D achieve the same vLLM behavior: N concurrent HTTP requests arrive at
vLLM's scheduler, which batches them into one or more GPU forward passes based on
`max_num_seqs`. Our `max_num_seqs=64` vs A's `32` — we can hold twice as many
concurrent sequences in the KV cache.

A typical 30-page PDF has ~100–150 regions; after pdfplumber filters text regions,
~20–30 non-null crops remain. All 20–30 crops are submitted concurrently — vLLM
batches all of them.

---

### L3 — Layout cross-request batching

This is the key structural difference between HPS/Triton and all other systems.

**C (HPS/Triton)** — configured but **disabled in practice**:
```
# layout-parsing/config.pbtxt
max_batch_size: 16
dynamic_batching { }        ← no max_queue_delay_microseconds → defaults to 0
instance_group [{ count: 2, kind: KIND_GPU }]
```

`dynamic_batching {}` with zero queue delay means Triton dispatches each request to
a free stub **immediately** rather than waiting to accumulate a batch. In practice,
each `execute()` receives exactly one request. `max_batch_size=16` is never reached.

To actually enable cross-request batching in HPS, `config.pbtxt` would need:
```protobuf
dynamic_batching {
    preferred_batch_size: [ 2, 4 ]
    max_queue_delay_microseconds: 100000
}
```

**A, B, D** — NO cross-request batching:

| System | Why not |
|---|---|
| A GLM-OCR | `layout_worker` is a single thread; processes one document's batch at a time |
| B PaddleOCR Simple | One PaddleX pipeline instance; no parallel document processing |
| **D Our Proposed** | RunPod serverless spawns one worker per job; each handler call is independent |

**Impact for our system (D):**

If 8 documents arrive simultaneously at the layout endpoint, RunPod spawns up to 8
workers. Each worker independently runs `batch_size=4` chunks through the layout model.
There is no coordination between workers — pages from different documents never share
a GPU forward pass.

This is a **throughput-via-horizontal-scaling** strategy vs Triton's
**GPU-efficiency-via-batching** strategy.

```
D — 8 concurrent documents (8 workers, no cross-batching):
  Worker 1: [page1, page2, page3, page4] → forward pass
  Worker 2: [page1, page2, page3, page4] → forward pass  (parallel, separate GPUs)
  ...

C — 8 concurrent documents (1 Triton instance, cross-batching):
  Triton:   [doc1/p1, doc1/p2, doc2/p1, doc2/p2, doc3/p1, doc3/p2, doc4/p1, doc4/p2] → 1 forward pass
```

**For most serverless workloads**, horizontal scaling is simpler and the T4 is
dedicated per worker, so there is no GPU sharing penalty. Cross-request batching only
pays off when GPU utilization is already high and you want fewer, more saturated
workers (the HPS/Triton design goal).

---

### Batching Summary

| Level | A GLM-OCR | B PaddleOCR | C HPS/Triton | D Our Proposed |
|---|---|---|---|---|
| **L1 layout intra-request** | batch_size=2 | Unknown | **batch_size=8** | batch_size=4 |
| **L2 VLM intra-request** | ThreadPoolExecutor ≤128 | Unknown | batch_size=4096, pixel-key grouped | **asyncio.gather (all crops)** |
| **L3 layout cross-request** | No | No | **Configured, disabled** (queue delay=0) | No |
| VLM cross-request | vLLM continuous (auto) | vLLM continuous (auto) | vLLM continuous (auto) | vLLM continuous (auto) |
| Scaling strategy | Vertical (one container) | Vertical | Fixed 2 stubs (6 req/min ceiling) | **Horizontal (RunPod workers)** |
| GPU contention | High (shared RTX 3090) | Medium | Medium (layout+vLLM same 16GB GPU) | **None (separate GPUs)** |

**Bottom line**: L3 cross-request batching is the theoretically correct advantage of
the Triton approach — but in the actual HPS deployment it is not enabled. Both our
system and HPS batch within a single request only (L1 + L2). HPS's real throughput
ceiling is 6 requests/min from 2 stubs; we scale by adding workers.

---

## Part 5 — Full Comparison Table

| Dimension | A GLM-OCR | B PaddleOCR Simple | C HPS/Triton | D Our Proposed |
|---|---|---|---|---|
| **Architecture** | Monolithic (one container) | Monolithic (one container) | 3-container compose | 2 serverless endpoints |
| **Cold start** | 2–3 min | 2–3 min | **5–10 min** (1st boot: CUDA compile) | Layout **20s** / VLM 60–90s |
| **Idle cost** | $0 (serverless) | $0 (serverless) | Full GPU rate 24/7 | $0 (serverless) |
| **GPU for layout** | RTX 3090 (shared) | RTX 3090 (shared) | A4000 (shared w/ vLLM) | **T4 (dedicated, no sharing)** |
| **GPU for VLM** | RTX 3090 (80% util) | ~85% | A4000 (**50% util default**) | RTX 3090 **(93% util)** |
| **KV cache** | ~18.8 GB | ~20 GB | **~8 GB** (50% of 16 GB) | **~21.9 GB** (93% of 24 GB) |
| **VRAM contention** | High — OOM at concurrency >4 | Medium | Medium (layout+vLLM same GPU) | **None** |
| **Max pages/request** | No hard limit | No hard limit | **10** (hard-coded) | No hard limit |
| **Layout batch size** | 2 | Unknown | 8 (within request) | 4 (within request) |
| **Cross-request layout batching** | No | No | **Configured, disabled in practice** | No |
| **VLM max_num_seqs** | 32 | Unknown | vLLM default | **64** |
| **Speculative decoding** | MTP 3 | No | No | **MTP 3** |
| **Hybrid text extraction** | No | No | No | **Yes (pdfplumber)** |
| **VLM-off mode** | Manual | Manual | **One config line** | Don't call VLM endpoint |
| **Cross-page post-processing** | No | No | **Yes (restructure-pages)** | No |
| **Custom handler code** | Yes (full pipeline) | Yes (full pipeline) | Gateway only (38 lines) | **Layout only** |
| **Per-request param override** | No | No | **Yes** | No |
| **Orchestration** | Internal (Python threads) | Internal (PaddleX) | Internal (Triton) | **Elixir Task.async_stream** |
| **Operational complexity** | Low | Low | High (3 services) | Medium (2 endpoints + orchestrator) |
| **Scale-to-zero** | Yes | Yes | No | Yes |

---

## Part 6 — Data Flow Comparison

### A — GLM-OCR (current monolith)
```
POST /glmocr/parse  {images: [pdf_b64]}
  │
  ├── page_loader  → PIL images (150 DPI)
  ├── layout_worker (Thread 2)
  │     batch_size=2 chunks → PP-DocLayoutV3 → apply_layout_postprocess
  │     crop_image_region per bbox → region_queue
  │
  ├── recognition_worker (Thread 3, ThreadPoolExecutor)
  │     per crop: build_request_from_image → HTTP POST /v1/chat/completions
  │     up to 128 concurrent threads → vLLM continuous batching
  │
  └── JSON assembler → {json_result, markdown_result}
```

### C — HPS/Triton
```
POST /layout-parsing  {file, fileType, ...}
  │
  ├── gateway (FastAPI) — async semaphore (16 slots)
  ├── triton_request_async → Triton gRPC :8001
  │
  └── Triton Python backend (layout-parsing/1/model.py)
        _group_inputs → ThreadPoolExecutor per group
        _preprocess: file_bytes → PIL images
        self.pipeline(images, ...) ← PaddleX handles everything:
          PP-DocLayoutV3 (batch_size=8) → crops → VLM HTTP to :8118
        _postprocess: prune_result, markdown, visualize
        → InferResult {layoutParsingResults, dataInfo}
```

### D — Our Proposed
```
Client (Elixir)
  │
  ├─① POST layout-endpoint
  │     PDF → render → PP-DocLayoutV3 (batch_size=4) → apply_layout_postprocess
  │     pdfplumber text for TEXT_LABELS regions
  │     crop+encode each region
  │     → {pages: [{regions: [{bbox, label, crop, text|null}]}]}
  │
  ├── Filter: regions where text == null
  │
  ├─② Task.async_stream (max_concurrency: 32)
  │     per crop: POST RunPod native vLLM /openai/v1/chat/completions
  │     vLLM batches concurrent crop requests internally
  │     → {region_id, content} per crop
  │
  └── Assemble: merge layout + pdfplumber + VLM text → final JSON
      (skip ② entirely for fully searchable PDFs)
```

---

## Part 7 — When to Use Each

| Use Case | Recommended |
|---|---|
| Cross-page table merging, title hierarchy | **C (HPS/Triton)** — `restructure-pages` is the only system with this |
| Searchable PDFs, cost-sensitive SaaS | **D (Our Proposed)** — pdfplumber skips 80% of VLM calls |
| Serverless, scale-to-zero, low ops | **D (Our Proposed)** — HPS requires 24/7 GPU, 5–10 min cold start |
| High GPU utilisation, max KV cache | **D (Our Proposed)** — 93% util, 22 GB KV cache vs HPS's 50%, 8 GB |
| Documents > 10 pages in one request | **D (Our Proposed)** — HPS has a hard 10-page cap |
| Development / quick start | **A (GLM-OCR)** — single container, no infrastructure |
| Layout-only, no VLM | **C with `use_vl_recognition: False`** or **D without calling VLM endpoint** |
| Elixir/Phoenix backend | **D** — native HTTP from Elixir Tasks, no Python layer |
| Per-request threshold/NMS tuning | **C (HPS/Triton)** — 20+ overridable params per call; D has fixed config |
