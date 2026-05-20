# GLM-OCR Modal Deployment

## Apps

| App name | File | Purpose |
|---|---|---|
| `layout-worker` | `layout.py` | PP-DocLayoutV3 layout detection + PDF text extraction |
| `glm-ocr` | `glm_ocr.py` | GLM-OCR 9B via vLLM — whole-page OCR |
| `glm-ocr-pipeline` | `glm_ocr_pipeline.py` | Full pipeline: layout → crop → parallel OCR |

## Architecture

```
layout-worker
  LayoutDetector   GPU (T4/L4)  detect(page_jpegs) → raw detections
  Processor        CPU          HTTP endpoint: PDF → layout + text extraction

glm-ocr
  GLMOCRWorker     GPU (L4)     recognize(image_bytes, prompt) → str
  OCRFrontend      CPU          HTTP endpoint: PDF → whole-page OCR

glm-ocr-pipeline
  PipelineFrontend CPU          HTTP endpoint: PDF → layout → crop → parallel OCR → Markdown
    calls LayoutDetector.detect  (cross-app Modal RPC)
    calls GLMOCRWorker.recognize (cross-app Modal RPC)
```

## Key Config

### GLMOCRWorker (glm_ocr.py)
- GPU: L4 (24 GB VRAM)
- Model: `zai-org/GLM-OCR` (2.37 GB, bfloat16)
- vLLM: `--gpu-memory-utilization 0.5` → ~9.6 GB KV cache
- Speculative decoding: MTP `num_speculative_tokens=3`
- `max_containers=2`, `@modal.concurrent(max_inputs=4, target_inputs=2)`
- Effective capacity: 2 × 4 = 8 concurrent recognize calls
- vLLM batches up to 4 seqs per container internally
- Memory snapshot: sleep/wake pattern — cold start ~5-7s after first deploy
- First deploy cold start: ~5 min (model load + CUDA compile + warmup + snapshot)

### LayoutDetector (layout.py)
- GPU: T4 (default) or L4
- Model: PP-DocLayoutV3 (~100 MB)
- `max_containers=8`, `@modal.concurrent(max_inputs=4, target_inputs=3)`
- `_gpu_lock` serializes CUDA forward pass; CPU preprocessing runs concurrently

## Prompts (GLM-OCR)

| Region label | Task | Prompt |
|---|---|---|
| text, paragraph_title, content, doc_title, abstract, etc. | text | `Text Recognition:` |
| table | table | `Table Recognition:` |
| display_formula, inline_formula | formula | `Formula Recognition:` |
| image, figure, chart, figure_title, table_title, chart_title | skip | — (no OCR) |
| header, footer, number, footnote, aside_text, reference, footer_image, header_image | abandon | — (discarded) |

## Pipeline Modes (TODO — to be implemented in glm_ocr_pipeline.py)

Three modes via per-category flags on the request:

```python
use_vlm_for_text:     bool = True   # False → vanilla (PDF text layer only)
use_vlm_for_tables:   bool = True
use_vlm_for_formulas: bool = True
use_vlm_for_images:   bool = False
min_text_chars:       int  = 10     # complementary fallback threshold
```

| Mode | use_vlm_for_text | use_vlm_for_tables | use_vlm_for_formulas | note |
|---|---|---|---|---|
| **Vanilla** | False | False | False | PDF text layer only, no GPU |
| **Complementary** | False | True | True | PDF text + VLM fallback if empty + VLM for structure |
| **Comprehensive** | True | True | True | All regions → VLM |

Modelled after PaddleOCR's `use_ocr_for_image_block`, `use_chart_recognition`, `use_seal_recognition` flags.

## Timing Model (TODO — to be added to glm_ocr.py and glm_ocr_pipeline.py)

See section below.

## Endpoints

| Endpoint | URL pattern |
|---|---|
| Layout detection | `https://ringkasan-net--layout-worker-processor-process.modal.run` |
| GLM-OCR whole-page | `https://ringkasan-net--glm-ocr-ocrfrontend-process.modal.run` |
| GLM-OCR prime | `https://ringkasan-net--glm-ocr-ocrfrontend-prime.modal.run` |
| Pipeline | `https://ringkasan-net--glm-ocr-pipeline-pipelinefrontend-process.modal.run` |

## Test scripts

```bash
python modal/test_glm_ocr.py modal/pmk.pdf --repeat 3
python modal/test_glm_ocr_pipeline.py modal/pmk.pdf --save
python modal/prime_glm_ocr.py
```

## Pending

- [ ] Add timing instrumentation to GLMOCRWorker.recognize (queue time + execution time)
- [ ] Add timing instrumentation to PipelineFrontend (per-region OCR timing)
- [ ] Implement vanilla/complementary/comprehensive modes in glm_ocr_pipeline.py
- [ ] Add `min_text_chars` fallback for complementary mode (PDF text → VLM if empty)
- [ ] Add asyncio.Semaphore to PipelineFrontend to cap concurrent recognize calls (match 2×4=8 capacity)
- [ ] Re-enable + test MTP speculative decoding stability after warmup fix (max_tokens=10)
- [ ] Redeploy glm-ocr after warmup fix to rebuild snapshot (rejection_greedy_sample_kernel)

---

## Timing Instrumentation

### Problem

The Modal dashboard shows three timing fields per function call:
- **Enqueued**: when the caller invoked `.remote()` — Modal received the request
- **Started**: when the container actually began executing the function
- **Execution**: time spent inside the function body

`Started − Enqueued = queue time` (time waiting for an available container slot).

The dashboard gives these per-call, but we need them **in our own response** so clients see queue time without opening the Modal dashboard.

### How layout.py does it

The GPU function records its own start timestamp and returns it:

```python
# Inside LayoutDetector.detect (GPU container):
t0 = time.time()
...
return {"pages": raw_pages, "_detect_start_ts": t0, "detect_s": detect_s}

# Inside Processor.process (CPU caller):
t_call = time.time()                                    # ← when remote() was called
gpu_result = await LayoutDetector().detect.remote.aio(page_jpegs)
t_gpu_done = time.time()                                # ← when remote() returned

detect_start_ts = gpu_result.pop("_detect_start_ts")   # ← when GPU fn started
detect_s        = gpu_result.pop("detect_s")            # ← GPU execution time

queued_s  = detect_start_ts - t_call    # time in Modal queue
```

This works because both `t_call` (CPU container clock) and `_detect_start_ts` (GPU container clock) are Unix timestamps. Modal containers are NTP-synced so the clocks are aligned within ~1ms.

### Plan for GLMOCRWorker.recognize

`recognize` currently returns `str`. Change it to return a dict, or add a parallel timing method. The cleanest: return a `(text, start_ts)` tuple, unpack in the caller.

```python
# In GLMOCRWorker.recognize:
@modal.method()
def recognize(self, image_bytes: bytes, prompt: str = "Text Recognition:") -> dict:
    t0 = time.time()
    ...
    return {
        "text":       text,
        "_start_ts":  t0,
        "exec_s":     round(time.time() - t0, 3),
    }

# In OCRFrontend / PipelineFrontend caller:
t_call   = time.time()
result   = await GLMOCRWorker().recognize.remote.aio(image_bytes, prompt)
t_done   = time.time()

queued_s = result["_start_ts"] - t_call   # queue wait
exec_s   = result["exec_s"]               # GPU execution
wall_s   = t_done - t_call                # total round-trip
```

### Fields to surface in pipeline response

```json
"timing": {
  "render_s":   0.4,    // PDF → JPEG
  "layout_s":   0.3,    // LayoutDetector.detect (includes queue)
  "layout_queued_s": 0.05,  // time waiting for layout GPU slot
  "layout_exec_s":   0.25,  // actual GPU inference
  "crop_s":     0.02,   // region cropping
  "ocr_s":      8.5,    // asyncio.gather across all recognize calls (wall time)
  "ocr_queued_s": 1.2,  // avg queue time per recognize call
  "ocr_exec_s":   7.3,  // avg execution time per recognize call
  "total_s":    9.2
}
```
