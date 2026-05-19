# Modal Layout Worker — Architecture & Design

PP-DocLayoutV3 document layout detection deployed on Modal serverless GPU infrastructure.

---

## Files

| File | Purpose |
|---|---|
| `modal/layout.py` | Full deployment: image, GPU class, HTTP endpoint, helpers |
| `modal/test_layout.py` | CLI test tool: sends files, prints timing + cost breakdown |

---

## Architecture: Two Containers Per Request

Every request runs across two Modal containers with different hardware:

```
Client
  │  POST {"file": base64, "fileType": 0|1, "dpi": 150}
  ▼
process()  ─── CPU container (scaledown_window=1000s)
  │  1. base64 decode
  │  2. PDF render → PIL images  (pypdfium2, CPU)
  │  3. JPEG-encode pages for transfer
  │  4. detect.remote(page_jpegs)  ──────────────────►  LayoutDetector.detect()  GPU container (T4/L4)
  │                                                         5. JPEG decode → PIL
  │                                                         6. processor preprocess → tensors
  │                                                         7. model inference  ◄── only GPU step
  │                                                         8. post_process detections
  │                                ◄──────────────────  returns raw detections + timestamps
  │  9. NMS (non-max suppression)
  │  10. reading order sort
  │  11. text extraction from PDF  (pypdfium2, CPU)
  │  12. assemble response + cost
  ▼
Client  ←  {"pages": [...], "meta": {"timing": {...}, "cost": {...}}}
```

### Why two containers?

`detect()` only runs what requires the GPU: tensor preprocessing, model forward pass, and decoding the output tensors back to Python values. Everything else — PDF rendering, searchable-page detection, NMS, reading order, and text extraction — is pure CPU work and runs in `process()` at no GPU cost.

Before this split the GPU container handled the entire pipeline (~10s billed GPU time). After the split, GPU billing is only for model inference (~5s).

---

## Key Components

### `layout_image` (shared container image)

```
nvidia/cuda:12.4.1-runtime-ubuntu22.04  ← runtime not devel (3-4 GB smaller)
  + Python 3.11
  + transformers, torch, torchvision, opencv-python-headless
  + pypdfium2, Pillow
  + huggingface_hub[hf_transfer], fastapi
```

Both `process()` and `LayoutDetector` use this image. `runtime` instead of `devel` was chosen because PyTorch bundles its own CUDA runtime — the compiler toolchain is not needed at inference time.

### `LayoutDetector` (GPU class)

Decorated with `@app.cls(gpu=GPU, ...)`. GPU type is set via `MODAL_GPU` env var at deploy time (default `T4`, also supports `L4`).

```bash
modal deploy modal/layout.py          # T4
MODAL_GPU=L4 modal deploy modal/layout.py  # L4
```

**Initialization (`@modal.enter(snap=True)`):**

Runs once and is captured in a memory + GPU snapshot:

1. Load `AutoImageProcessor` and `AutoModelForObjectDetection` from the Modal Volume (or HuggingFace if the volume is empty)
2. Move model to GPU
3. Run a **warm-up forward pass** on a dummy 224×224 image — this forces CUDA kernel compilation, which gets saved in the GPU snapshot

On subsequent cold starts the snapshot is restored, skipping model loading and kernel compilation entirely.

**`detect(page_jpegs: list[bytes]) → dict`:**

Receives JPEG-encoded page images (already rendered by `process()`). For each page:
- Decode JPEG → PIL Image
- `self._processor(...)` — resize/normalize to tensor (CPU within GPU container)
- `.to(self._device)` — move to GPU
- `self._model(**inputs)` — **GPU inference**
- `post_process_object_detection(...)` — decode model output to bboxes/scores/labels

Returns raw per-page detections plus `_detect_start_ts` (Unix timestamp at function entry, used to compute dispatch latency) and `detect_s` (pure inference duration).

### `process()` (CPU endpoint)

`@app.function` + `@modal.fastapi_endpoint`. Long `scaledown_window=1000s` keeps this container warm cheaply (CPU, not GPU). Handles the full pipeline except inference:

| Step | What | Why here |
|---|---|---|
| Base64 decode | `base64.b64decode` | CPU |
| PDF render | `pypdfium2.PdfDocument` + `page.render()` | CPU-only library |
| Searchable check | `_searchable_pages()` — counts characters per page | CPU, needs PDF object |
| JPEG encode | `PIL Image → BytesIO` at quality=90 | Compact serialization for GPU transfer |
| `detect.remote()` | Dispatch to GPU container | GPU needed |
| NMS | `_nms_regions()` — removes overlapping boxes (IoU > 0.65) | CPU, pure Python/numpy |
| Reading order | `_reading_order()` → `_xycut_segment()` — XY-cut algorithm | CPU, pure Python/numpy |
| Text extraction | `_extract_text()` — bounded text from pypdfium2 textpage | CPU, needs PDF object |
| Response assembly | Build final JSON with pages, timing, cost | CPU |

### Helper functions (module-level)

All available to both containers since they share the same image:

| Function | What it does |
|---|---|
| `_searchable_pages(pdf)` | Returns set of page indices with >10 extractable characters |
| `_extract_text(pdf, page_num, bbox_px, dpi)` | Extracts text within a bbox using pypdfium2 bounded text; applies a camelCase→hyphen fix |
| `_nms_regions(regions)` | Non-maximum suppression: keeps highest-score region when overlap >65% of smaller box |
| `_reading_order(regions, page_width)` | Top-level: separates column regions from full-width; applies XY-cut to column segments |
| `_xycut_segment(regions, w)` | Recursive XY-cut with fallback to `_sorted_layout_boxes` |
| `_sorted_layout_boxes(regions, w)` | Fallback: sorts left-column / right-column / full-width regions by y position |
| `_recursive_xy_cut(boxes, indices, res)` | Core XY-cut: projects boxes onto axes, finds gaps, splits recursively |
| `_projection(boxes, axis)` | 1D occupancy array along an axis |
| `_segments(arr, min_gap)` | Finds contiguous occupied segments with gaps > min_gap |

---

## Request Timing

The response includes a full timing breakdown under `meta.timing`:

| Field | Measured in | What it covers |
|---|---|---|
| `render_s` | `process()` | PDF decode + page rendering |
| `queued_s` | `process()` | Time from calling `detect.remote()` to `detect()` starting — dispatch + cold start |
| `detect_s` | `detect()` | Actual GPU inference time |
| `text_s` | `process()` | NMS + reading order + text extraction |
| `execution_s` | — | Alias for `detect_s` — the GPU-billed duration |
| `wall_s` | `process()` | Total server time (render + queue + detect + text) |

The test additionally reports:
- `encode_s` — client-side base64 encoding time
- `wall (client)` — total round-trip including network

`queued_s` is the most informative field: near-zero means the GPU container was warm; 10-20s means it cold-started from a snapshot.

---

## Cost Model

GPU billing is only for `LayoutDetector` (the GPU container). Modal charges for: snapshot restore time + inference time + scaledown idle window. Three estimates are provided:

| Estimate | Formula | When accurate |
|---|---|---|
| `lower` | `detect_s + idle(5s)` | Warm container, no queue |
| `queued` ★ | `queued_s + detect_s + idle(5s)` | Normal use — queued_s ≈ snapshot restore |
| `wall` | `wall_s + idle(5s)` | Upper bound; includes RPC overhead |

Rates (set via `MODAL_GPU`):

| GPU | Rate/sec | Rate/hour |
|---|---|---|
| T4 (default) | $0.000164 | ~$0.59 |
| L4 | $0.000222 | ~$0.80 |

Cost is reported in USD and IDR (fixed rate: 17,500 IDR/USD) per request and per page.

---

## Cold Start Optimizations

1. **GPU memory snapshot** — `enable_memory_snapshot=True` + `experimental_options={"enable_gpu_snapshot": True}`: captures full CPU + GPU memory state after `load()`. Restores in seconds instead of loading weights from scratch.

2. **Warm-up forward pass in `load()`** — runs dummy inference before snapshot is taken, so CUDA kernel compilation is captured and not repeated on each cold start.

3. **`runtime` CUDA image** — `nvidia/cuda:12.4.1-runtime-ubuntu22.04` instead of `-devel`: ~3-4 GB smaller, faster container boot.

4. **CPU/GPU work split** — `process()` has `scaledown_window=1000s` (kept warm cheaply). `LayoutDetector` has `scaledown_window=20s` (scales down fast to save cost). GPU billing is only for the ~5s model inference.

5. **SafeTensors format** — model is loaded from `PaddlePaddle/PP-DocLayoutV3_safetensors`, which uses memory-mapped loading rather than pickle.

6. **Weights in Modal Volume** — `download_weights` (run once) stores the model at `/weights/PP-DocLayoutV3`. Subsequent starts load from the volume, not HuggingFace.

---

## Operations

**First-time setup:**
```bash
modal run modal/layout.py::download_weights
```

**Deploy:**
```bash
modal deploy modal/layout.py                    # T4 GPU
MODAL_GPU=L4 modal deploy modal/layout.py       # L4 GPU
```

**Test:**
```bash
python modal/test_layout.py document.pdf
python modal/test_layout.py document.pdf --repeat 3          # sequential
python modal/test_layout.py document.pdf --concurrent 4      # parallel
python modal/test_layout.py document.pdf --repeat 2 --concurrent 3
python modal/test_layout.py document.pdf --no-regions        # summary only
python modal/test_layout.py document.pdf --save              # save JSON
```

**Logs (live):**
```bash
modal app logs layout-worker
```

**Tune detection threshold** (default 0.3, lower = more detections):
```bash
# set DETECT_THRESHOLD env var on the LayoutDetector class in layout.py
```
