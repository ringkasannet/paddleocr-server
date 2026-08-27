# GLM-OCR Modal Deployment

## Apps

| App name | File | Purpose |
|---|---|---|
| `glm-ocr-single` | `glm_ocr_single.py` | **Primary** — official glmocr SDK, PP-DocLayoutV3 + vLLM in one container |
| `layout-worker` | `layout.py` | PP-DocLayoutV3 layout detection + PDF text extraction |
| `glm-ocr` | `glm_ocr.py` | GLM-OCR 9B via vLLM — whole-page OCR (no layout) |
| `glm-ocr-pipeline` | `glm_ocr_pipeline.py` | Hand-rolled 3-app pipeline: layout-worker → glm-ocr |

`glm-ocr-single` is the production app. The others remain for comparison / fallback.

---

## glm-ocr-single (primary)

### Architecture

```
DocumentOCRWorker   GPU (L4)
  start()  [snap=True]   load PP-DocLayoutV3 (CPU) → start vLLM → warmup → move layout to GPU → sleep → snapshot
  wake()   [snap=False]  wake vLLM → warmup layout → prime MM cache → build Pipeline
  process()              render PDF @ 300 DPI → glmocr pipeline → blocks + markdown
  fullpage()             render PDF @ 300 DPI → full-page vLLM (no layout) — benchmark only
```

### Key Config

- GPU: L4 (24 GB VRAM)
- Model: `zai-org/GLM-OCR` (9B, bfloat16 on L4 cc=8.9)
- vLLM flags: `--gpu-memory-utilization 0.8`, `--max-model-len 8192`, `--max-num-seqs 32`, `--max-num-batched-tokens 32768`
- Speculative decoding: MTP `num_speculative_tokens=3`
- `max_containers=10`, `@modal.concurrent(max_inputs=2, target_inputs=2)`
- GPU snapshot: `enable_memory_snapshot=True` + `enable_gpu_snapshot: True`
- Cold start from snapshot: ~0.9s; first deploy (full warmup + snapshot): ~10–15 min
- Default DPI: **300** (see DPI section below)

### Endpoints

| Endpoint | URL |
|---|---|
| process (layout + OCR) | `https://ringkasan-net--glm-ocr-single-documentocrworker-process.modal.run` |
| fullpage (whole-page OCR) | `https://ringkasan-net--glm-ocr-single-documentocrworker-fullpage.modal.run` |

### Request schema

```json
{ "file": "<base64 PDF>", "pages": [0, 1, 2], "dpi": 300 }
```

`pages` and `dpi` are optional. `pages` defaults to all pages; `dpi` defaults to 300.

### Response schema

```json
{
  "markdown": "...",
  "blocks": [
    { "page": 0, "order": 0, "label": "paragraph_title", "text": "## Pasal 11", "bbox": [x0,y0,x1,y1] }
  ],
  "meta": {
    "pages": [0],
    "pages_info": [{ "page": 0, "width_px": 2480, "height_px": 3509 }],
    "total_regions": 25,
    "ocr_regions": 25,
    "skip_regions": 0,
    "timing": {
      "render_s": 0.15, "ocr_wall_s": 2.1, "assemble_s": 0.0, "total_s": 2.3,
      "cold_start": { "wakeup_s": 0.51, "health_s": 0.0, "layout_gpu_s": 0.28,
                      "batch_warmup_s": 0.06, "total_s": 0.86 }
    }
  }
}
```

---

## DPI and PP-DocLayoutV3 Detection Threshold

### Finding

PP-DocLayoutV3 has a minimum pixel-height threshold below which small headings (e.g. "Pasal 11") are not detected, regardless of score threshold or input_size tuning.

| DPI | Page size (A4) | "Pasal 11" height | Detected? |
|---|---|---|---|
| 200 | 1653 × 2339 px | ~8–9 px | ✗ |
| 250 | 2067 × 2924 px | ~10–11 px | ✗ |
| 280 | 2315 × 3275 px | ~12–13 px | ✓ |
| 300 | 2480 × 3509 px | ~13 px | ✓ |

**Detection threshold**: heading must be ≥ ~12 px tall in the rendered image. The breakpoint is between 250 DPI (fails) and 280 DPI (works). We use **300 DPI** for a comfortable safety margin.

### Why raising score threshold doesn't help

PP-DocLayoutV3 outputs score=0.0 for "Pasal 11" at 200 DPI — not a low-confidence detection but a zero, meaning the detector never fired. No threshold adjustment fixes a detector that simply did not fire. The fix is resolution.

### ZAI API internal rendering

The ZAI hosted API (`api.z.ai/api/paas/v4/layout_parsing`) renders PDFs internally at approximately 240 DPI, which is below the detection threshold. Sending a raw PDF to ZAI will also miss small headings.

---

## Benchmark: Modal vs ZAI (page32.pdf, 3 concurrent requests)

Tested 2026-06-27 on a 1-page Indonesian tax regulation PDF containing "Pasal 11" as a small centered heading.

| Approach | Detects | Wall avg | Wall min–max | Notes |
|---|---|---|---|---|
| Modal 200 DPI raw PDF | ✗ 0/3 | 17.6s | 16.7–18.6s | Cold start dominated |
| **Modal 300 DPI raw PDF** | **✓ 3/3** | **16.3s** | **16.0–16.8s** | Cold start; server_s ≈ 2s warm |
| ZAI raw PDF | ✗ 0/3 | 9.4s | 4.4–19.2s | Highly variable; ZAI renders at ~240 DPI |
| ZAI 300 DPI JPEG | ✓ 3/3 | 6.3s | 5.4–7.3s | 1.2s local convert + 5.1s ZAI API |

**Warm Modal 300 DPI**: single request wall time ≈ 3.8s (server_s ≈ 1.6–2s). The 16s in the table reflects 12 simultaneous requests competing for container slots.

**Conclusion**: Modal 300 DPI is the recommended approach. ZAI 300 DPI JPEG is a valid alternative when using the ZAI hosted API — convert locally with PyMuPDF at 300 DPI before sending.

---

## Test Scripts

```bash
# Primary endpoint
python modal/test_glm_ocr_single.py document.pdf
python modal/test_glm_ocr_single.py document.pdf --dpi 300 --save
python modal/test_glm_ocr_single.py document.pdf --app single-fullpage   # whole-page benchmark

# DPI + ZAI benchmark
python modal/test_dpi_benchmark.py page32.pdf
python modal/test_dpi_benchmark.py page32.pdf --concurrent 3
python modal/test_dpi_benchmark.py page32.pdf --concurrent 3 --repeat 2

# Legacy apps
python modal/test_glm_ocr.py modal/pmk.pdf --repeat 3
python modal/test_glm_ocr_pipeline.py modal/pmk.pdf --save
```

---

## Older Apps (glm-ocr, glm-ocr-pipeline, layout-worker)

### GLMOCRWorker (glm_ocr.py)
- GPU: L4, util=0.6, seqs=16, batched-tokens=8192, MTP num_speculative_tokens=3
- `max_containers=2`, `@modal.concurrent(max_inputs=16, target_inputs=8)`
- Snapshot working; rebuild required after any vLLM config change (`/prime`)

### LayoutDetector (layout.py)
- GPU: T4 (default) or L4, PP-DocLayoutV3 (~100 MB)
- `max_containers=8`, `@modal.concurrent(max_inputs=4, target_inputs=3)`
- `_gpu_lock` serializes CUDA forward passes

### Older endpoints

| Endpoint | URL |
|---|---|
| Layout detection | `https://ringkasan-net--layout-worker-processor-process.modal.run` |
| GLM-OCR whole-page | `https://ringkasan-net--glm-ocr-ocrfrontend-process.modal.run` |
| Pipeline | `https://ringkasan-net--glm-ocr-pipeline-pipelinefrontend-process.modal.run` |

---

## Done

- [x] `glm-ocr-single`: single-container pipeline (PP-DocLayoutV3 + vLLM) — replaces hand-rolled 3-app pipeline
- [x] GPU snapshot working with current config; layout model on GPU in snapshot
- [x] MTP speculative decoding; 8-case sequential + 16-concurrent batch warmup
- [x] Default DPI raised 200 → 300 to fix PP-DocLayoutV3 missing small headings
- [x] DPI benchmark: `test_dpi_benchmark.py` comparing Modal vs ZAI at 200/300 DPI
- [x] `fullpage` endpoint for whole-page OCR benchmark (no layout detection)
- [x] Concurrent-request safety patch for glmocr `_workers.py` (layout + vLLM semaphores)
- [x] Cold-start timing surfaced in response (`cold_start` in `timing`)

## Pending

- [ ] Implement vanilla/complementary/comprehensive pipeline modes in glm_ocr_single.py
- [ ] Investigate ZAI `layout_parsing` API as drop-in replacement for non-critical paths
