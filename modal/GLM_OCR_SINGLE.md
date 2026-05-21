# GLM-OCR Single Container — Notes & Findings

## Architecture

One L4 GPU container running both layout detection and OCR:

- **PP-DocLayoutV3** — GPU (DETR-based, ~400 MB), loaded from CPU snapshot
- **vLLM / GLM-OCR 9B** — GPU, managed via sleep/wake + GPU memory snapshot
- All OCR calls go to `localhost:8000` — no cross-container network hops

## Snapshot Strategy

Modal memory snapshot with `enable_memory_snapshot=True` + `experimental_options={"enable_gpu_snapshot": True}`.

**Why GPU snapshot is required**: without `enable_gpu_snapshot`, Modal runs `snap=True` in a CPU-only environment (`CUDA_VISIBLE_DEVICES=none`). vLLM cannot start without GPU access, so the snapshot can never be created.

**Flow:**
1. `snap=True` (`start()`): load layout weights on CPU → start vLLM → warmup → `/sleep` (offloads weights to CPU RAM) → snapshot taken
2. `snap=False` (`wake()`): `/wake_up` (weights from CPU RAM back to GPU) → layout `.to("cuda")` → batch warmup (16 concurrent) → ready

**Snapshot restore time (warm wake):** ~4s total
- wakeup_s: ~0.5s (vLLM weight restore via PCIe)
- health_s: ~0.01s
- layout_gpu_s: ~3.3s (`.to("cuda")` + warmup inference)
- batch_warmup_s: ~0.3s (16 concurrent Triton pre-compile)

## GPU Snapshot SIGSEGV Fix

**Error:** `Transient snapshot error: failed to restore container from snapshot with exit code 139. Will retry with no snapshots.`

Exit code 139 = SIGSEGV. Caused by CUDA graphs in the snapshot containing stale device pointers after restore.

**Root cause:** The original `snap=True` warmup used only 3 image sizes with `max_tokens=10`. This produced very few decode steps → vLLM captured only a small set of CUDA graph variants in the snapshot. After restore, real inference needed graph shapes not present in the snapshot → vLLM allocated new GPU memory → conflicted with the restored snapshot's memory layout → crash.

**Fix:** Expanded warmup to 8 image sizes with `max_tokens=128`. More decode steps across more size variants → vLLM captures a broader CUDA graph space before the snapshot is taken → the restored snapshot is self-contained → no new GPU allocations needed after restore → no SIGSEGV.

**Current warmup (snap=True):**
```python
warmup_cases = [
    ((336,  336),  112_896,   512_000, 128, "Text Recognition:"),    # 576 visual tokens
    ((1500, 200),  112_896,   512_000, 128, "Text Recognition:"),    # ~1530 visual tokens
    ((640,  800),  112_896,   512_000, 128, "Text Recognition:"),    # ~2612 visual tokens
    ((672,  450),  112_896, 1_003_520, 128, "Table Recognition:"),   # ~1543 visual tokens
    ((1500, 500),  112_896, 1_003_520, 128, "Table Recognition:"),   # ~3826 visual tokens
    ((1000, 1000), 112_896, 1_003_520, 128, "Table Recognition:"),   # ~5120 visual tokens
    ((336,  168),  112_896,   512_000, 128, "Formula Recognition:"), # ~576 visual tokens
    ((640,  800),  112_896,   512_000, 128, "Formula Recognition:"), # ~2612 visual tokens
]
```

## vLLM Configuration

```
--enable-sleep-mode
--gpu-memory-utilization 0.6
--max-model-len          8192
--max-num-seqs           16
--max-num-batched-tokens 8192
--dtype                  bfloat16
--speculative-config     {"method": "mtp", "num_speculative_tokens": 3}
```

**Key decisions:**
- `--max-num-seqs 16`: matches L4 memory bandwidth (300 GB/s). Tested 64 → regression (40% GPU util, 36s vs 29s). 16 is optimal.
- `--max-num-batched-tokens 8192`: enables chunked prefill for large visual token crops (up to 5120 tokens). Reduces `ocr_first_result_s` from ~3.5s to ~0.6s.
- `MTP num_speculative_tokens 3`: 91% acceptance rate on document text → ~2.73× effective decode throughput. Critical for long-output regions (large tables).
- `--enforce-eager`: **do not use with MTP**. Disables CUDA graphs which MTP requires for efficiency. Result: 119s max region (vs 6s with CUDA graphs).

## Performance Baselines (warm container, pmk.pdf 31 pages)

| config | total_s | ocr_wall_s | max_exec |
|---|---|---|---|
| MTP + CUDA graphs | ~24-28s | ~21-25s | ~4-6s |
| No MTP + CUDA graphs | ~31s | ~28s | ~4.6s |
| MTP + enforce-eager | ~245s | ~241s | ~119s |

## Batch Warmup in `wake()`

Added `_batch_warmup(n=16)` called after `_activate_layout_gpu()` in `wake()`. Sends 16 concurrent requests to vLLM to pre-compile batch-16 Triton kernels before the first user request arrives.

**Why:** `snap=True` warmup sends requests sequentially (effective batch size = 1 inside vLLM). Real inference submits 16 concurrent requests (batch = 16). Triton JIT-compiles separate kernels per batch size. Without this, the first cold-start request pays ~4s for batch-16 compilation.

## Lessons Learned

1. `enable_gpu_snapshot` is required (not optional) when `snap=True` needs GPU.
2. Removing GPU snapshot breaks snap=True — Modal runs it CPU-only without the flag.
3. `--enforce-eager` + MTP is catastrophically slow — do not combine.
4. SIGSEGV on restore is fixed by comprehensive warmup coverage, not by removing MTP or adding `--enforce-eager`.
5. `max_tokens` per request has no effect on snapshot stability.
6. Reducing `--max-num-seqs` to reduce CUDA graph count hurts throughput proportionally — not worth it.
