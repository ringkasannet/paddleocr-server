# Modal Cold Start — vLLM & Inference Deployments

Findings, patterns, and decisions from deploying GLM-OCR and PP-DocLayoutV3 on Modal with memory snapshots.

---

## How Modal Memory Snapshots Work

Modal memory snapshots freeze a container's in-memory state so future cold starts restore from that frozen state instead of re-running the full initialization. Two snapshot types exist:

**CPU snapshot** (`enable_memory_snapshot=True`)
- Captures Python process memory: imported modules, CPU tensors, object state
- No GPU state captured
- Stable across GPU worker rotation — nothing to go stale
- Restore: Modal downloads the snapshot from storage, restores process memory
- snap=True runs on a CPU-only worker (Modal sets `CUDA_VISIBLE_DEVICES=none`)

**GPU snapshot** (`experimental_options={"enable_gpu_snapshot": True}`)
- Additionally captures GPU VRAM: model weight tensors, CUDA graphs
- Requires snap=True to have GPU access — impossible without this flag
- Unstable: CUDA virtual addresses baked into the snapshot become stale when restored on a different GPU worker → SIGSEGV (exit code 139)
- Fix: comprehensive warmup coverage (see SIGSEGV section below)

**Lifecycle:**
```
snap=True  → runs once during snapshot creation (on Modal's snapshot worker)
             → snapshot is taken of the container's memory state
snap=False → runs on every serving container after snapshot restore
             → also runs on the original snap=True container after snapshot is taken
```

Modal's dashboard "Startup" timer ends when the container process starts — **before** `snap=False` completes. The real cold start includes `snap=False` execution time on top of what the dashboard shows.

---

## Architecture Overview

### glm-ocr-single (Single GPU Container)

Everything runs in one L4 GPU container — no cross-container RPC hops.

```
L4 GPU container
├── PP-DocLayoutV3 (DETR, ~400 MB)  — GPU, in-process
└── vLLM / GLM-OCR 9B               — GPU, subprocess on localhost:8000
```

**Snapshot:** GPU snapshot required (vLLM subprocess needs GPU in snap=True)

**Warm latency (pmk.pdf, 31 pages):**
- render: ~0.5s, layout: ~2.5s, OCR: ~25s, total: ~28s

---

### layout-worker (CPU + GPU Split)

```
CPU Processor container  →  LayoutDetector GPU container
(PDF render, text, NMS)      (DETR inference only)
```

**Snapshot:** CPU-only snapshot (model weights loaded to CPU RAM in snap=True)

**Warm dispatch:** ~1.5s cross-container RPC overhead per call

---

## Snapshot Configuration Reference

### glm-ocr-single (L4, vLLM + DETR)

```python
@app.cls(
    gpu="L4",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},  # required: vLLM needs GPU in snap=True
    scaledown_window=60,
    timeout=600,
    max_containers=2,
)
@modal.concurrent(max_inputs=2, target_inputs=1)
class DocumentOCRWorker:
```

**Environment variables for snapshot stability:**
```python
.env({
    "VLLM_SERVER_DEV_MODE":          "1",    # enables /sleep and /wake_up endpoints
    "TORCHINDUCTOR_COMPILE_THREADS": "1",    # reduces Triton JIT parallelism → more stable graph capture
    "PYTORCH_CUDA_ALLOC_CONF":       "expandable_segments:True",
    "TORCH_NCCL_ENABLE_MONITORING":  "0",    # suppress broken-pipe noise
    "TORCH_CPP_LOG_LEVEL":           "ERROR",
})
```

---

### layout-worker LayoutDetector (T4/L4, DETR only)

```python
@app.cls(
    gpu="T4",
    enable_memory_snapshot=True,
    # NO enable_gpu_snapshot — CPU-only snapshot eliminates SIGSEGV permanently
    scaledown_window=60,
    timeout=120,
    max_containers=8,
)
@modal.concurrent(max_inputs=4, target_inputs=3)
class LayoutDetector:
```

**Environment variables:**
```python
.env({
    "HF_HUB_ENABLE_HF_TRANSFER":        "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",  # eliminates ~0.1s import overhead
})
```

---

## snap=True Patterns

### vLLM + sleep/wake (glm-ocr-single, glm-ocr)

```python
@modal.enter(snap=True)
def start(self) -> None:
    # 1. Pre-load layout model weights to CPU RAM
    #    CPU tensors are captured in the CPU snapshot.
    #    wake() only pays PCIe transfer (~50ms), not a full volume reload.
    self._layout_model = AutoModelForObjectDetection.from_pretrained(LAYOUT_MODEL_ID).eval()

    # 2. Start vLLM subprocess on GPU
    cmd = ["vllm", "serve", MODEL_ID, "--enable-sleep-mode", ...]
    self._proc = subprocess.Popen(cmd)
    _wait_ready(VLLM_PORT)

    # 3. Exhaustive warmup — see warmup section below
    _warmup_sequential(VLLM_PORT)

    # 4. Sleep: offloads model weights to CPU RAM, keeps CUDA graphs in VRAM
    requests.post(f"http://localhost:{VLLM_PORT}/sleep", timeout=120)
    # ← snapshot is taken here
```

**Why `--enable-sleep-mode`:** vLLM's `/sleep` endpoint offloads model weights (18 GB) from GPU VRAM to CPU RAM before snapshot. The snapshot captures weights in CPU RAM. `/wake_up` in snap=False PCIe-transfers them back (~0.5s).

**Why GPU snapshot is mandatory here:** snap=True launches `vllm serve` as a subprocess. vLLM reads `CUDA_VISIBLE_DEVICES` at startup. Without `enable_gpu_snapshot`, Modal sets `CUDA_VISIBLE_DEVICES=none` → vLLM's `device_id_to_physical_device_id` calls `int("none")` → `ValueError`. GPU snapshot is the only way snap=True gets GPU visibility.

---

### PyTorch model (layout-worker LayoutDetector)

```python
@modal.enter(snap=True)
def load(self):
    # Load weights to CPU RAM only — no GPU needed, no SIGSEGV risk.
    # Snapshot captures ~400 MB of model tensor data.
    # activate() (snap=False) pays only ~50ms PCIe transfer to move to GPU.
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    model_dir = os.path.join(WEIGHTS_PATH, "PP-DocLayoutV3")
    self._processor = AutoImageProcessor.from_pretrained(model_dir)
    self._model     = AutoModelForObjectDetection.from_pretrained(model_dir).eval()
    # NO .to("cuda") here
```

**Why CPU-only for PyTorch models:** `from_pretrained()` defaults to CPU. No CUDA API is called → snap=True runs fine without `enable_gpu_snapshot` → CPU snapshot is taken → stable across worker rotation.

---

## snap=False Patterns

### vLLM wake (glm-ocr-single)

```python
@modal.enter(snap=False)
def wake(self) -> None:
    t0 = time.time()

    # 1. Restore vLLM weights: CPU RAM → GPU via PCIe (~0.5s for 18 GB on L4)
    requests.post(f"http://localhost:{VLLM_PORT}/wake_up", timeout=120)
    _wait_ready(VLLM_PORT)
    t_wakeup = time.time()

    # 2. Create persistent HTTP session to vLLM (loopback)
    self._session = requests.Session()
    self._session.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=32))

    # 3. Move layout model: CPU RAM → GPU (~50ms PCIe + ~3s warmup)
    self._activate_layout_gpu()
    t_layout = time.time()

    # 4. Concurrent batch warmup — see batch warmup section below
    batch_warmup_s = self._batch_warmup(n=16)

    self._cold_start_timing = {
        "wakeup_s":       round(t_wakeup - t0,       3),   # vLLM weight restore
        "health_s":       round(t_ready  - t_wakeup, 3),   # health check
        "layout_gpu_s":   round(t_layout - t_ready,  3),   # layout .to("cuda") + warmup
        "batch_warmup_s": batch_warmup_s,                   # Triton batch-16 pre-compile
        "total_s":        round(t_done   - t0,       3),
    }
```

**Measured wake() breakdown (L4, warm snapshot):**

| step | time |
|---|---|
| vLLM `/wake_up` PCIe transfer | ~0.5s |
| health check | ~0.01s |
| layout `.to("cuda")` + warmup | ~3.3s |
| batch warmup (16 concurrent) | ~0.3s |
| **total wake()** | **~4.1s** |

---

### PyTorch model activate (layout-worker)

```python
@modal.enter(snap=False)
def activate(self):
    import torch
    torch.backends.cudnn.benchmark = False  # skip per-shape autotuning — saves ~2.5s
    self._device = "cuda"
    self._model  = self._model.to("cuda").eval()  # CPU RAM → GPU (~50ms PCIe)
    # warmup inference to compile CUDA kernels before first request
    with torch.no_grad():
        self._model(**dummy_inputs_on_cuda)
```

**`cudnn.benchmark = False` — tested, net negative:** cuDNN's default tests multiple convolution algorithms and caches the fastest. The hypothesis was this costs ~2.5s during activate() warmup. In practice, CUDA context first-init (~2.5s) dominates activate() regardless of this setting, so benchmark=False only saved ~0.3s on cold start but cost ~0.67s on every detect() inference (suboptimal algorithm). Net: 0.44s worse total rpc_wall. Do not use.

**Modal dashboard "Startup" vs actual dispatch:**
```
Modal "Startup" timer ends here
        ↓
Container starts (6.75s: snapshot restore + init)
activate() runs  (3.5s: PCIe + warmup) ← NOT in Modal's metric
routing overhead (0.5s)
        ↓
queued_s = 10.5s   ← what the client actually waits
```

---

## Lazy Activation Fallback Guard

`snap=False` only runs on snapshot-restored containers. Containers running fresh (no snapshot yet, or during snapshot creation) run only `snap=True`. If `snap=True` doesn't move the model to GPU, inference on those containers would use CPU.

Add a guard at the top of every inference method:

```python
@modal.method()
def detect(self, page_jpegs):
    if not hasattr(self, "_device"):
        # snap=False didn't run — activate GPU now (first request only, then cached)
        torch.backends.cudnn.benchmark = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model  = self._model.to(self._device).eval()
        # warmup
        with torch.no_grad():
            self._model(**dummy_inputs)
        print(f"[detect] lazy activation on {self._device}")
    # ... inference
```

**Important:** set `t0 = time.time()` AFTER this guard. Otherwise lazy activation time is counted inside `queued_s`, making cold start metrics misleading.

---

## vLLM Warmup Strategy

### The SIGSEGV Problem

`Transient snapshot error: failed to restore container from snapshot with exit code 139`

Exit 139 = SIGSEGV. Caused by CUDA graph device pointers baked into the GPU snapshot becoming invalid when restored in a different CUDA context (different GPU worker or different memory layout).

**Root cause of original SIGSEGV:** snap=True warmup ran only 3 image sizes with `max_tokens=10`. vLLM built CUDA graphs for only a small set of (batch_size × sequence_length) shapes. After restore, real inference requested shapes not in the snapshot → vLLM allocated new GPU memory → address conflict with restored snapshot → crash.

**Fix:** exhaustive warmup covering the full visual token range real inference will use:

```python
warmup_cases = [
    # (image_size,    min_pixels,  max_pixels,   max_tok, prompt)
    # Text: covers 576, ~1530, ~2612 visual tokens
    ((336,  336),  112_896,   512_000, 128, "Text Recognition:"),
    ((1500, 200),  112_896,   512_000, 128, "Text Recognition:"),
    ((640,  800),  112_896,   512_000, 128, "Text Recognition:"),
    # Table: covers ~1543, ~3826, ~5120 visual tokens (full budget)
    ((672,  450),  112_896, 1_003_520, 128, "Table Recognition:"),
    ((1500, 500),  112_896, 1_003_520, 128, "Table Recognition:"),
    ((1000, 1000), 112_896, 1_003_520, 128, "Table Recognition:"),
    # Formula: covers ~576, ~2612 visual tokens
    ((336,  168),  112_896,   512_000, 128, "Formula Recognition:"),
    ((640,  800),  112_896,   512_000, 128, "Formula Recognition:"),
]
```

**Why `max_tokens=128` (not 10):** With only 10 decode steps, vLLM captures CUDA graphs for short-sequence shapes only. Longer outputs at inference time request graph shapes not in the snapshot → new GPU allocation → SIGSEGV. 128 tokens forces enough decode steps to capture the graph shapes real inference needs.

**Visual token math (CogViT 14×14 patches, 196 px/token):**
```
 112,896 px  →   576 tokens   (minimum)
 302,400 px  → 1,543 tokens
 512,000 px  → 2,612 tokens   (text max budget)
 750,000 px  → 3,826 tokens
1,003,520 px → 5,120 tokens   (table max budget)
```

### Snapshot Lifecycle Issue

When a GPU snapshot fails to restore, Modal falls back to a full cold start but **does not rebuild the broken snapshot**. Every subsequent cold start retries the same stale snapshot → SIGSEGV loop. The only reset is a code change that triggers a new deploy (new snap=True run).

---

## Batch Warmup in snap=False (Triton Batch-16 JIT)

snap=True warmup sends requests **sequentially** — inside vLLM, each request is processed as batch_size=1. Real inference sends 16 concurrent requests simultaneously — vLLM processes them as batch_size=16.

Triton JIT-compiles separate kernels per batch size. batch_size=1 warmup does not compile batch_size=16 kernels. Without the fix, the first real request triggers ~4s of Triton compilation inside the request.

**Fix:** after `/wake_up` in snap=False, send `n` concurrent requests using ThreadPoolExecutor to force vLLM to process a batch of N and compile the batch-N kernels before any user request arrives:

```python
def _batch_warmup(self, n: int = 16) -> float:
    from PIL import Image
    warmup_cases = [...]  # same sizes as snap=True warmup

    def _send_one(idx: int) -> int:
        # small dummy request, max_tokens=32 (just enough to trigger kernel compile)
        resp = self._session.post(f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                                  json={..., "max_tokens": 32}, timeout=120)
        return resp.status_code

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as pool:
        statuses = list(pool.map(_send_one, range(n)))
    return round(time.time() - t0, 3)  # typically ~0.3s
```

---

## vLLM Configuration Decisions

### Flags

```
--enable-sleep-mode              # required for snapshot pattern (enables /sleep, /wake_up)
--gpu-memory-utilization 0.6     # L4: 0.6 × 24 GB = 14.4 GB for KV cache
--max-model-len 8192             # covers all document crop sizes
--max-num-seqs 16                # optimal for L4 300 GB/s memory bandwidth
--max-num-batched-tokens 8192    # enables chunked prefill for 5120-token table crops
--dtype bfloat16
--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

### Key Decisions

**`--max-num-seqs 16`** (L4): Memory bandwidth bound. L4 has 300 GB/s. GLM-OCR 9B in bfloat16 ≈ 18 GB weights. Each decode step reads all weights: 18 GB / 300 GB/s ≈ 60ms. At 16 sequences: 16 tokens/step. Tested 64 → GPU util dropped 80% → 40%, total_s increased 29s → 36s due to memory bandwidth saturation. 16 is optimal.

**`--max-num-batched-tokens 8192`**: Without this, MTP's default prefill cap is 2048 tokens. Large table crops produce 5120 visual tokens → prefill split across 3 steps. With 8192: one prefill step → `ocr_first_result_s` drops from ~3.5s to ~0.6s.

**MTP `num_speculative_tokens: 3`**: 91% acceptance rate on document text (predictable regulatory language). Effective throughput: 3 × 0.91 = 2.73× vs no speculative decoding. Critical for large table regions that generate hundreds of output tokens. Without MTP, one 700-token table region took 41s; with MTP, ~15s.

**`--enforce-eager` + MTP**: catastrophically broken. MTP's speculative decode requires CUDA graph optimization for efficiency. In eager mode: each draft+verify cycle runs as raw Python-level forward passes → 119s max region vs 6s with CUDA graphs. Never combine.

### Performance Baselines (warm, pmk.pdf 31 pages, L4)

| config | total_s | ocr_wall_s | max_exec |
|---|---|---|---|
| MTP + CUDA graphs | ~24-28s | ~21-25s | ~4-6s |
| No MTP + CUDA graphs | ~31s | ~28s | ~4.6s |
| MTP + enforce-eager | ~245s | ~241s | ~119s |

---

## Cold Start Timing Reference

### glm-ocr-single (L4, GPU snapshot)

```
snap=True (runs once, Modal snapshot worker):
  layout model load to CPU     ~1.5s
  vLLM startup                 ~60-120s  (weights loaded, CUDA graphs compiled)
  sequential warmup (8 cases)  ~30s
  /sleep (weights to CPU RAM)  ~5s
  ─────────────────────────────────
  total snap=True              ~2-3 min  (only paid once per deploy)

snap=False (every cold start, adds ~4s to Modal's "Startup" metric):
  /wake_up (CPU RAM → GPU)     ~0.5s
  health check                 ~0.01s
  layout .to("cuda") + warmup  ~3.3s
  batch warmup (16 concurrent) ~0.3s
  ─────────────────────────────────
  total snap=False             ~4.1s
```

**Modal dashboard shows:** ~6-10s startup (snapshot restore + snap=False combined)

---

### layout-worker LayoutDetector (T4, CPU snapshot)

```
snap=True (runs once, CPU-only Modal worker):
  model weights to CPU RAM     ~1.5s  (from Modal volume)
  ─────────────────────────────
  total snap=True              ~1.5s  (snapshot: ~400 MB)

snap=False (every cold start):
  snapshot restore (400 MB)    ~5s
  container init               ~1.75s
  .to("cuda") + warmup         ~0.5s  (PCIe ~50ms + warmup ~0.5s with benchmark=False)
  ─────────────────────────────────
  total snap=False             ~7.25s

cross-container routing        ~0.5s
─────────────────────────────────────
queued_s (measured)            ~7.5-8s  (target after cudnn fix)
```

**Modal dashboard shows:** ~6.75s startup (excludes snap=False time)

**Gap explained:** Modal's "Startup" timer ends when the container process starts, before snap=False runs. `queued_s - Modal startup ≈ snap=False time + routing`.

---

## CPU vs GPU Snapshot Decision Matrix

| scenario | recommendation | reason |
|---|---|---|
| vLLM subprocess in snap=True | GPU snapshot required | vLLM reads `CUDA_VISIBLE_DEVICES` at startup; `none` → ValueError |
| PyTorch model, load to CPU in snap=True | CPU snapshot preferred | No GPU in snap=True needed; no SIGSEGV risk; stable across worker rotation |
| PyTorch model, load to GPU in snap=True | GPU snapshot required | CUDA tensor addresses baked in; works but fragile (SIGSEGV on worker rotation) |
| Import-only snap=True (no model load) | CPU snapshot, tiny | Only Python module state captured (~50 MB); activate() loads model from volume |

**Import-only snapshot (not recommended for large models):** Snapshot restore is fast (~0.3s) but activate() must load the model from the volume (~4s with cold volume cache). For a 400 MB model, the volume load is slower than restoring from a 400 MB CPU snapshot (~5s). Net negative unless the volume is always warm.

---

## Common Mistakes and Fixes

| mistake | symptom | fix |
|---|---|---|
| Removed `enable_gpu_snapshot` with vLLM snap=True | `ValueError: invalid literal for int() with base 10: 'none'` | Re-add `enable_gpu_snapshot`; Modal sets CUDA_VISIBLE_DEVICES=none without it |
| Warmup too shallow (3 cases, max_tokens=10) | SIGSEGV (exit 139) on snapshot restore, intermittently | Expand warmup to 8+ cases, max_tokens=128 to cover full CUDA graph shape space |
| `--enforce-eager` + MTP speculative decoding | 119s per region (was 6s) | Never combine; MTP requires CUDA graphs for efficiency |
| `--max-num-seqs 64` on L4 | GPU utilization drops 80% → 40%, throughput regresses | L4 memory bandwidth saturates; use 16 for optimal throughput |
| `cudnn.benchmark = True` (default) in snap=False warmup | activate() takes 3.5s instead of 0.5s | Set `torch.backends.cudnn.benchmark = False` before first inference |
| No lazy activation guard | CPU inference when snap=False doesn't run | Add `if not hasattr(self, "_device"):` guard at top of inference method |
| Import-only snapshot for large models | Cold start SLOWER than before (volume load > snapshot restore) | Keep model weights in CPU snapshot; snapshot restore beats volume fetch for models >100 MB |
| Stale GPU snapshot after worker rotation | SIGSEGV on every cold start, never self-heals | Redeploy to trigger new snap=True; Modal never auto-rebuilds failed snapshots |
