# GPU Compatibility — PaddleOCR HPS Stack

The HPS stack has three components with different GPU requirements. All three must
be satisfied simultaneously. The binding constraint is usually **flash-attn** (SM ≥ 80).

---

## Quick Reference

| GPU | SM | PP¹ | vLLM² | FA³ | HPS Works? | Notes |
|-----|----|-----|-------|-----|------------|-------|
| V100 | 7.0 | ✅ | ⚠️ | ❌ | ❌ | No flash-attn; vLLM is very slow (~1 tok/s/req) |
| T4 | 7.5 | ✅ | ⚠️ | ❌ | ❌ | No flash-attn; no AWQ quant |
| RTX 2080 Ti | 7.5 | ✅ | ⚠️ | ❌ | ❌ | Same as T4 |
| A100 40/80GB | 8.0 | ✅ | ✅ | ✅ | ✅ | Gold standard |
| A30 | 8.0 | ✅ | ✅ | ✅ | ✅ | |
| RTX 3080 | 8.6 | ✅ | ✅ | ✅ | ✅ | |
| RTX 3090 | 8.6 | ✅ | ✅ | ✅ | ✅ | Good value |
| RTX A4000 | 8.6 | ✅ | ✅ | ✅ | ✅ | |
| RTX A4500 | 8.6 | ✅ | ✅ | ✅ | ✅ | Tested ✓ (2.2 p/s) |
| RTX A5000 | 8.6 | ✅ | ✅ | ✅ | ✅ | |
| RTX A6000 | 8.6 | ✅ | ✅ | ✅ | ✅ | |
| RTX 4070 Ti Super | 8.9 | ✅ | ✅ | ✅ | ✅ | Tested ✓ (1.9 p/s) |
| RTX 4080 | 8.9 | ✅ | ✅ | ✅ | ✅ | |
| RTX 4090 | 8.9 | ✅ | ✅ | ✅ | ✅ | Best consumer value |
| L4 | 8.9 | ✅ | ✅ | ✅ | ✅ | Good cloud option |
| L40 / L40S | 8.9 | ✅ | ✅ | ✅ | ✅ | |
| H100 SXM/PCIe | 9.0 | ✅ | ✅ | ✅ | ✅ | Best overall; uses FA3 |
| H200 | 9.0 | ✅ | ✅ | ✅ | ✅ | |
| B100 / B200 | 10.0 | ❌ | ✅⁴ | ❌ | ❌ | PaddlePaddle not compiled for SM 100 |
| RTX 5080 / 5090 | 12.0 | ❌ | ✅⁴ | ❌ | ❌ | PaddlePaddle not compiled for SM 120 |

¹ PaddlePaddle (Triton layout detection)  
² vLLM (VLM inference)  
³ flash-attn 2.8.2 (required by vLLM for full performance)  
⁴ vLLM supports Blackwell but the PaddlePaddle layout model will fail  

---

## Why Each Component Fails

### PaddlePaddle (Triton layout detection)
Official wheels are compiled for: **SM 61, 70, 75, 80, 86, 89, 90**

- SM < 6.1 (Maxwell and older): not compiled in
- SM 10.0 (Blackwell datacenter): not in official wheels — community workaround exists at [horhe-dvlp/paddlepaddle-sm120-wheels](https://github.com/horhe-dvlp/paddlepaddle-sm120-wheels)
- SM 12.0 (Blackwell consumer RTX 50xx): not in official wheels, needs custom build

Error seen when unsupported:
```
Mismatched GPU Architecture: The installed PaddlePaddle package was compiled
for 60 61 70 75 80 86, but your current GPU is 120
```

### vLLM
Minimum: **SM 7.5** (can start, but degraded)  
Recommended: **SM 8.0+**

- SM 7.5 (T4, RTX 20xx): no flash-attn, no AWQ, very slow generation
- SM 8.0+: full flash-attn 2, chunked prefill, prefix caching, CUDA graphs
- SM 9.0 (H100): flash-attn 3, FP8 quantization
- SM 10.0+: requires CUDA 12.8 runtime

### flash-attn 2.8.2
Hard minimum: **SM 8.0**

- SM < 8.0: not compiled in at all — vLLM falls back to slower PyTorch attention
- SM 8.0–9.0: fully supported
- SM 10.0/12.0 (Blackwell): not supported in 2.x line; tracking issue open

### CUDA toolkit version requirements

| Architecture | SM | Minimum CUDA |
|---|---|---|
| Ada Lovelace | 89 | CUDA 11.8 |
| Hopper | 90 | CUDA 11.8 (basic) / 12.0 (full) |
| Blackwell datacenter | 100 | CUDA 12.8 |
| Blackwell consumer | 120 | CUDA 12.8 |

The base image (`paddleocr-hps:paddlex3.4-gpu`) ships CUDA **11.8** toolkit, so SM 89
and SM 90 work (CUDA 11.8 added their support). SM 100/120 require CUDA 12.8 which
the container doesn't have — this is the primary reason Blackwell fails.

---

## Recommended GPUs by Use Case

**Cloud / RunPod / Vast.ai (best availability):**
- RTX 3090 — cheapest Ampere, fully compatible
- RTX 4090 — fastest consumer, Ada Lovelace, fully compatible
- A100 80GB — best throughput per dollar for high concurrency

**If limited to ≤ $1/hr on Vast.ai:**
- RTX 3080/3090 or A4000/A4500 (Ampere, SM 86)
- RTX 4070 Ti / 4080 (Ada, SM 89) — also fine

**Avoid:**
- V100, T4, RTX 20xx series — no flash-attn, vLLM throughput is ~5-10× slower
- Any RTX 50xx / Blackwell — PaddlePaddle layout detection fails

---

## Performance Observed

| GPU | SM | VRAM | Throughput (31 pages, 8 concurrent) |
|-----|-----|------|--------------------------------------|
| RTX A4500 | 8.6 | 20 GB | 2.2 pages/sec |
| RTX 4070 Ti Super | 8.9 | 16 GB | 1.9 pages/sec |
| Tesla V100 | 7.0 | 16 GB | ~0.2 pages/sec (524 timeout) |
| Blackwell (SM 120) | 12.0 | 12 GB | ❌ fails at startup |

The A4500 vs 4070 Ti Super gap is mostly VRAM: less KV cache = less batching headroom.
The V100 failure is flash-attn missing — vLLM falls back to slow attention kernels.

---

## NVIDIA Container Runtime — Driver Constraint Issue

The base image sets `NVIDIA_REQUIRE_CUDA=driver>=470,driver<471`. On hosts with newer
drivers (e.g. driver 580 / CUDA 13.0), the NVIDIA Container Runtime interprets this as
a constraint violation and skips injecting the host's `libcuda.so.1`. The container then
falls back to the CUDA stub library (`/lib/x86_64-linux-gnu/libcuda.so.1`), which has no
real GPU access, causing `torch._C._cuda_init()` → `RuntimeError: CUDA unknown error`.

Symptom: `nvidia-smi` works, `/dev/nvidia0` exists, but `torch.cuda.is_available()` returns False.

**Fix (already applied to Dockerfile.hps):**
```dockerfile
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.8"
```
This overrides the base image constraint to only require CUDA ≥11.8, removing the
`driver<471` upper bound so the runtime injects the driver library on any modern driver.

**Temporary workaround (without rebuilding):**
```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
bash /opt/start_hps.sh
```
(Only works if the runtime mounted the library in `/usr/local/nvidia/lib64/` despite the check.)

---

## Blackwell Workaround (Experimental)

If you must use an RTX 50xx or B-series GPU, a community-built PaddlePaddle wheel
for SM 120 exists. This is unsupported and untested with this image:

1. Replace PaddlePaddle in the Dockerfile with the community SM 120 wheel
2. Replace flash-attn with a Blackwell-compatible build (not yet in 2.x line)
3. Ensure CUDA 12.8+ runtime is available in the image

Not recommended for production until official support lands.
