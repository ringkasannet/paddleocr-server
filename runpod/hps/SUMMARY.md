# PaddleOCR HPS on RunPod — Project Summary

## What We're Building

A production-ready PaddleOCR High-Performance Serving (HPS) stack on RunPod GPU pods
that processes PDF documents via OCR+VLM and returns structured JSON. The goal is fast
cold starts (no 15-20 min setup) and high throughput (~2-3 pages/second under concurrent load).

---

## Architecture

```
Client
  │  POST /v1/hps/pipeline  (JSON: {file: url, fileType: 0})
  ▼
FastAPI Gateway  :8080     /workspace/.venv_gateway/bin/uvicorn
  │  gRPC (paddlex_hps_client)
  ▼
Triton Server    :8000/:8001  /opt/tritonserver/bin/tritonserver
  │  layout-parsing model   → PP-DocLayoutV3 (PaddlePaddle, runs on GPU)
  │  restructure-pages model
  │  HTTP OpenAI-compatible API
  ▼
vLLM Server      :8118     python3 -m paddleocr genai_server
                             model: PaddleOCR-VL-1.5-0.9B
```

**Concurrency model:** Triton uses dynamic batching (up to 16 layout-parsing requests at once,
2 Python backend instances). vLLM uses continuous batching across all pending VLM requests.
The gateway caps at 16 concurrent inference requests. This allows one pod to efficiently
handle 8-16 simultaneous document requests.

---

## Environment

| Item | Value |
|------|-------|
| RunPod GPU | RTX A4500, 20 GB VRAM |
| Pod image (old) | `ringkasannet/paddleocr-hps:paddlex3.4-gpu` |
| Pod image (new, target) | `ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready` |
| Source base image | `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.4-gpu` |
| Container Python | `/paddlex/py310/bin/python3` (Python 3.10.4) |
| Container CUDA toolkit | 11.8 (nvcc) — but torch is 2.8.0+**cu128** |
| CUDA driver on pod | 580.x (supports up to CUDA 13.0) |
| Network volume | `/workspace` — persists across pod restarts |
| Docker Hub | `ringkasannet` |

---

## Storage Strategy

| Location | Contents | Persists? |
|----------|----------|-----------|
| Docker image | paddleocr, vllm, flash-attn, paddlex[ocr] | Yes (in image layer) |
| `/opt/tritonserver/` | Triton binary + backends | Yes (in image) |
| `/paddlex/py310/` | PaddlePaddle, paddlex, paddlex_hps_server | Yes (in image) |
| `/workspace/hps/` | HPS SDK (model_repo, pipeline_config.yaml, client whl) | Yes (network vol) |
| `/workspace/gateway/` | gateway app.py | Yes (network vol) |
| `/workspace/.venv_gateway/` | fastapi, uvicorn, paddlex[serving], paddlex_hps_client | Yes (network vol) |
| `/workspace/models/paddlex/` | PaddleOCR-VL-1.5-0.9B weights (~2 GB) | Yes (network vol) |
| `/workspace/models/hf_cache/` | HuggingFace cache | Yes (network vol) |
| `/workspace/.cache/vllm/` | vLLM compilation cache | Yes (network vol) |

Model weights are **never** in the Docker image. They download once on first run and stay on
the network volume. Symlinks connect the expected paths to the network volume:
```bash
ln -sfn /workspace/models/paddlex /root/.paddlex
ln -sfn /workspace/.cache/vllm /root/.cache/vllm
```

---

## GPU Memory Split

**Stable config: `gpu-memory-utilization: 0.50`**

| Component | VRAM usage |
|-----------|-----------|
| vLLM KV cache (0.50 util) | ~14.6 GiB |
| Triton + PP-DocLayoutV3 layout detection | ~4-5 GiB |
| Total | ~19-20 GiB (safe on 20 GB A4500) |

**0.75 GPU util causes OOM.** Even though vLLM itself fits, Triton's PaddlePaddle layout
detection runs on GPU and is evicted when vLLM over-allocates KV cache. Error:
`ResourceExhaustedError: Out of memory on GPU 0, 19.61 GB allocated`. Do not increase
gpu-memory-utilization beyond 0.50 without first reducing Triton instance_count to 1.

---

## Performance Benchmarks

| Scenario | Time | Pages/sec |
|----------|------|-----------|
| Single 3-page PDF (warm) | ~1.2s | ~2.5 p/s |
| 31-page PDF, 8 concurrent chunks | ~11-21s | ~1.5-2.8 p/s |
| Single 31-page PDF (sequential) | ~38s (cold) / ~15s (warm) | ~0.8-2 p/s |

**The critical fix that unlocked performance:** `pipeline_config.yaml` was using
`backend: native` (slow PaddlePaddle inference). Patching to `backend: vllm-server` with
`server_url: http://localhost:8118/v1` reduced a 3-page PDF from 38s → 1.2s.

---

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.hps` | Custom image with all Python packages pre-installed |
| `build_push.sh` | Build and push image to Docker Hub |
| `start_hps_v2.sh` | Startup script for the new pre-built image (use this) |
| `start_hps.sh` | Startup script for old image (runs setup on boot) |
| `setup.sh` | One-time setup: downloads SDK, installs Python packages, sets up gateway venv |
| `SETUP.md` | Full problem log with all errors encountered and their fixes |
| `PLAN.md` | Original architecture plan and gate-by-gate verification steps |
| `HANDOFF_DOCKER.md` | Guide for examining base image and building the custom image |

---

## Current Main Task: Build & Push Custom Docker Image

The old image requires 15-20 min setup on every fresh pod (installing vllm, paddleocr,
flash-attn into `/paddlex/py310/`). The new image bakes all of that in.

### Dockerfile.hps — What It Does

```dockerfile
FROM ringkasannet/paddleocr-hps:paddlex3.4-gpu

# 1. flash-attn pre-built wheel (CUDA 11.8 nvcc vs torch cu128 mismatch — cannot compile from source)
RUN pip install "https://.../flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"

# 2. Pin numpy immediately after (flash-attn upgrades to 2.x, breaking pandas/paddleocr CLI)
RUN pip install "numpy==1.26.4" --force-reinstall --no-deps

# 3. paddleocr (provides paddleocr CLI needed by genai_server)
RUN pip install paddleocr==3.5.0

# 4. vllm — direct pip install (bypasses CUDA check that fails at build time)
#    install_genai_server_deps calls is_cuda_available() → libcuda.so.1 not present during build
RUN pip install "vllm==0.10.2" "transformers<5.0.0" "einops" "uvloop"

# 5. matplotlib ≥3.9 (compatible with both numpy 1.x and 2.x)
RUN pip install "matplotlib>=3.9" --upgrade

# 6. paddlex[ocr] extras required for VLM pipeline
RUN pip install "paddlex[ocr]==$PADDLEX_VER"

# 7. numpy FINAL pin — must be absolute last step
RUN pip install "numpy==1.26.4" --force-reinstall --no-deps
```

### Build Command (run from WSL)

```bash
cd /mnt/d/paddleocr/PaddleOCR-main/runpod/hps
docker build -f Dockerfile.hps -t ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready --progress=plain .
```

### Push Command

```bash
docker push ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready
```

### Test After Build

```bash
docker run --rm ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready \
  /paddlex/py310/bin/python3 -c "
import paddleocr, vllm, flash_attn, numpy, matplotlib
print('paddleocr:', paddleocr.__version__)
print('vllm:', vllm.__version__)
print('numpy:', numpy.__version__)
print('All OK')
"
```

### After Pushing: Update RunPod Pod

Change pod image from `ringkasannet/paddleocr-hps:paddlex3.4-gpu`
to `ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready`

Then on fresh pod boot, run:
```bash
bash /workspace/paddleocr-server/runpod/hps/start_hps_v2.sh
```

`start_hps_v2.sh` auto-bootstraps the gateway venv if it's missing from the network volume
(first boot only, ~2 min). Subsequent boots start all three services immediately.

---

## All Known Problems and Fixes

### 1. `wget` / `curl` missing from base image
**Fix:** Added to Dockerfile: `apt-get install -y curl wget`  
`wait_healthy` in start scripts uses `wget -qO-` (not curl).

### 2. `config.pbtxt` missing in model_repo
Triton expects `config.pbtxt` but SDK ships `config_gpu.pbtxt` / `config_cpu.pbtxt`.  
**Fix:** `cp config_gpu.pbtxt config.pbtxt` in setup.sh.

### 3. `genai_server` subcommand missing
`genai_server` is dynamically registered only when vllm is importable.  
**Fix:** Install vllm before trying to use the subcommand.

### 4. flash-attn compilation fails during build
Container has CUDA 11.8 nvcc but torch 2.8.0+cu128. Source compilation fails.  
**Fix:** Use pre-built wheel from `mjun0812/flash-attention-prebuild-wheels`.

### 5. numpy 2.x breaks compiled packages
flash-attn and vllm both upgrade numpy to 2.x. pandas + old matplotlib compiled for 1.x.  
**Fix:** Always pin `numpy==1.26.4 --force-reinstall --no-deps` as the **last** pip install.

### 6. `install_genai_server_deps vllm` fails during Docker build
The command internally calls `is_cuda_available()` → tries to import paddle → needs
`libcuda.so.1` → not available at image build time.  
**Fix:** Directly pip install the packages it would have installed:
`vllm==0.10.2`, `transformers<5.0.0`, `einops`, `uvloop`.

### 7. `backend: native` in pipeline_config.yaml (performance killer)
Default config uses slow PaddlePaddle native inference instead of vLLM server.  
**Fix:** setup.sh patches pipeline_config.yaml:
```yaml
SubModules.VLRecognition.genai_config:
  backend: vllm-server
  server_url: http://localhost:8118/v1
  max_concurrency: 32
```

### 8. OOM at `gpu-memory-utilization: 0.75`
Triton's PaddlePaddle layout detection runs on GPU. vLLM at 0.75 util leaves only 87 MB
free, causing Triton cv worker to OOM during concurrent requests.  
**Fix:** Permanently set `gpu-memory-utilization: 0.50`.

### 9. `ln: failed to create symbolic link '/root/.cache/vllm'`
`/root/.cache` doesn't exist on a fresh pod container.  
**Fix:** Added `mkdir -p /root/.cache` before the symlink in all start scripts.

### 10. Gateway wait_healthy timeout (30s)
Gateway takes ~150s to start but the original timeout was 30s.  
**Fix:** Increased to 60s.

### 11. Triton restart race condition
On restart, new vLLM started while old Triton still held GPU memory.  
**Fix:** `pkill` then `pgrep` wait loop (up to 35s) before starting new processes.

### 12. Python packages lost on pod restart
`/paddlex/py310/` is the container filesystem — ephemeral, wiped on pod restart.  
**Fix:** Build custom Docker image with packages pre-installed (the current main task).

---

## HPS vs Section 4 + Load Balancer

| | HPS (1 pod) | Section 4 per pod |
|--|------------|-------------------|
| Concurrency | 16 requests simultaneously | 1 request at a time |
| 31-page PDF latency | ~11-21s | ~47-87s |
| Throughput per pod | ~2-3 pages/s | ~0.4-0.7 pages/s |
| To match HPS throughput | 1 pod | 4-5 pods (4-5× cost) |
| Architecture complexity | High (3 services) | Low (1 command) |

**Conclusion:** HPS is 4-5× more cost-efficient per GPU for large documents under concurrent
load. Section 4 only makes sense for simple/low-scale deployments or when ops simplicity
outweighs cost. For scale, the right approach is HPS on RunPod serverless — multiple HPS
pods auto-scaled by queue depth.

---

## Quick Reference: Pod Commands

```bash
# Start all services
bash /workspace/paddleocr-server/runpod/hps/start_hps_v2.sh

# Check health
wget -qO- http://localhost:8080/health && echo "Gateway OK"
wget -qO- http://localhost:8000/v2/health/ready && echo "Triton OK"
wget -qO- http://localhost:8118/health && echo "vLLM OK"

# Check GPU
nvidia-smi

# Test OCR (replace URL with any PDF)
curl -s -X POST http://localhost:8080/v1/hps/pipeline \
  -H "Content-Type: application/json" \
  -d '{"file": "https://example.com/doc.pdf", "fileType": 0, "visualize": false}' \
  | python3 -m json.tool | head -30

# External URL (RunPod proxy)
# https://<pod-id>-8080.proxy.runpod.net/v1/hps/pipeline
```
