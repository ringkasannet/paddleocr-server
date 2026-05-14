# PaddleOCR HPS on RunPod — Complete Setup Guide

## Architecture

```
Client → FastAPI Gateway (port 8080)
              ↓ gRPC (paddlex_hps_client)
          Triton Server (ports 8000/8001)   ← PP-DocLayoutV3 layout detection
              ↓ HTTP OpenAI-compatible
          vLLM Server (port 8118)            ← PaddleOCR-VL-1.5-0.9B
```

---

## Environment

| Item | Value |
|------|-------|
| Pod image | `ringkasannet/paddleocr-hps:paddlex3.4-gpu` |
| Source image | `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.4-gpu` |
| GPU | RTX A4500, 20 GB VRAM |
| CUDA driver | 580.x (supports up to CUDA 13.0) |
| Container CUDA runtime | 11.8 (from the paddlex/hps base) |
| Container Python | `/paddlex/py310/bin/python3` = Python 3.10.4 |

### What the image provides (read-only, always present)
| Component | Path |
|-----------|------|
| Triton binary | `/opt/tritonserver/bin/tritonserver` |
| Triton Python backend + stub | `/opt/tritonserver/backends/python/` |
| PaddlePaddle + paddlex | `/paddlex/py310/lib/python3.10/site-packages/` |
| `paddlex_hps_server` package | `/paddlex/py310/lib/python3.10/site-packages/paddlex_hps_server/` |

---

## Problems Encountered and Solutions

### Problem 1: `wget` missing from image
**Error:** `bash: wget: command not found`
**Fix:**
```bash
apt-get update && apt-get install -y wget
```

### Problem 2: HPS SDK only provides `config_gpu.pbtxt` / `config_cpu.pbtxt`
**Error:** `failed to open text file /workspace/hps/server/model_repo/layout-parsing/config.pbtxt: No such file or directory`
**Cause:** The SDK ships two device-specific config files but Triton expects `config.pbtxt`
**Fix:**
```bash
cp /workspace/hps/server/model_repo/layout-parsing/config_gpu.pbtxt \
   /workspace/hps/server/model_repo/layout-parsing/config.pbtxt
```

### Problem 3: `genai_server` subcommand missing
**Error:** `paddleocr: error: argument subcommand: invalid choice: 'genai_server'`
**Cause:** `genai_server` is dynamically registered only when vllm is importable. It requires
running `install_genai_server_deps vllm` first.
**Fix:** See Problem 4 and Problem 7.

### Problem 4: `flash-attn` compilation fails
**Error:** `RuntimeError: The detected CUDA version (11.8) mismatches PyTorch (12.8)`
**Cause:** Container has CUDA 11.8 toolkit (nvcc) but `/paddlex/py310/` has torch 2.8.0+cu128.
Compiling flash-attn from source fails.
**Fix:** Pre-install a compiled wheel:
```bash
/paddlex/py310/bin/pip install \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"
```

### Problem 5: numpy 2.x breaks compiled packages
**Error:** `ValueError: numpy.dtype size changed` / `AttributeError: _ARRAY_API not found`
**Cause:** Installing flash-attn upgrades numpy to 2.2.6, breaking pandas and old matplotlib
(both compiled for numpy 1.x).
**Fix (permanent — do this LAST after all other installs):**
```bash
/paddlex/py310/bin/pip install "numpy==1.26.4" --force-reinstall --no-deps
```
Also upgrade matplotlib to a numpy 2.x compatible version before any runs:
```bash
/paddlex/py310/bin/pip install "matplotlib>=3.9" --upgrade
```

### Problem 6: vllm installed in wrong environment
**Cause:** `install_genai_server_deps` hardcodes `/paddlex/py310/bin/python` for its pip calls.
vllm (0.10.2, PyTorch-based) is installed into `/paddlex/py310/` but `paddleocr` CLI was
only in `/workspace/.venv_vlm/`. Neither environment had both.
**Fix:** Install paddleocr into `/paddlex/py310/` where vllm already lives:
```bash
/paddlex/py310/bin/pip install paddleocr==3.5.0
```

### Problem 7: PYTHONPATH mixing causes numpy conflicts
**Symptom:** Adding `/paddlex/py310/` to PYTHONPATH made paddleocr from `.venv_vlm/` see
vllm, but also caused numpy version conflicts between the two environments.
**Fix:** Don't use PYTHONPATH. Run genai_server entirely from `/paddlex/py310/`:
```bash
/paddlex/py310/bin/python3 -m paddleocr genai_server ...
```

### Problem 8: paddlex `[ocr]` extras missing
**Error:** `DependencyError: PaddleOCR-VL-1.5 requires additional dependencies. pip install "paddlex[ocr]"`
**Fix:**
```bash
PADDLEX_VER=$(/paddlex/py310/bin/python3 -c "import paddlex; print(paddlex.__version__)")
/paddlex/py310/bin/pip install "paddlex[ocr]==$PADDLEX_VER"
```

### Problem 9: `PADDLEX_HPS_PIPELINE_CONFIG_PATH` not set
**Error:** `Exception: The pipeline () does not exist!`
**Cause:** `env.py` defaults `PIPELINE_CONFIG_PATH` to `""` (empty string).
**Fix:** Export before starting Triton:
```bash
export PADDLEX_HPS_PIPELINE_CONFIG_PATH=/workspace/hps/server/pipeline_config.yaml
export PADDLEX_HPS_DEVICE_TYPE=gpu
```

---

## Complete Setup Script (run once on a fresh pod)

```bash
#!/usr/bin/env bash
# Run once on a fresh pod to set up /workspace for HPS serving
set -euo pipefail

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# ── 1. System packages ───────────────────────────────────────
apt-get update -qq && apt-get install -y wget

# ── 2. HPS SDK ───────────────────────────────────────────────
wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz \
  -O /tmp/sdk.tar.gz
tar -xf /tmp/sdk.tar.gz -C /tmp/
mkdir -p /workspace/hps/server /workspace/hps/client
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/server/. /workspace/hps/server/
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/client/. /workspace/hps/client/
rm /tmp/sdk.tar.gz

# Patch pipeline config URL
sed -i 's|http://paddleocr-vlm-server:8080/v1|http://localhost:8118/v1|g' \
  /workspace/hps/server/pipeline_config.yaml

# Fix missing config.pbtxt for GPU
cp /workspace/hps/server/model_repo/layout-parsing/config_gpu.pbtxt \
   /workspace/hps/server/model_repo/layout-parsing/config.pbtxt

# ── 3. flash-attn (pre-compiled, avoids CUDA mismatch) ───────
/paddlex/py310/bin/pip install --no-cache-dir \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"

# ── 4. vllm genai server deps ─────────────────────────────────
# Uses /paddlex/py310/bin/python internally (hardcoded in paddleocr)
# Run from .venv_vlm but installs into /paddlex/py310/
/workspace/.venv_vlm/bin/paddleocr install_genai_server_deps vllm

# ── 5. Fix matplotlib for numpy 2.x compatibility ────────────
/paddlex/py310/bin/pip install "matplotlib>=3.9" --upgrade

# ── 6. Install paddleocr into /paddlex/py310/ ─────────────────
# Needed because vllm is there and genai_server requires both
/paddlex/py310/bin/pip install paddleocr==3.5.0

# ── 7. Install paddlex[ocr] extras ───────────────────────────
PADDLEX_VER=$(/paddlex/py310/bin/python3 -c "import paddlex; print(paddlex.__version__)")
/paddlex/py310/bin/pip install "paddlex[ocr]==$PADDLEX_VER"

# ── 8. Fix numpy LAST (after all installs that may upgrade it)
/paddlex/py310/bin/pip install "numpy==1.26.4" --force-reinstall --no-deps

# ── 9. Gateway venv ───────────────────────────────────────────
if [ ! -f /workspace/.venv_gateway/bin/python3 ]; then
  /paddlex/py310/bin/python3 -m venv /workspace/.venv_gateway
  /workspace/.venv_gateway/bin/pip install --upgrade pip -q
fi
/workspace/.venv_gateway/bin/pip install --no-cache-dir \
  fastapi==0.123.6 uvicorn==0.35.0 "paddlex[serving]>=3.4.0"
/workspace/.venv_gateway/bin/pip install --no-cache-dir \
  -r /workspace/hps/client/requirements.txt
WHL=$(ls /workspace/hps/client/paddlex_hps_client-*.whl | head -1)
/workspace/.venv_gateway/bin/pip install --no-cache-dir "$WHL"

# ── 10. Gateway app ───────────────────────────────────────────
mkdir -p /workspace/gateway
wget -q https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/paddleocr_vl_docker/hps/gateway/app.py \
  -O /workspace/gateway/app.py

echo ""
echo "=== Setup complete. Run: bash /workspace/start_hps.sh ==="
```

---

## Start Script (run every pod boot)

```bash
#!/usr/bin/env bash
# /workspace/start_hps.sh
set -euo pipefail

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export PADDLEX_HPS_PIPELINE_CONFIG_PATH=/workspace/hps/server/pipeline_config.yaml
export PADDLEX_HPS_DEVICE_TYPE=gpu
export HF_HOME=/workspace/models/hf_cache
export LD_LIBRARY_PATH=/opt/tritonserver/lib:${LD_LIBRARY_PATH:-}

MODEL_NAME=${MODEL_NAME:-PaddleOCR-VL-1.5-0.9B}
GPU_MEM_UTIL=${GPU_MEMORY_UTILIZATION:-0.50}

wait_healthy() {
  local url=$1 label=$2 timeout=$3
  local deadline=$((SECONDS + timeout))
  while [[ $SECONDS -lt $deadline ]]; do
    curl -sf "$url" > /dev/null 2>&1 && echo "[start] $label ready." && return 0
    echo "[start] Waiting for $label... (${SECONDS}s)"
    sleep 10
  done
  echo "[ERROR] $label not healthy after ${timeout}s" && exit 1
}

# 1. vLLM server
echo "[start] Starting vLLM server..."
VLLM_CFG=/tmp/vllm_backend.yaml
echo "gpu-memory-utilization: $GPU_MEM_UTIL" > $VLLM_CFG
/paddlex/py310/bin/python3 -m paddleocr genai_server \
  --model_name "$MODEL_NAME" \
  --backend vllm \
  --host 0.0.0.0 \
  --port 8118 \
  2>&1 | sed -u 's/^/[vllm] /' &
wait_healthy "http://localhost:8118/health" "vLLM" 600

# 2. Triton server
echo "[start] Starting Triton server..."
/opt/tritonserver/bin/tritonserver \
  --model-repository=/workspace/hps/server/model_repo \
  --backend-directory=/opt/tritonserver/backends \
  --backend-config=python,python-runtime=/paddlex/py310/bin/python3 \
  --backend-config=python,shm-default-byte-size=67108864 \
  --http-port=8000 --grpc-port=8001 \
  --model-control-mode=explicit \
  --load-model=layout-parsing \
  --load-model=restructure-pages \
  --allow-metrics=false \
  2>&1 | sed -u 's/^/[triton] /' &
wait_healthy "http://localhost:8000/v2/health/ready" "Triton" 120

# 3. FastAPI gateway
echo "[start] Starting gateway..."
HPS_TRITON_URL="localhost:8001" \
HPS_VLM_URL="http://localhost:8118" \
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=16 \
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=64 \
HPS_INFERENCE_TIMEOUT=600 \
/workspace/.venv_gateway/bin/uvicorn app:app \
  --app-dir /workspace/gateway \
  --host 0.0.0.0 --port 8080 \
  --workers 4 \
  2>&1 | sed -u 's/^/[gateway] /' &
wait_healthy "http://localhost:8080/health" "Gateway" 30

echo ""
echo "========================================="
echo " HPS ready — gateway at :8080"
echo "   POST /layout-parsing"
echo "   POST /restructure-pages"
echo "   GET  /health/ready"
echo "========================================="
wait
```

---

## Key Paths Reference

| Item | Path |
|------|------|
| Triton binary | `/opt/tritonserver/bin/tritonserver` |
| Triton Python backend | `/opt/tritonserver/backends/python/` |
| vLLM + all deps | `/paddlex/py310/` |
| paddleocr CLI | `/paddlex/py310/bin/python3 -m paddleocr` |
| Gateway venv | `/workspace/.venv_gateway/` |
| Gateway app | `/workspace/gateway/app.py` |
| HPS model repo | `/workspace/hps/server/model_repo/` |
| HPS pipeline config | `/workspace/hps/server/pipeline_config.yaml` |
| Models cache | `/workspace/models/hf_cache/` |

## Critical Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PADDLEX_HPS_PIPELINE_CONFIG_PATH` | `/workspace/hps/server/pipeline_config.yaml` | Required for Triton backends |
| `PADDLEX_HPS_DEVICE_TYPE` | `gpu` | Enables GPU inference in Triton |
| `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` | `True` | Skips slow connectivity checks |
| `HF_HOME` | `/workspace/models/hf_cache` | Persists model weights |
| `LD_LIBRARY_PATH` | `/opt/tritonserver/lib:...` | Required for Triton binary |
