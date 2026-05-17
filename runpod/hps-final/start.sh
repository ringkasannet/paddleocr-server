#!/usr/bin/env bash
# ============================================================
# HPS Startup for paddlex3.4-gpu-ready image
# Everything is baked into the image: Python packages, HPS SDK,
# gateway venv, gateway app, and model weights.
# Network volume (/workspace) is OPTIONAL — used for vLLM CUDA
# cache persistence only (saves ~3-5 min on recompile).
#
# Run every pod boot:
#   bash /workspace/paddleocr-server/runpod/hps/start_hps_v2.sh
# ============================================================
set -euo pipefail

# Propagate Ctrl+C / docker stop to all child processes
_shutdown() { echo "[start] Shutting down..."; kill 0; exit 0; }
trap _shutdown INT TERM

# ── Paths — prefer /workspace overrides, fall back to image defaults ──────────
PADDLE_PY=/paddlex/py310/bin/python3
TRITON_BIN=/opt/tritonserver/bin/tritonserver
TRITON_BACKENDS=/opt/tritonserver/backends
TRITON_LIB=/opt/tritonserver/lib

# SDK: workspace copy allows config customisation without rebuilding image
if [ -d /workspace/hps/server/model_repo ]; then
    MODEL_REPO=/workspace/hps/server/model_repo
    PIPELINE_CONFIG=/workspace/hps/server/pipeline_config.yaml
else
    MODEL_REPO=/opt/hps/server/model_repo
    PIPELINE_CONFIG=/opt/hps/server/pipeline_config.yaml
fi

# Gateway: workspace copy allows hot-patching app.py without rebuilding image
if [ -f /workspace/.venv_gateway/bin/uvicorn ]; then
    GATEWAY_BIN=/workspace/.venv_gateway/bin/uvicorn
else
    GATEWAY_BIN=/opt/.venv_gateway/bin/uvicorn
fi
if [ -d /workspace/gateway ] && [ -f /workspace/gateway/app.py ]; then
    GATEWAY_APP=/workspace/gateway
else
    GATEWAY_APP=/opt/gateway
fi

# ── Config (override via env vars) ────────────────────────────
MODEL_NAME=${MODEL_NAME:-PaddleOCR-VL-1.5-0.9B}
VLLM_PORT=${VLLM_PORT:-8118}
TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-8000}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-8001}
GATEWAY_PORT=${GATEWAY_PORT:-8080}
UVICORN_WORKERS=${UVICORN_WORKERS:-4}
INIT_TIMEOUT=${INIT_TIMEOUT:-600}

# ── Stop existing services ────────────────────────────────────
# Empty string hides all GPUs from CUDA; unset lets CUDA see all available devices
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    unset CUDA_VISIBLE_DEVICES
fi

echo "[start] Stopping existing services..."
pkill -f "tritonserver|uvicorn|genai_server" 2>/dev/null || true
for i in $(seq 1 35); do
    pgrep -f "tritonserver|uvicorn|genai_server" > /dev/null 2>&1 || break
    echo "[start] Waiting for processes to exit... (${i}s)"
    sleep 1
done
echo "[start] All services stopped."

# ── Environment ───────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export PADDLEX_HPS_PIPELINE_CONFIG_PATH=$PIPELINE_CONFIG
LAYOUT_DEVICE=${LAYOUT_DEVICE:-gpu}
export PADDLEX_HPS_DEVICE_TYPE=$LAYOUT_DEVICE

# Low-VRAM mode: move layout detection to CPU (e.g. RTX 3070 8 GB)
# Set LAYOUT_DEVICE=cpu to enable. Copies config to /tmp so image is unchanged.
if [ "$LAYOUT_DEVICE" = "cpu" ]; then
    echo "[start] Low-VRAM mode: layout detection on CPU"
    LAYOUT_CONFIG=/tmp/layout_config_cpu.pbtxt
    cp "$MODEL_REPO/layout-parsing/config.pbtxt" "$LAYOUT_CONFIG"
    sed -i 's/kind: KIND_GPU/kind: KIND_CPU/' "$LAYOUT_CONFIG"
    sed -i '/gpus:/d' "$LAYOUT_CONFIG"
    cp "$LAYOUT_CONFIG" "$MODEL_REPO/layout-parsing/config.pbtxt"
fi
export LD_LIBRARY_PATH=$TRITON_LIB:${LD_LIBRARY_PATH:-}

# ── Model weights: image has them at /root/.paddlex/official_models/ ──────────
# Only redirect to network volume if it already has models (legacy setup)
if [ -d /workspace/models/paddlex ] && [ "$(ls -A /workspace/models/paddlex 2>/dev/null)" ]; then
    echo "[start] Using model weights from network volume"
    ln -sfn /workspace/models/paddlex /root/.paddlex
else
    echo "[start] Using model weights from image (/root/.paddlex)"
fi

# ── vLLM CUDA cache: persist to network volume if available ──────────────────
# This avoids ~3-5 min recompile on every pod restart (optional but recommended)
if [ -d /workspace ] || mkdir -p /workspace 2>/dev/null; then
    mkdir -p /workspace/.cache/vllm
    mkdir -p /root/.cache
    ln -sfn /workspace/.cache/vllm /root/.cache/vllm
fi

wait_healthy() {
    local url=$1 label=$2 timeout=$3
    local t_start=$SECONDS
    local deadline=$((SECONDS + timeout))
    while [[ $SECONDS -lt $deadline ]]; do
        if wget -qO- "$url" > /dev/null 2>&1; then
            echo "[start] $label ready in $((SECONDS - t_start))s"
            return 0
        fi
        echo "[start] Waiting for $label... (${SECONDS}s)"
        sleep 10
    done
    echo "[ERROR] $label did not become healthy within ${timeout}s"
    exit 1
}

T_TOTAL=$SECONDS

# ── 1. vLLM ───────────────────────────────────────────────────
echo "[start] Starting vLLM (port $VLLM_PORT)..."
T_VLLM=$SECONDS
GPU_MEM_UTIL=${GPU_MEMORY_UTILIZATION:-0.50}
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}

{
  echo "gpu-memory-utilization: ${GPU_MEM_UTIL}"
  echo "enable-prefix-caching: false"
  echo "mm-processor-cache-gb: 0"
  [ -n "$MAX_MODEL_LEN" ]           && echo "max-model-len: ${MAX_MODEL_LEN}"
  [ -n "$MAX_NUM_BATCHED_TOKENS" ]  && echo "max-num-batched-tokens: ${MAX_NUM_BATCHED_TOKENS}"
  [ "$ENFORCE_EAGER" = "true" ]     && echo "enforce-eager: true"
} > /tmp/vllm_backend.yaml
echo "[start] vllm_backend.yaml:"; cat /tmp/vllm_backend.yaml
$PADDLE_PY -m paddleocr genai_server \
    --model_name "$MODEL_NAME" \
    --backend vllm \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --backend_config /tmp/vllm_backend.yaml \
    2>&1 | sed -u 's/^/[vllm] /' &
VLLM_PID=$!
wait_healthy "http://localhost:$VLLM_PORT/health" "vLLM" "$INIT_TIMEOUT"
echo "[start] vLLM load time: $((SECONDS - T_VLLM))s"

# ── 2. Triton ─────────────────────────────────────────────────
echo "[start] Starting Triton..."
T_TRITON=$SECONDS
"$TRITON_BIN" \
    --model-repository="$MODEL_REPO" \
    --backend-directory="$TRITON_BACKENDS" \
    --backend-config=python,python-runtime="$PADDLE_PY" \
    --backend-config=python,shm-default-byte-size=67108864 \
    --http-port="$TRITON_HTTP_PORT" \
    --grpc-port="$TRITON_GRPC_PORT" \
    --model-control-mode=explicit \
    --load-model=layout-parsing \
    --load-model=restructure-pages \
    --allow-metrics=false \
    2>&1 | sed -u 's/^/[triton] /' &
TRITON_PID=$!
wait_healthy "http://localhost:$TRITON_HTTP_PORT/v2/health/ready" "Triton" 120
echo "[start] Triton load time: $((SECONDS - T_TRITON))s"

# ── 3. Gateway ────────────────────────────────────────────────
echo "[start] Starting gateway (port $GATEWAY_PORT)..."
T_GATEWAY=$SECONDS
HPS_TRITON_URL="localhost:$TRITON_GRPC_PORT" \
HPS_VLM_URL="http://localhost:$VLLM_PORT" \
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=${HPS_MAX_CONCURRENT_INFERENCE_REQUESTS:-16} \
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=${HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS:-64} \
HPS_INFERENCE_TIMEOUT=${HPS_INFERENCE_TIMEOUT:-600} \
"$GATEWAY_BIN" app:app \
    --app-dir "$GATEWAY_APP" \
    --host 0.0.0.0 \
    --port "$GATEWAY_PORT" \
    --workers "$UVICORN_WORKERS" \
    2>&1 | sed -u 's/^/[gateway] /' &
GATEWAY_PID=$!
wait_healthy "http://localhost:$GATEWAY_PORT/health" "Gateway" 60
echo "[start] Gateway load time: $((SECONDS - T_GATEWAY))s"

echo ""
echo "============================================"
echo " HPS ready in $((SECONDS - T_TOTAL))s total"
echo "   Gateway: http://localhost:$GATEWAY_PORT"
echo "   Triton:  http://localhost:$TRITON_HTTP_PORT"
echo "   vLLM:    http://localhost:$VLLM_PORT"
echo "   SDK:     $MODEL_REPO"
echo "   Gateway: $GATEWAY_APP ($GATEWAY_BIN)"
echo "============================================"

wait -n $VLLM_PID $TRITON_PID $GATEWAY_PID
echo "[start] A process exited — shutting down."
kill $VLLM_PID $TRITON_PID $GATEWAY_PID 2>/dev/null || true
