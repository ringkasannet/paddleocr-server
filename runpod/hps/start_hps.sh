#!/usr/bin/env bash
# ============================================================
# HPS Startup for paddlex/hps:paddlex3.4-gpu on RunPod
# Run every pod boot: bash /workspace/start_hps.sh
# ============================================================
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────
PADDLE_PY=/paddlex/py310/bin/python3
TRITON_BIN=/opt/tritonserver/bin/tritonserver
TRITON_BACKENDS=/opt/tritonserver/backends
TRITON_LIB=/opt/tritonserver/lib
GATEWAY_BIN=/workspace/.venv_gateway/bin/uvicorn
GATEWAY_APP=/workspace/gateway
MODEL_REPO=/workspace/hps/server/model_repo

# ── Config (override via env vars) ────────────────────────────
MODEL_NAME=${MODEL_NAME:-PaddleOCR-VL-1.5-0.9B}
VLLM_PORT=${VLLM_PORT:-8118}
TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-8000}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-8001}
GATEWAY_PORT=${GATEWAY_PORT:-8080}
UVICORN_WORKERS=${UVICORN_WORKERS:-4}
INIT_TIMEOUT=${INIT_TIMEOUT:-600}

# Kill any existing services and wait for them to fully exit
echo "[start] Stopping existing services..."
pkill -f "tritonserver|uvicorn|genai_server" 2>/dev/null || true
# Wait until all processes are gone (up to 35s for Triton's 30s grace period)
for i in $(seq 1 35); do
    pgrep -f "tritonserver|uvicorn|genai_server" > /dev/null 2>&1 || break
    echo "[start] Waiting for processes to exit... (${i}s)"
    sleep 1
done
echo "[start] All services stopped."

export PYTHONUNBUFFERED=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export HF_HOME=${HF_HOME:-/workspace/models/hf_cache}
export PADDLEX_HPS_PIPELINE_CONFIG_PATH=/workspace/hps/server/pipeline_config.yaml

# Persist model weights and compile cache to network volume
mkdir -p /workspace/models/paddlex /workspace/models/hf_cache /workspace/.cache/vllm
mkdir -p /root/.cache
ln -sfn /workspace/models/paddlex /root/.paddlex
ln -sfn /workspace/.cache/vllm /root/.cache/vllm
export PADDLEX_HPS_DEVICE_TYPE=gpu
export LD_LIBRARY_PATH=$TRITON_LIB:${LD_LIBRARY_PATH:-}

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
VLLM_CFG=/tmp/vllm_backend.yaml
cat > "$VLLM_CFG" << 'EOF'
gpu-memory-utilization: 0.50
EOF
$PADDLE_PY -m paddleocr genai_server \
    --model_name "$MODEL_NAME" \
    --backend vllm \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --backend_config "$VLLM_CFG" \
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
echo "============================================"

wait -n $VLLM_PID $TRITON_PID $GATEWAY_PID
echo "[start] A process exited — shutting down."
kill $VLLM_PID $TRITON_PID $GATEWAY_PID 2>/dev/null || true
