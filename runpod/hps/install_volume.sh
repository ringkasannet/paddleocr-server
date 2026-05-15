#!/usr/bin/env bash
# ============================================================
# Network Volume Installer for paddlex/hps base image
#
# Run ONCE on a pod using ringkasannet/paddleocr-hps:paddlex3.4-gpu
# Everything installs into /workspace (persists across pods).
#
# Usage:
#   bash /workspace/install_volume.sh
# ============================================================
set -euo pipefail

WORKSPACE=/workspace
SDK_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz"

# Python environments from the paddlex/hps base image
PADDLE_PY=/paddlex/py310/bin/python3
PADDLE_PIP=/paddlex/py310/bin/pip

echo "=== Step 1: Downloading HPS SDK ==="
wget -q --show-progress "$SDK_URL" -O /tmp/sdk.tar.gz
tar -xf /tmp/sdk.tar.gz -C /tmp/
mkdir -p $WORKSPACE/hps/server $WORKSPACE/hps/client
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/server/. $WORKSPACE/hps/server/
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/client/. $WORKSPACE/hps/client/
rm /tmp/sdk.tar.gz
echo "  SDK extracted to $WORKSPACE/hps/"

echo "=== Step 2: Patching pipeline_config.yaml ==="
sed -i 's|http://paddleocr-vlm-server:8080/v1|http://localhost:8118/v1|g' \
    $WORKSPACE/hps/server/pipeline_config.yaml
grep server_url $WORKSPACE/hps/server/pipeline_config.yaml
echo "  Patched."

echo "=== Step 3: Creating gateway venv ==="
$PADDLE_PY -m venv $WORKSPACE/.venvs/gateway
$WORKSPACE/.venvs/gateway/bin/pip install --upgrade pip -q
$WORKSPACE/.venvs/gateway/bin/pip install --no-cache-dir \
    fastapi==0.123.6 uvicorn==0.35.0 "tritonclient[grpc]"

# Install gateway requirements
if [ -f $WORKSPACE/gateway/requirements.txt ]; then
    $WORKSPACE/.venvs/gateway/bin/pip install --no-cache-dir \
        -r $WORKSPACE/gateway/requirements.txt
fi

# Install paddlex_hps_client wheel from SDK
if [ -f $WORKSPACE/hps/client/requirements.txt ]; then
    $WORKSPACE/.venvs/gateway/bin/pip install --no-cache-dir \
        -r $WORKSPACE/hps/client/requirements.txt
fi
WHL=$(ls $WORKSPACE/hps/client/paddlex_hps_client-*.whl 2>/dev/null | head -1)
if [ -n "$WHL" ]; then
    $WORKSPACE/.venvs/gateway/bin/pip install --no-cache-dir "$WHL"
    echo "  Installed: $(basename $WHL)"
fi
echo "  Gateway venv ready."

echo "=== Step 4: Copying gateway app ==="
# Gateway app.py lives in the PaddleOCR repo — copy to workspace
REPO_GATEWAY=/workspace/PaddleOCR/deploy/paddleocr_vl_docker/hps/gateway
if [ -d "$REPO_GATEWAY" ]; then
    mkdir -p $WORKSPACE/gateway
    cp -r $REPO_GATEWAY/. $WORKSPACE/gateway/
    echo "  Copied from repo."
else
    echo "  WARNING: gateway app not found at $REPO_GATEWAY"
    echo "  Upload gateway/app.py manually to $WORKSPACE/gateway/"
fi

echo "=== Step 5: Writing start_hps.sh ==="
cat > $WORKSPACE/start_hps.sh << 'STARTSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

# Paths from paddlex/hps base image
PADDLE_BIN=/paddlex/py310/bin
TRITON_BIN=/opt/tritonserver/bin/tritonserver
TRITON_BACKENDS=/opt/tritonserver/backends
TRITON_LIB=/opt/tritonserver/lib
GATEWAY_BIN=/workspace/.venvs/gateway/bin
GATEWAY_APP=/workspace/gateway
MODEL_REPO=/workspace/hps/server/model_repo

MODEL_NAME=${MODEL_NAME:-PaddleOCR-VL-1.5-0.9B}
GPU_MEM_UTIL=${GPU_MEMORY_UTILIZATION:-0.50}
VLLM_PORT=${VLLM_PORT:-8118}
TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-8000}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-8001}
GATEWAY_PORT=${GATEWAY_PORT:-8080}
INIT_TIMEOUT=${INIT_TIMEOUT:-600}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/workspace/models/hf_cache}
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export LD_LIBRARY_PATH=$TRITON_LIB:${LD_LIBRARY_PATH:-}

wait_healthy() {
    local url=$1 label=$2 timeout=$3
    local deadline=$((SECONDS + timeout))
    while [[ $SECONDS -lt $deadline ]]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "[start] $label ready."
            return 0
        fi
        echo "[start] Waiting for $label... (${SECONDS}s)"
        sleep 10
    done
    echo "[ERROR] $label did not become healthy within ${timeout}s"
    exit 1
}

# 1. vLLM server
echo "[start] Starting vLLM server (port $VLLM_PORT)..."
VLLM_CFG=/tmp/vllm_backend.yaml
echo "gpu-memory-utilization: $GPU_MEM_UTIL" > $VLLM_CFG
PATH="$PADDLE_BIN:$PATH" "$PADDLE_BIN/paddleocr" genai_server \
    --model_name "$MODEL_NAME" \
    --backend vllm \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --backend_config "$VLLM_CFG" \
    2>&1 | sed -u 's/^/[vllm] /' &
VLLM_PID=$!
wait_healthy "http://localhost:$VLLM_PORT/health" "vLLM" "$INIT_TIMEOUT"

# 2. Triton server
echo "[start] Starting Triton server..."
"$TRITON_BIN" \
    --model-repository="$MODEL_REPO" \
    --backend-directory="$TRITON_BACKENDS" \
    --backend-config=python,python-runtime="$PADDLE_BIN/python3" \
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

# 3. FastAPI gateway
echo "[start] Starting gateway (port $GATEWAY_PORT)..."
HPS_TRITON_URL="localhost:$TRITON_GRPC_PORT" \
HPS_VLM_URL="http://localhost:$VLLM_PORT" \
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=${HPS_MAX_CONCURRENT_INFERENCE_REQUESTS:-16} \
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=${HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS:-64} \
HPS_INFERENCE_TIMEOUT=600 \
"$GATEWAY_BIN/uvicorn" app:app \
    --app-dir "$GATEWAY_APP" \
    --host 0.0.0.0 \
    --port "$GATEWAY_PORT" \
    --workers ${UVICORN_WORKERS:-4} \
    2>&1 | sed -u 's/^/[gateway] /' &
GATEWAY_PID=$!
wait_healthy "http://localhost:$GATEWAY_PORT/health" "Gateway" 30

echo ""
echo "============================================"
echo " HPS running — gateway at :$GATEWAY_PORT"
echo "============================================"

wait -n $VLLM_PID $TRITON_PID $GATEWAY_PID
kill $VLLM_PID $TRITON_PID $GATEWAY_PID 2>/dev/null || true
STARTSCRIPT
chmod +x $WORKSPACE/start_hps.sh

echo ""
echo "============================================"
echo " Install complete."
echo " Run: bash /workspace/start_hps.sh"
echo "============================================"
