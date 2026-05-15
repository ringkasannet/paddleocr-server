#!/usr/bin/env bash
# Runs as root (USER root in Dockerfile, matching official compose.yaml)
set -euo pipefail
export PYTHONUNBUFFERED=1   # flush Python stdout/stderr immediately

GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.65}"
VLLM_PORT="${VLLM_PORT:-8118}"
PADDLE_PORT="${PADDLE_PORT:-8080}"
MODEL_NAME="${MODEL_NAME:-PaddleOCR-VL-1.5-0.9B}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-/workspace/PaddleOCR-VL.yaml}"

echo "================================================================"
echo " PaddleOCR RunPod container starting"
echo "  vLLM genai server  -> :${VLLM_PORT}"
echo "  PaddleX pipeline   -> :${PADDLE_PORT}"
echo "  Model              -> ${MODEL_NAME}"
echo "================================================================"

# ── 1. Start vLLM genai server (provided by base image) ──────
echo "[1/2] Starting vLLM genai server..."

# gpu-memory-utilization must be passed via --backend_config, not as a CLI flag
cat > /tmp/vllm_backend.yaml << EOF
gpu-memory-utilization: ${GPU_MEM_UTIL}
EOF

paddleocr genai_server \
    --model_name "${MODEL_NAME}" \
    --backend vllm \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --backend_config /tmp/vllm_backend.yaml \
    2>&1 | sed -u 's/^/[vllm] /' &
VLLM_PID=${PIPESTATUS[0]}

# ── 2. Wait for vLLM health ───────────────────────────────────
echo "[wait] Waiting for vLLM on port ${VLLM_PORT} (model download may take a few minutes)..."
MAX_WAIT=600
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM exited unexpectedly."
        exit 1
    fi
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "[ERROR] vLLM did not become healthy within ${MAX_WAIT}s."
        kill "${VLLM_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 10
    ELAPSED=$(( ELAPSED + 10 ))
    echo "[wait] ${ELAPSED}s elapsed..."
done
echo "[wait] vLLM is healthy."

# ── 3. Start PaddleX pipeline (system paddleocr-vl env) ──────
echo "[2/2] Starting PaddleX pipeline server..."
/workspace/.paddleocr/bin/paddlex --serve \
    --pipeline "${PIPELINE_CONFIG}" \
    --host 0.0.0.0 \
    --port "${PADDLE_PORT}" \
    2>&1 | sed -u 's/^/[paddlex] /' &
PADDLE_PID=$!

echo "================================================================"
echo " Both services running"
echo "  vLLM    PID=${VLLM_PID}  http://0.0.0.0:${VLLM_PORT}"
echo "  PaddleX PID=${PADDLE_PID} http://0.0.0.0:${PADDLE_PORT}"
echo "================================================================"

# ── 4. Exit if either service dies ───────────────────────────
while true; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM (PID=${VLLM_PID}) died. Stopping container."
        kill "${PADDLE_PID}" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "${PADDLE_PID}" 2>/dev/null; then
        echo "[ERROR] PaddleX (PID=${PADDLE_PID}) died. Stopping container."
        kill "${VLLM_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 30
done
