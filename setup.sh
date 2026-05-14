#!/usr/bin/env bash
# ============================================================
# Fresh pod setup for PaddleOCR HPS
# Image: ringkasannet/paddleocr-hps:paddlex3.4-gpu
#
# Run once on a fresh /workspace:
#   bash /workspace/setup.sh
# ============================================================
# ssh root@213.173.98.71 -p 19009 -i ~/.ssh/id_ed25519
# scp -P 19009 /mnt/d/paddleocr/PaddleOCR-main/runpod/hps/setup.sh \
#   root@213.173.98.71:/workspace/setup.sh

set -euo pipefail

WORKSPACE=/workspace
SDK_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz"
PADDLE_PY=/paddlex/py310/bin/python3
PADDLE_PIP=/paddlex/py310/bin/pip

log() { echo "[setup $(date +%H:%M:%S)] $*"; }

# ── 0. System packages (fast, other tasks need wget) ──────────────
log "System packages..."
apt-get update -qq && apt-get install -y wget -qq

# ── 1. Parallel phase ─────────────────────────────────────────────
log "Starting parallel tasks A / B / C..."

# ── Task A: HPS SDK ───────────────────────────────────────────────
(
  set -euo pipefail
  log "[A] Downloading HPS SDK..."
  wget -q "$SDK_URL" -O /tmp/sdk.tar.gz
  tar -xf /tmp/sdk.tar.gz -C /tmp/
  mkdir -p $WORKSPACE/hps/server $WORKSPACE/hps/client
  cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/server/. $WORKSPACE/hps/server/
  cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/client/. $WORKSPACE/hps/client/
  rm /tmp/sdk.tar.gz

  # Patch pipeline_config.yaml: use vllm-server instead of native
  $PADDLE_PY - << 'PYEOF'
import yaml
path = "/workspace/hps/server/pipeline_config.yaml"
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg["SubModules"]["VLRecognition"]["genai_config"] = {
    "backend": "vllm-server",
    "server_url": "http://localhost:8118/v1",
    "max_concurrency": 16,
}
with open(path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print("[A] pipeline_config.yaml: backend set to vllm-server, max_concurrency=16")
PYEOF

  # Fix missing config.pbtxt and tune Triton batching
  cp $WORKSPACE/hps/server/model_repo/layout-parsing/config_gpu.pbtxt \
     $WORKSPACE/hps/server/model_repo/layout-parsing/config.pbtxt
  sed -i 's/max_batch_size: 8/max_batch_size: 16/' \
    $WORKSPACE/hps/server/model_repo/layout-parsing/config.pbtxt
  sed -i 's/count: 1/count: 2/' \
    $WORKSPACE/hps/server/model_repo/layout-parsing/config.pbtxt
  log "[A] Done — SDK + configs ready"
) > /tmp/setup_A.log 2>&1 &
PID_A=$!

# ── Task B: Python deps ───────────────────────────────────────────
(
  set -euo pipefail
  export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

  # flash-attn: pre-built wheel avoids CUDA 11.8 vs torch 2.8 mismatch
  log "[B] Installing flash-attn..."
  $PADDLE_PIP install --no-cache-dir -q \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"

  # Install paddleocr into /paddlex/py310/ first so we can call its CLI
  log "[B] Installing paddleocr 3.5.0..."
  $PADDLE_PIP install -q paddleocr==3.5.0

  # install_genai_server_deps is hardcoded to use /paddlex/py310/bin/python
  log "[B] Installing vllm deps..."
  /paddlex/py310/bin/paddleocr install_genai_server_deps vllm

  # matplotlib must be >= 3.9 to avoid numpy 2.x binary conflict
  log "[B] Upgrading matplotlib..."
  $PADDLE_PIP install -q "matplotlib>=3.9" --upgrade

  # paddlex[ocr] extras required for VLM pipeline
  PADDLEX_VER=$($PADDLE_PY -c "import paddlex; print(paddlex.__version__)")
  log "[B] Installing paddlex[ocr]==$PADDLEX_VER..."
  $PADDLE_PIP install -q "paddlex[ocr]==$PADDLEX_VER"

  log "[B] Done — Python deps ready"
) > /tmp/setup_B.log 2>&1 &
PID_B=$!

# ── Task C: Gateway app ───────────────────────────────────────────
(
  set -euo pipefail
  log "[C] Downloading gateway app..."
  mkdir -p $WORKSPACE/gateway
  wget -q https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/paddleocr_vl_docker/hps/gateway/app.py \
    -O $WORKSPACE/gateway/app.py
  log "[C] Done — gateway app ready"
) > /tmp/setup_C.log 2>&1 &
PID_C=$!

# Stream all three logs in parallel while waiting
tail -f /tmp/setup_A.log /tmp/setup_B.log /tmp/setup_C.log &
TAIL_PID=$!

# Wait for all three tasks
FAILED=0
wait $PID_A && log "Task A finished" || { log "Task A FAILED — see /tmp/setup_A.log"; FAILED=1; }
wait $PID_B && log "Task B finished" || { log "Task B FAILED — see /tmp/setup_B.log"; FAILED=1; }
wait $PID_C && log "Task C finished" || { log "Task C FAILED — see /tmp/setup_C.log"; FAILED=1; }

kill $TAIL_PID 2>/dev/null || true

[ $FAILED -eq 1 ] && { log "One or more tasks failed. Aborting."; exit 1; }

# ── 2. Gateway venv (needs SDK from Task A) ───────────────────────
log "Setting up gateway venv..."
if [ ! -f $WORKSPACE/.venv_gateway/bin/python3 ]; then
  $PADDLE_PY -m venv $WORKSPACE/.venv_gateway
fi
$WORKSPACE/.venv_gateway/bin/pip install --upgrade pip -q
$WORKSPACE/.venv_gateway/bin/pip install --no-cache-dir -q \
  fastapi==0.123.6 uvicorn==0.35.0 "paddlex[serving]>=3.4.0"
$WORKSPACE/.venv_gateway/bin/pip install --no-cache-dir -q \
  -r $WORKSPACE/hps/client/requirements.txt
WHL=$(ls $WORKSPACE/hps/client/paddlex_hps_client-*.whl | head -1)
$WORKSPACE/.venv_gateway/bin/pip install --no-cache-dir -q "$WHL"
log "Gateway venv ready"

# ── 3. numpy fix — MUST BE LAST (flash-attn/vllm may upgrade it) ──
log "Pinning numpy to 1.26.4..."
$PADDLE_PIP install -q "numpy==1.26.4" --force-reinstall --no-deps
log "numpy pinned"

# ── Done ──────────────────────────────────────────────────────────
log ""
log "================================================"
log " Setup complete!"
log " Start services: bash /workspace/start_hps.sh"
log "================================================"
