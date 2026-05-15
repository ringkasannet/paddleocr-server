#!/usr/bin/env bash
# ============================================================
# HPS First-Run Installer
#
# Run ONCE on a RunPod pod with the network volume mounted.
# Installs all Python dependencies into /workspace/.venvs/.
# After this completes, every subsequent pod start only needs
# start_hps.sh — no reinstallation required.
#
# Usage:
#   bash /workspace/install.sh
# ============================================================
set -euo pipefail

WORKSPACE=/workspace
VENVS=$WORKSPACE/.venvs
SDK_CLIENT=$WORKSPACE/hps/client

# Detect CUDA version to select the right PaddlePaddle wheel index
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | tr -d '.')
if [[ "$CUDA_VER" -ge 126 ]]; then
    PP_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
elif [[ "$CUDA_VER" -ge 123 ]]; then
    PP_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu123/"
else
    PP_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu118/"
fi
echo "[install] CUDA $CUDA_VER detected → using index $PP_INDEX"

# ── System packages ───────────────────────────────────────────
echo "[install] Installing system packages..."
apt-get update -qq && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    > /dev/null

# ── paddle venv (vLLM server + Triton Python backends) ────────
echo "[install] Creating paddle venv..."
python3 -m venv "$VENVS/paddle"
"$VENVS/paddle/bin/pip" install --upgrade pip setuptools wheel -q

echo "[install] Installing PaddlePaddle GPU..."
"$VENVS/paddle/bin/pip" install --no-cache-dir paddlepaddle-gpu==3.2.1 \
    -i "$PP_INDEX"

echo "[install] Installing paddleocr + paddlex..."
"$VENVS/paddle/bin/pip" install --no-cache-dir "paddleocr[doc-parser]" paddlex

echo "[install] paddle venv ready."

# ── gateway venv ──────────────────────────────────────────────
echo "[install] Creating gateway venv..."
python3 -m venv "$VENVS/gateway"
"$VENVS/gateway/bin/pip" install --upgrade pip -q

echo "[install] Installing gateway Python deps..."
"$VENVS/gateway/bin/pip" install --no-cache-dir \
    -r "$WORKSPACE/gateway/requirements.txt"

if [[ -f "$SDK_CLIENT/requirements.txt" ]]; then
    "$VENVS/gateway/bin/pip" install --no-cache-dir \
        -r "$SDK_CLIENT/requirements.txt"
fi

WHL=$(ls "$SDK_CLIENT"/paddlex_hps_client-*.whl 2>/dev/null | head -1)
if [[ -n "$WHL" ]]; then
    echo "[install] Installing paddlex_hps_client wheel: $(basename $WHL)"
    "$VENVS/gateway/bin/pip" install --no-cache-dir "$WHL"
else
    echo "[install] WARNING: paddlex_hps_client wheel not found in $SDK_CLIENT"
fi

echo "[install] gateway venv ready."

# ── Copy pipeline config into model repo ──────────────────────
echo "[install] Copying pipeline_config_local.yaml into model repo..."
cp "$WORKSPACE/hps/pipeline_config_local.yaml" \
   "$WORKSPACE/hps/server/pipeline_config.yaml"

# ── Fix permissions ───────────────────────────────────────────
chmod +x "$WORKSPACE/start_hps.sh"
chmod +x "$WORKSPACE/tritonserver/bin/tritonserver" 2>/dev/null || true

echo ""
echo "============================================"
echo " Install complete. Run to start:"
echo "   bash /workspace/start_hps.sh"
echo "============================================"
