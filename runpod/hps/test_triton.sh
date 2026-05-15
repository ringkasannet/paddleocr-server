#!/usr/bin/env bash
# ============================================================
# Triton Portability Test
#
# 1. Checks Triton version inside paddlex/hps image
# 2. Extracts tritonserver binary to /tmp/triton_test/
# 3. Runs the binary inside the RunPod PyTorch image
#    to verify portability WITHOUT needing a GPU
#
# Usage (WSL):
#   bash test_triton.sh [runpod-image]
#
# Example:
#   bash test_triton.sh runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# ============================================================
set -euo pipefail

HPS_IMAGE="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.4-gpu"
RUNPOD_IMAGE="${1:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
TEST_DIR="/tmp/triton_test"

echo "============================================"
echo " Step 1: Triton version in paddlex/hps"
echo "============================================"
docker run --rm "$HPS_IMAGE" tritonserver --version
echo ""

echo "============================================"
echo " Step 2: Extracting tritonserver binary"
echo "============================================"
rm -rf "$TEST_DIR" && mkdir -p "$TEST_DIR"
docker create --name triton_test_tmp "$HPS_IMAGE" bash
docker cp triton_test_tmp:/opt/tritonserver "$TEST_DIR/tritonserver"
docker rm triton_test_tmp
echo "Extracted to $TEST_DIR/tritonserver"
echo "Size: $(du -sh $TEST_DIR/tritonserver | cut -f1)"
echo ""

echo "============================================"
echo " Step 3: Pulling RunPod image"
echo "============================================"
docker pull "$RUNPOD_IMAGE"
echo ""

echo "============================================"
echo " Step 4: Testing binary inside RunPod image"
echo "============================================"
echo "Running: tritonserver --version inside $RUNPOD_IMAGE"
docker run --rm \
    -v "$TEST_DIR/tritonserver:/opt/tritonserver" \
    "$RUNPOD_IMAGE" \
    bash -c "
        export LD_LIBRARY_PATH=/opt/tritonserver/lib:\${LD_LIBRARY_PATH:-}
        echo '--- ldd check ---'
        ldd /opt/tritonserver/bin/tritonserver 2>&1 | grep -E 'not found|libcuda|libcudart' || true
        echo ''
        echo '--- version ---'
        /opt/tritonserver/bin/tritonserver --version
    "

echo ""
echo "============================================"
echo " Done"
echo "============================================"
echo "If 'version' printed above without errors, the binary is portable."
echo "If you see 'not found' in ldd output, those libs are missing from RunPod image."
