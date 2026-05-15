#!/usr/bin/env bash
# Build and push the pre-installed HPS image (with model weights baked in)
#
# Run from WSL:
#   cd /mnt/d/paddleocr/PaddleOCR-main/runpod/hps
#   bash build_push.sh [tag]
#
# Requires Docker BuildKit (default in Docker Desktop 20.10+).
# Model weights are read directly from your local paddlex cache —
# no need to copy 2.2 GB to the Windows filesystem.

set -euo pipefail

IMAGE="ringkasannet/paddleocr-hps"
TAG="${1:-paddlex3.4-gpu-ready}"
MODELS_DIR="${PADDLEX_MODELS:-$HOME/.paddlex/official_models}"

if [ ! -d "$MODELS_DIR/PaddleOCR-VL" ]; then
    echo "[build] ERROR: Model weights not found at $MODELS_DIR/PaddleOCR-VL"
    echo "[build] Run the services once on a pod to download them, then copy here:"
    echo "[build]   scp -r root@<pod>:/root/.paddlex/official_models ~/.paddlex/"
    exit 1
fi

echo "[build] Building $IMAGE:$TAG"
echo "[build] Models from: $MODELS_DIR ($(du -sh "$MODELS_DIR" | cut -f1))"

DOCKER_BUILDKIT=1 docker build \
    --build-context paddlex_models="$MODELS_DIR" \
    -f Dockerfile.hps \
    -t "$IMAGE:$TAG" \
    --progress=plain \
    .

echo "[build] Pushing $IMAGE:$TAG ..."
docker push "$IMAGE:$TAG"

echo ""
echo "Done: $IMAGE:$TAG"
echo "Pod image: $IMAGE:$TAG"
echo "Start command: bash /opt/start_hps.sh"
