#!/usr/bin/env bash
# Build and push the GLM-OCR RunPod serverless worker image.
# Model weights are NOT baked in — loaded at runtime from RunPod model cache.
#
# Run from WSL:
#   cd /mnt/d/paddleocr/PaddleOCR-main/runpod/glm-ocr
#   bash build_push.sh [tag]
#
# Requires Docker BuildKit (default in Docker Desktop 20.10+).

set -euo pipefail

IMAGE="ringkasannet/glm-ocr-worker"
TAG="${1:-latest}"

echo "[build] Building $IMAGE:$TAG"

DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile \
    -t "$IMAGE:$TAG" \
    --progress=plain \
    .

echo "[build] Pushing $IMAGE:$TAG ..."
docker push "$IMAGE:$TAG"

echo ""
echo "Done: $IMAGE:$TAG"
echo ""
echo "RunPod endpoint settings:"
echo "  Image          : $IMAGE:$TAG"
echo "  Model cache    : zai-org/GLM-OCR"
echo "  Env vars       : HF_TOKEN=hf_xxx"
echo "                   GPU_MEM_UTIL=0.65  (optional)"
echo "                   MAX_MODEL_LEN=4096  (optional)"
echo "                   MAX_TOKENS=2048     (optional)"
