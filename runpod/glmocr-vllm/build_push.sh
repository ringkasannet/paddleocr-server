#!/usr/bin/env bash
# Build and push the GLM-OCR vLLM RunPod serverless worker.
#
# Run from WSL:
#   cd /mnt/d/paddleocr/PaddleOCR-main/runpod/glmocr-vllm
#   bash build_push.sh [tag]
#
# RunPod endpoint setup:
#   Cached model : zai-org/GLM-OCR
#   GPU          : RTX 3090 or L40S (needs ≥20 GB VRAM)
#   Env vars     : GPU_MEM_UTIL=0.90 (optional)

set -euo pipefail

IMAGE="ringkasannet/glmocr-vllm-worker"
TAG="${1:-v1}"

echo "=== Building $IMAGE:$TAG ==="
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile \
    -t "$IMAGE:$TAG" \
    --progress=plain \
    .

echo "=== Pushing $IMAGE:$TAG ==="
docker push "$IMAGE:$TAG"

echo ""
echo "Done: $IMAGE:$TAG"
echo ""
echo "RunPod endpoint settings:"
echo "  Image        : $IMAGE:$TAG"
echo "  Cached model : zai-org/GLM-OCR"
echo "  GPU          : RTX 3090 / L40S (≥20 GB VRAM)"
