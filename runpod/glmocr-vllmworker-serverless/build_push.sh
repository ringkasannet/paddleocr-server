#!/usr/bin/env bash
# Build and push the GLM-OCR direct-Pipeline RunPod serverless worker.
#
# Usage:
#   bash build_push.sh [tag]              # model cache mode (default)
#   bash build_push.sh [tag] --bake       # bake weights into image (needs HF_TOKEN env var)
#
# RunPod endpoint settings (model cache mode):
#   Image       : ringkasannet/glm-ocr-pipeline-worker:<tag>
#   Model cache : zai-org/GLM-OCR
#   Env vars    : (none required; optionals below)
#                 HF_TOKEN=hf_xxx         (if model is private / gated)
#                 GPU_MEM_UTIL=0.80
#                 MAX_MODEL_LEN=8192
#                 MAX_TOKENS=2048
#
# RunPod endpoint settings (baked image mode):
#   Same as above PLUS:
#                 HF_HOME=/app/hf_cache
#                 HUGGINGFACE_HUB_CACHE=/app/hf_cache/hub
#                 TRANSFORMERS_CACHE=/app/hf_cache/hub

set -euo pipefail

IMAGE="ringkasannet/glm-ocr-pipeline-worker"
TAG="${1:-latest}"
BAKE="${2:-}"

echo "[build] Building $IMAGE:$TAG"

if [ "$BAKE" = "--bake" ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[build] ERROR: HF_TOKEN env var required for --bake"
        exit 1
    fi
    echo "[build] Bake mode — weights will be embedded in image"
    DOCKER_BUILDKIT=1 docker build \
        -f Dockerfile \
        -t "$IMAGE:$TAG" \
        --build-arg BAKE_MODEL=1 \
        --build-arg HF_TOKEN="$HF_TOKEN" \
        --progress=plain \
        .
else
    DOCKER_BUILDKIT=1 docker build \
        -f Dockerfile \
        -t "$IMAGE:$TAG" \
        --progress=plain \
        .
fi

echo "[build] Pushing $IMAGE:$TAG ..."
docker push "$IMAGE:$TAG"

echo ""
echo "Done: $IMAGE:$TAG"
