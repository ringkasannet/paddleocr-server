#!/usr/bin/env bash
# Build and push the GLM-OCR RunPod serverless worker.
# Model weights are NOT baked in — served from RunPod model cache.
#
# Run from WSL:
#   cd /mnt/d/paddleocr/PaddleOCR-main/runpod/glmocr-server
#   bash build_push.sh [tag]
#
# RunPod endpoint settings:
#   Image          : ringkasannet/glmocr-server-worker:<tag>
#   Cached models  : zai-org/GLM-OCR
#                    PaddlePaddle/PP-DocLayoutV3_safetensors
#   GPU            : L40S / A100 40GB+ (vLLM ~7 GB + layout ~2 GB + overhead)
#   Env vars       : GPU_MEM_UTIL=0.60   (default; raise if GPU has more VRAM)
#                    MAX_MODEL_LEN=4096
#                    MAX_TOKENS=2048
#                    ENABLE_MTP=1        (speculative decoding, default on)

set -euo pipefail

IMAGE="ringkasannet/glmocr-server-worker"
TAG="${1:-v1}"

echo "[build] Building $IMAGE:$TAG"
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile \
    -t "$IMAGE:$TAG" \
    --progress=plain \
    .

echo "[build] Pushing $IMAGE:$TAG …"
docker push "$IMAGE:$TAG"

echo ""
echo "Done: $IMAGE:$TAG"
echo ""
echo "RunPod endpoint settings:"
echo "  Image          : $IMAGE:$TAG"
echo "  Cached model 1 : zai-org/GLM-OCR"
echo "  Cached model 2 : PaddlePaddle/PP-DocLayoutV3_safetensors"
echo "  GPU            : L40S / A100 40 GB+"
echo "  Env vars       : GPU_MEM_UTIL=0.60"
echo "                   MAX_MODEL_LEN=4096"
echo "                   MAX_TOKENS=2048"
echo "                   ENABLE_MTP=1"
echo ""
echo "Request format:"
echo '  {"input": {"images": ["data:application/pdf;base64,<b64>"]}}'
