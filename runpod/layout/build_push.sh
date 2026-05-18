#!/usr/bin/env bash
# Build and push both layout worker variants for cold-start comparison.
#
# Run from WSL:
#   cd /mnt/d/paddleocr/PaddleOCR-main/runpod/layout
#   bash build_push.sh [tag]
#
# Produces two images:
#   ringkasannet/layout-worker:<tag>-baked   (Option A — weights in image)
#   ringkasannet/layout-worker:<tag>-cached  (Option B — RunPod model cache)
#
# RunPod endpoint setup:
#   Option A: image = ...-baked,  no Cached model setting
#   Option B: image = ...-cached, Cached model = PaddlePaddle/PP-DocLayoutV3_safetensors

set -euo pipefail

IMAGE="ringkasannet/layout-worker"
TAG="${1:-v1}"

echo "=== Option A: baked weights ==="
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile.a \
    -t "$IMAGE:$TAG-baked" \
    --progress=plain \
    .
docker push "$IMAGE:$TAG-baked"

echo ""
echo "=== Option B: RunPod model cache ==="
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile.b \
    -t "$IMAGE:$TAG-cached" \
    --progress=plain \
    .
docker push "$IMAGE:$TAG-cached"

echo ""
echo "=== Pod: HTTP server (port 8080) ==="
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile.pod \
    -t "$IMAGE:$TAG-pod" \
    --progress=plain \
    .
docker push "$IMAGE:$TAG-pod"

echo ""
echo "Done."
echo ""
echo "  $IMAGE:$TAG-baked"
echo "    → RunPod endpoint: no Cached model setting needed"
echo ""
echo "  $IMAGE:$TAG-cached"
echo "    → RunPod endpoint: Cached model = PaddlePaddle/PP-DocLayoutV3_safetensors"
echo ""
echo "  $IMAGE:$TAG-pod"
echo "    → RunPod Pod: exposes HTTP on port 8080, POST / with same JSON input"
