#!/usr/bin/env bash
# Usage:
#   ./build.sh <dockerhub-username> [tag]
#   ./build.sh myuser                     # builds myuser/paddleocr-runpod:latest
#   ./build.sh myuser v1.0                # builds myuser/paddleocr-runpod:v1.0

set -euo pipefail

DOCKER_USER="${1:?Usage: ./build.sh <dockerhub-username> [tag]}"
TAG="${2:-latest}"
IMAGE="${DOCKER_USER}/paddleocr-runpod:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building: ${IMAGE}"
docker build \
    --platform linux/amd64 \
    -t "${IMAGE}" \
    "${SCRIPT_DIR}"

echo ""
echo "Pushing: ${IMAGE}"
docker push "${IMAGE}"

echo ""
echo "Done! Use this image on RunPod:"
echo "  ${IMAGE}"
echo ""
echo "RunPod pod settings:"
echo "  Container image : ${IMAGE}"
echo "  Expose HTTP port: 8080  (PaddleOCR pipeline API)"
echo "  Expose TCP port : 8118  (vLLM server, optional)"
echo ""
echo "Optional env vars you can set on the RunPod pod:"
echo "  GPU_MEMORY_UTILIZATION=0.65"
echo "  MODEL_NAME=PaddleOCR-VL-1.5-0.9B"
echo "  VLLM_PORT=8118"
echo "  PADDLE_PORT=8080"
echo "  HF_HOME=/runpod-volume/hf_cache   # mount a volume to cache the model"
