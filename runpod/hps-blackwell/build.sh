#!/usr/bin/env bash
# Build and push ringkasannet/paddleocr-hps Blackwell (SM 120) image.
# Uses official Nvidia Triton + PaddlePaddle cu129 instead of Baidu's base.
#
# Run from WSL:
#   cd /mnt/d/paddleocr/PaddleOCR-main/runpod/hps-blackwell
#   bash build.sh               # → ringkasannet/paddleocr-hps:blackwell
#   bash build.sh v2            # → ringkasannet/paddleocr-hps:blackwell-v2

set -euo pipefail

IMAGE="ringkasannet/paddleocr-hps"
TAG="blackwell${1:+-$1}"
MODELS_DIR="${PADDLEX_MODELS:-$HOME/.paddlex/official_models}"

pip install -q huggingface_hub 2>/dev/null || true

_hf_download() {
    local repo=$1 dest=$2
    if [ ! -d "$dest" ]; then
        echo "[build] Downloading $repo → $dest ..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$repo', local_dir='$dest', repo_type='model')
print('[build] $repo done.')
"
    else
        echo "[build] Found $dest — skipping download."
    fi
}

_hf_download "PaddlePaddle/PaddleOCR-VL-1.5" "$MODELS_DIR/PaddleOCR-VL"
_hf_download "PaddlePaddle/PP-DocLayoutV2"    "$MODELS_DIR/PP-DocLayoutV2"

echo "[build] Image  : $IMAGE:$TAG"
echo "[build] Models : $MODELS_DIR"
echo "[build] Note   : Blackwell (SM 120) — no flash-attn, vLLM uses fallback attention"

DOCKER_BUILDKIT=1 docker build \
    --build-context paddlex_models="$MODELS_DIR" \
    -f Dockerfile \
    -t "$IMAGE:$TAG" \
    --progress=plain \
    --no-cache \
    .

echo "[build] Pushing $IMAGE:$TAG ..."
docker push "$IMAGE:$TAG"

echo ""
echo "Done: $IMAGE:$TAG"
echo ""
echo "RunPod pod settings:"
echo "  Container image  : $IMAGE:$TAG"
echo "  Start command    : bash /opt/start_hps.sh"
echo "  Expose TCP port  : 8080"
echo "  Required driver  : CUDA 12.9+ (RTX 50xx, Blackwell)"
echo ""
echo "Env vars:"
echo "  GPU_MEMORY_UTILIZATION=0.50"
echo "  HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=16"
