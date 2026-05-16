#!/usr/bin/env bash
# Build and push ringkasannet/paddleocr-hps image with model weights baked in.
#
# Run from WSL:
#   cd /mnt/d/paddleocr/PaddleOCR-main/runpod/hps-final
#   bash build.sh               # → ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready
#   bash build.sh v2            # → ringkasannet/paddleocr-hps:v2
#
# Requires:
#   - Docker with BuildKit (default in Docker Desktop 20.10+)
#   - Model weights at $HOME/.paddlex/official_models/PaddleOCR-VL/
#     (download by running the pod once, then scp back)

set -euo pipefail

IMAGE="ringkasannet/paddleocr-hps"
TAG="${1:-paddlex3.4-gpu-ready}"
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

# Both models are baked into the image — official PaddlePaddle repos on HuggingFace
_hf_download "PaddlePaddle/PaddleOCR-VL-1.5" "$MODELS_DIR/PaddleOCR-VL"
_hf_download "PaddlePaddle/PP-DocLayoutV2"    "$MODELS_DIR/PP-DocLayoutV2"

echo "[build] Image  : $IMAGE:$TAG"
echo "[build] Models : $MODELS_DIR ($(du -sh "$MODELS_DIR" | cut -f1))"

DOCKER_BUILDKIT=1 docker build \
    --build-context paddlex_models="$MODELS_DIR" \
    -f Dockerfile \
    -t "$IMAGE:$TAG" \
    --progress=plain \
    .

echo "[build] Pushing $IMAGE:$TAG ..."
docker push "$IMAGE:$TAG"

echo ""
echo "Done: $IMAGE:$TAG"
echo ""
echo "RunPod pod settings:"
echo "  Container image  : $IMAGE:$TAG"
echo "  Start command    : bash /opt/start_hps.sh"
echo "  Expose TCP port  : 8080  (gateway — use TCP not HTTP to bypass proxy timeout)"
echo ""
echo "Env vars (set in RunPod template):"
echo "  GPU_MEMORY_UTILIZATION=0.65   # default 0.50 — increase for more KV cache"
echo "  HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=16  # default 16"
echo "  HPS_INFERENCE_TIMEOUT=600     # default 600s"
echo "  UVICORN_WORKERS=4             # default 4"
