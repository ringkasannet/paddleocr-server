#!/usr/bin/env bash
# ============================================================
# HPS Network Volume Extractor
#
# Run from WSL2. Docker Desktop must be running and WSL
# integration enabled (Docker Desktop → Settings → Resources
# → WSL Integration → enable your distro).
#
# Extracts from paddlex/hps Docker image:
#   1. Triton server binary + libraries → ./volume_prep/tritonserver/
#   2. HPS SDK server files (model_repo, backends) → ./volume_prep/hps/server/
#   3. HPS SDK client files (gateway wheel) → ./volume_prep/hps/client/
#
# Usage:
#   cd /mnt/d/paddleocr/PaddleOCR-main/runpod/hps
#   bash extract.sh
# ============================================================
set -euo pipefail

HPS_IMAGE="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.4-gpu"
SDK_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/volume_prep"
REPO_ROOT="$SCRIPT_DIR/../.."

mkdir -p "$OUT_DIR"/{tritonserver,hps/server,hps/client,gateway}

echo "=== Step 1: Pulling paddlex/hps image ==="
#docker pull "$HPS_IMAGE"

# ── Extract Triton binary ─────────────────────────────────────
echo ""
echo "=== Step 2: Extracting Triton server binary ==="
docker create --name hps_tmp "$HPS_IMAGE" bash
docker cp hps_tmp:/opt/tritonserver "$OUT_DIR/tritonserver"
docker rm hps_tmp
echo "  → $OUT_DIR/tritonserver/"

# ── Download + extract HPS SDK in WSL directly ───────────────
echo ""
echo "=== Step 3: Downloading HPS SDK (server + client) ==="
wget -q --show-progress "$SDK_URL" -O /tmp/paddlex_hps_sdk.tar.gz
tar -xf /tmp/paddlex_hps_sdk.tar.gz -C /tmp/
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/server/. "$OUT_DIR/hps/server/"
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/client/. "$OUT_DIR/hps/client/"
rm /tmp/paddlex_hps_sdk.tar.gz
echo "  → $OUT_DIR/hps/"

# ── Copy gateway app ──────────────────────────────────────────
echo ""
echo "=== Step 4: Copying gateway app ==="
cp -r "$REPO_ROOT/deploy/paddleocr_vl_docker/hps/gateway/." "$OUT_DIR/gateway/"
echo "  → $OUT_DIR/gateway/"

# ── Copy scripts + config ─────────────────────────────────────
echo ""
echo "=== Step 5: Copying scripts and config ==="
cp "$SCRIPT_DIR/pipeline_config_local.yaml" "$OUT_DIR/hps/pipeline_config_local.yaml"
cp "$SCRIPT_DIR/install.sh"                 "$OUT_DIR/install.sh"
cp "$SCRIPT_DIR/start_hps.sh"               "$OUT_DIR/start_hps.sh"
chmod +x "$OUT_DIR/install.sh" "$OUT_DIR/start_hps.sh"
echo "  → scripts copied"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=== Extraction complete ==="
echo "Sizes:"
du -sh "$OUT_DIR"/*/  2>/dev/null | sort -h
echo ""
echo "Total: $(du -sh "$OUT_DIR" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Upload $OUT_DIR/ contents to RunPod network volume at /workspace/"
echo "     Example: rsync -avz --progress $OUT_DIR/ user@pod:/workspace/"
echo "  2. SSH into a RunPod pod and run: bash /workspace/install.sh"
