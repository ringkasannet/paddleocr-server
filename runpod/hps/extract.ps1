# ============================================================
# HPS Network Volume Extractor
#
# Pulls paddlex/hps Docker image locally, extracts:
#   1. Triton server binary + libraries → ./volume_prep/tritonserver/
#   2. HPS SDK server files (model_repo, backends) → ./volume_prep/hps/server/
#   3. HPS SDK client files (gateway wheel) → ./volume_prep/hps/client/
#
# Run this once on Windows. Then upload volume_prep/ to RunPod network volume.
# ============================================================

$HPS_IMAGE    = "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.4-gpu"
$SDK_URL      = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz"
$OUT_DIR      = "$PSScriptRoot\volume_prep"

Write-Host "`n=== Step 1: Pulling paddlex/hps image ===" -ForegroundColor Cyan
docker pull $HPS_IMAGE
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to pull image"; exit 1 }

# ── Extract Triton binary from image ──────────────────────────
Write-Host "`n=== Step 2: Extracting Triton server binary ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$OUT_DIR\tritonserver" | Out-Null

docker create --name hps_tmp $HPS_IMAGE bash | Out-Null
docker cp "hps_tmp:/opt/tritonserver" "$OUT_DIR\tritonserver"
docker rm hps_tmp | Out-Null

Write-Host "  Triton extracted to $OUT_DIR\tritonserver"

# ── Download + extract HPS SDK inside the image ───────────────
# Running inside the HPS image ensures wget and compatible glibc
Write-Host "`n=== Step 3: Downloading HPS SDK (server + client) ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$OUT_DIR\hps\server" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\hps\client" | Out-Null

$sdk_cmd = @"
apt-get install -y wget -qq && \
wget -q "$SDK_URL" -O /tmp/sdk.tar.gz && \
tar -xf /tmp/sdk.tar.gz -C /tmp/ && \
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/server/. /output/server/ && \
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/client/. /output/client/ && \
echo SDK_DONE
"@

docker run --rm `
  -v "${OUT_DIR}\hps:/output" `
  $HPS_IMAGE `
  bash -c $sdk_cmd

if ($LASTEXITCODE -ne 0) { Write-Error "SDK extraction failed"; exit 1 }
Write-Host "  SDK extracted to $OUT_DIR\hps"

# ── Copy gateway application ───────────────────────────────────
Write-Host "`n=== Step 4: Copying gateway app ===" -ForegroundColor Cyan
$gateway_src = "$PSScriptRoot\..\..\deploy\paddleocr_vl_docker\hps\gateway"
New-Item -ItemType Directory -Force -Path "$OUT_DIR\gateway" | Out-Null
Copy-Item -Path "$gateway_src\*" -Destination "$OUT_DIR\gateway" -Recurse -Force
Write-Host "  Gateway copied to $OUT_DIR\gateway"

# ── Copy scripts and configs ───────────────────────────────────
Write-Host "`n=== Step 5: Copying scripts ===" -ForegroundColor Cyan
Copy-Item "$PSScriptRoot\pipeline_config_local.yaml" "$OUT_DIR\hps\pipeline_config_local.yaml" -Force
Copy-Item "$PSScriptRoot\install.sh"   "$OUT_DIR\install.sh" -Force
Copy-Item "$PSScriptRoot\start_hps.sh" "$OUT_DIR\start_hps.sh" -Force

# ── Show sizes ─────────────────────────────────────────────────
Write-Host "`n=== Extraction complete ===" -ForegroundColor Green
Write-Host "Directory sizes:"
Get-ChildItem $OUT_DIR -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum
    Write-Host ("  {0,-20} {1,6} MB" -f $_.Name, [math]::Round($size/1MB, 0))
}

Write-Host "`nNext: Upload $OUT_DIR\ contents to RunPod network volume at /workspace/"
Write-Host "      Then SSH into a RunPod pod and run: bash /workspace/install.sh"
