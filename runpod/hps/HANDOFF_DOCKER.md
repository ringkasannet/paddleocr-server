# Handoff: Examine Base Docker Image & Build Custom HPS Image

## Goal

Build a custom Docker image `ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready` that contains everything pre-installed so that fresh RunPod pods start immediately without running `setup.sh`.

The current problem: packages installed into `/paddlex/py310/` (paddleocr, vllm, flash-attn) are in the container filesystem, not the network volume. Every fresh pod requires re-running `setup.sh` (~15-20 min). We want to bake those into the image.

Model weights (~2GB) should NOT be in the image — they stay on the network volume `/workspace/models/`.

---

## Environment

| Item | Value |
|------|-------|
| Base image | `ringkasannet/paddleocr-hps:paddlex3.4-gpu` (retag of Baidu's `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.4-gpu`) |
| Target image | `ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready` |
| Container Python | `/paddlex/py310/bin/python3` = Python 3.10.4 |
| CUDA in container | 11.8 (nvcc), but torch is 2.8.0+cu128 |
| GPU at runtime | RTX A4500, 20 GB VRAM, CUDA driver 580.x |
| Docker Hub user | `ringkasannet` |
| Local machine | Windows 11, NVIDIA GPU available, Docker Desktop + WSL2 |

---

## What the Base Image Already Contains (read-only)

| Component | Path | Status |
|-----------|------|--------|
| Triton binary | `/opt/tritonserver/bin/tritonserver` | ✅ |
| Triton Python backend | `/opt/tritonserver/backends/python/` | ✅ |
| PaddlePaddle-GPU | `/paddlex/py310/lib/python3.10/site-packages/` | ✅ |
| paddlex | `/paddlex/py310/lib/python3.10/site-packages/paddlex/` | ✅ |
| paddlex_hps_server | `/paddlex/py310/lib/python3.10/site-packages/paddlex_hps_server/` | ✅ |

## What Is NOT in the Base Image (needs to be added)

| Package | Notes |
|---------|-------|
| `flash-attn 2.8.2` | Must use pre-built wheel (CUDA 11.8 nvcc vs torch cu128 mismatch) |
| `paddleocr 3.5.0` | Needed for CLI and `genai_server` subcommand |
| `vllm 0.10.2` | Installed via `paddleocr install_genai_server_deps vllm` |
| `paddlex[ocr]` extras | Required for VLM pipeline |
| `matplotlib>=3.9` | Needed for numpy 2.x compatibility |
| `numpy 1.26.4` (pinned) | Must be last install — flash-attn and vllm upgrade it to 2.x |

---

## Step 1 — Examine the Base Image

Before building, examine what's actually in the base image to avoid redundant installs:

```bash
# Pull the base image
docker pull ringkasannet/paddleocr-hps:paddlex3.4-gpu

# Full pip list
docker run --rm ringkasannet/paddleocr-hps:paddlex3.4-gpu \
  /paddlex/py310/bin/pip list 2>/dev/null | sort > base_pip_list.txt

# Image layer sizes
docker history ringkasannet/paddleocr-hps:paddlex3.4-gpu

# Key paths and disk usage
docker run --rm ringkasannet/paddleocr-hps:paddlex3.4-gpu bash -c "
  echo '=== Python packages count ===' && /paddlex/py310/bin/pip list | wc -l
  echo '=== Key packages ===' && /paddlex/py310/bin/pip show paddlepaddle-gpu paddlex numpy torch 2>/dev/null | grep -E 'Name|Version'
  echo '=== Disk usage ===' && du -sh /paddlex /opt/tritonserver 2>/dev/null
  echo '=== curl/wget ===' && which curl wget 2>/dev/null || echo 'missing'
  echo '=== numpy version ===' && /paddlex/py310/bin/python3 -c 'import numpy; print(numpy.__version__)'
  echo '=== torch version ===' && /paddlex/py310/bin/python3 -c 'import torch; print(torch.__version__)'
"
```

Check specifically if any of our target packages are already there:
```bash
docker run --rm ringkasannet/paddleocr-hps:paddlex3.4-gpu \
  /paddlex/py310/bin/pip show flash-attn vllm paddleocr 2>/dev/null | grep -E "Name|Version|not found"
```

---

## Step 2 — Build the Custom Image

The Dockerfile is at: `d:\paddleocr\PaddleOCR-main\runpod\hps\Dockerfile.hps`

Current content:
```dockerfile
FROM ringkasannet/paddleocr-hps:paddlex3.4-gpu

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y curl wget -qq \
    && rm -rf /var/lib/apt/lists/*

# flash-attn pre-built wheel
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"

# Pin numpy after flash-attn upgrade
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "numpy==1.26.4" --force-reinstall --no-deps

# paddleocr (provides CLI)
RUN /paddlex/py310/bin/pip install --no-cache-dir paddleocr==3.5.0

# vllm via paddleocr CLI
RUN /paddlex/py310/bin/paddleocr install_genai_server_deps vllm

# matplotlib
RUN /paddlex/py310/bin/pip install --no-cache-dir "matplotlib>=3.9" --upgrade

# paddlex[ocr] extras
RUN PADDLEX_VER=$(/paddlex/py310/bin/pip show paddlex | grep ^Version | awk '{print $2}') && \
    /paddlex/py310/bin/pip install --no-cache-dir "paddlex[ocr]==$PADDLEX_VER"

# FINAL numpy pin
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "numpy==1.26.4" --force-reinstall --no-deps

# Verify
RUN /paddlex/py310/bin/python3 -c "import paddleocr; print('paddleocr OK')" && \
    /paddlex/py310/bin/python3 -c "import vllm; print('vllm OK')" && \
    /paddlex/py310/bin/python3 -c "import numpy; print('numpy', numpy.__version__)"
```

Build command (from WSL, no GPU needed for build):
```bash
cd /mnt/d/paddleocr/PaddleOCR-main/runpod/hps
docker build -f Dockerfile.hps -t ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready --progress=plain .
```

---

## Step 3 — Test the Image

```bash
# Verify all packages load correctly
docker run --rm ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready \
  /paddlex/py310/bin/python3 -c "
import paddleocr, vllm, flash_attn, numpy, matplotlib
print('paddleocr:', paddleocr.__version__)
print('vllm:', vllm.__version__)
print('numpy:', numpy.__version__)
print('All OK')
"

# Verify genai_server subcommand is registered
docker run --rm ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready \
  /paddlex/py310/bin/python3 -m paddleocr --help 2>&1 | grep genai_server
```

---

## Step 4 — Push to Docker Hub

```bash
docker push ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready
```

---

## Step 5 — Update RunPod Pod Image

Change the RunPod pod image from:
```
ringkasannet/paddleocr-hps:paddlex3.4-gpu
```
To:
```
ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready
```

With the new image, `setup.sh` Task B is skipped automatically (it checks if vllm/paddleocr/flash_attn are importable). `start_hps.sh` works immediately on fresh pods.

---

## Known Issues & Context

### numpy version conflict
flash-attn and vllm both upgrade numpy to 2.x during installation. pandas and old matplotlib were compiled for numpy 1.x. The fix is always pinning numpy==1.26.4 as the LAST pip install step. Do NOT change this ordering.

### flash-attn compilation
The container has CUDA 11.8 nvcc but torch 2.8.0+cu128 (CUDA 12.8). Compiling flash-attn from source fails. The pre-built wheel from `mjun0812/flash-attention-prebuild-wheels` is the only working approach.

### install_genai_server_deps
`paddleocr install_genai_server_deps vllm` is hardcoded inside paddleocr to call `/paddlex/py310/bin/python` for pip. This is why paddleocr must be installed BEFORE calling this command, and everything must be in `/paddlex/py310/`.

### genai_server subcommand
The `genai_server` subcommand is dynamically registered only when `vllm` is importable in the same Python environment. After `install_genai_server_deps vllm`, the subcommand appears in `/paddlex/py310/bin/python3 -m paddleocr`.

### Model weights
Do NOT include model weights in the Docker image. They are ~2GB and are downloaded to `/workspace/models/paddlex/` (network volume) on first run. The `start_hps.sh` creates symlinks:
```bash
ln -sfn /workspace/models/paddlex /root/.paddlex
ln -sfn /workspace/.cache/vllm /root/.cache/vllm
```

---

## File Locations

| File | Location |
|------|----------|
| Dockerfile | `d:\paddleocr\PaddleOCR-main\runpod\hps\Dockerfile.hps` |
| Build script | `d:\paddleocr\PaddleOCR-main\runpod\hps\build_push.sh` |
| Start script | `d:\paddleocr\PaddleOCR-main\runpod\hps\start_hps.sh` |
| Setup script | `d:\paddleocr\PaddleOCR-main\runpod\hps\setup.sh` |
| SETUP.md (full history) | `d:\paddleocr\PaddleOCR-main\runpod\hps\SETUP.md` |

---

## What to Report Back

After examining the base image (Step 1), report:
1. Which of our target packages are already in the base image
2. numpy version in base image
3. torch version in base image  
4. Total base image size
5. Any packages that might conflict with our installs
6. Whether the Dockerfile needs adjustments based on findings
