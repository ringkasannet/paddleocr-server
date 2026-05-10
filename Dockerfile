# ============================================================
# PaddleOCR RunPod — single container, two services
#
# Base: paddleocr-genai-vllm-server  (vLLM already installed)
# We only add an isolated PaddleX serving venv on top.
#
# Port 8118  →  vLLM genai server  (internal)
# Port 8080  →  PaddleX HTTP API   (expose this on RunPod)
# ============================================================
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/models/hf_cache

USER root

WORKDIR /workspace

# Confirm the vLLM service binary is in the base image
RUN which paddleocr

# ── System packages needed by PaddlePaddle / OpenCV ──────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        python3-venv python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

# ── PaddleX serving virtual environment ──────────────────────
# Isolated from the vLLM env in the base image
RUN python3 -m venv /workspace/.paddleocr

RUN /workspace/.paddleocr/bin/pip install --upgrade pip setuptools wheel

# PaddlePaddle GPU (CUDA 12.6 build is compatible with CUDA 12.x driver)
RUN /workspace/.paddleocr/bin/pip install --no-cache-dir \
    paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

RUN /workspace/.paddleocr/bin/pip install --no-cache-dir "paddleocr[doc-parser]"

# Install PaddleX serving plugin
RUN PATH="/workspace/.paddleocr/bin:$PATH" paddlex --install serving

# ── Pipeline configuration ────────────────────────────────────
RUN PATH="/workspace/.paddleocr/bin:$PATH" paddlex --get_pipeline_config PaddleOCR-VL --save_path /workspace

COPY patch_config.py /workspace/patch_config.py
RUN /workspace/.paddleocr/bin/python /workspace/patch_config.py

# ── RunPod serverless SDK ─────────────────────────────────────
# Installed in the system Python (same env as paddleocr genai_server)
RUN pip install --no-cache-dir runpod

# ── Startup scripts ───────────────────────────────────────────
# start.sh  → Pod / local testing (persistent HTTP servers on :8080 + :8118)
# handler.py → Serverless (RunPod job handler wrapping both services)
COPY start.sh /workspace/start.sh
COPY handler.py /workspace/handler.py
RUN chmod +x /workspace/start.sh

EXPOSE 8118 8080

# Build with --build-arg MODE=serverless (default) or MODE=pod
# Results in two tags sharing all heavy layers:
#   ringkasannet/paddleocr-runpod:serverless
#   ringkasannet/paddleocr-runpod:pod
ARG MODE=serverless
ENV MODE=${MODE}
CMD ["bash", "-c", "if [ \"$MODE\" = \"pod\" ]; then exec bash /workspace/start.sh; else exec python -u /workspace/handler.py; fi"]
