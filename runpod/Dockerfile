# ============================================================
# PaddleOCR RunPod — Serverless
#
# Base: paddleocr-genai-vllm-server (vLLM already installed)
# Adds: PaddlePaddle + paddleocr Python API in isolated venv
#
# At runtime:
#   - vLLM genai server starts as subprocess (port 8118, internal)
#   - PaddleOCRVL Python API calls vLLM directly (no HTTP server)
# ============================================================
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-offline

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/models/hf_cache

USER root
WORKDIR /workspace

RUN which paddleocr

# ── System packages needed by PaddlePaddle / OpenCV ──────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        python3-venv python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

# ── PaddlePaddle virtual environment ─────────────────────────
# Isolated from vLLM env — PaddlePaddle conflicts with vLLM deps
RUN python3 -m venv /workspace/.paddleocr

RUN /workspace/.paddleocr/bin/pip install --upgrade pip setuptools wheel

RUN /workspace/.paddleocr/bin/pip install --no-cache-dir \
    paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

RUN /workspace/.paddleocr/bin/pip install --no-cache-dir "paddleocr[doc-parser]"

# ── RunPod serverless SDK ─────────────────────────────────────
RUN /workspace/.paddleocr/bin/pip install --no-cache-dir runpod

# ── Handler ───────────────────────────────────────────────────
COPY handler.py /workspace/handler.py

# ── Service defaults ──────────────────────────────────────────
ENV MODEL_NAME=PaddleOCR-VL-1.5-0.9B
ENV GPU_MEMORY_UTILIZATION=0.85
ENV VLLM_PORT=8118
ENV RUNPOD_INIT_TIMEOUT=600

CMD ["/workspace/.paddleocr/bin/python", "-u", "/workspace/handler.py"]
