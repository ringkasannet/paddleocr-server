# ============================================================
# PaddleOCR RunPod — Pod (persistent GPU / local testing)
#
# Base: paddleocr-genai-vllm-server (vLLM already installed)
# Adds: PaddlePaddle + paddleocr + paddlex serving in isolated venv
#
# At runtime (start.sh):
#   - vLLM genai server on port 8118
#   - PaddleX HTTP pipeline server on port 8080  ← expose this
# ============================================================
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-offline

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/models/hf_cache

USER root
WORKDIR /workspace

RUN which paddleocr

# ── System packages ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        python3-venv python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

# ── PaddlePaddle + PaddleX serving virtual environment ───────
RUN python3 -m venv /workspace/.paddleocr

RUN /workspace/.paddleocr/bin/pip install --upgrade pip setuptools wheel

RUN /workspace/.paddleocr/bin/pip install --no-cache-dir \
    paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

RUN /workspace/.paddleocr/bin/pip install --no-cache-dir "paddleocr[doc-parser]"

# PaddleX HTTP serving plugin
RUN PATH="/workspace/.paddleocr/bin:$PATH" paddlex --install serving

# Pipeline configuration
RUN PATH="/workspace/.paddleocr/bin:$PATH" paddlex --get_pipeline_config PaddleOCR-VL --save_path /workspace

COPY patch_config.py /workspace/patch_config.py
RUN /workspace/.paddleocr/bin/python /workspace/patch_config.py

# ── Startup script ────────────────────────────────────────────
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 8118 8080

# ── Service defaults ──────────────────────────────────────────
ENV MODEL_NAME=PaddleOCR-VL-1.5-0.9B
ENV GPU_MEMORY_UTILIZATION=0.65
ENV VLLM_PORT=8118
ENV PADDLE_PORT=8080
ENV RUNPOD_INIT_TIMEOUT=600

CMD ["bash", "/workspace/start.sh"]
