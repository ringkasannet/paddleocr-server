# GLM-OCR RunPod serverless worker
# Bundles vLLM + glmocr into a single image.
# Both model weights are baked in to eliminate cold-start downloads:
#   - zai-org/GLM-OCR              (VLM, ~1.8 GB)
#   - PaddlePaddle/PP-DocLayoutV3_safetensors  (layout detector, ~0.6 GB)
#
# Build: docker build --build-arg HF_TOKEN=hf_xxx -t glm-ocr-worker:v2 .

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Model cache lives inside the image — no runtime download needed.
# HF_HUB_OFFLINE prevents accidental re-downloads if weights are already baked in.
ENV HF_HOME=/app/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/hf_cache/hub \
    TRANSFORMERS_CACHE=/app/hf_cache/hub \
    PYTHONUNBUFFERED=1

# Install uv for fast package installs, then vLLM + glmocr
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "vllm>=0.19.0" \
        "transformers>=5.3.0" \
        "glmocr[selfhosted,server]" \
        runpod

# Bake in model weights — separate layer so pip cache is not invalidated
# when the token or model revision changes.
# Explicitly clear offline flag in case a cached layer inherited it.
ENV HF_HUB_OFFLINE=0
ARG HF_TOKEN=""
RUN hf download zai-org/GLM-OCR \
        ${HF_TOKEN:+--token "${HF_TOKEN}"} && \
    hf download PaddlePaddle/PP-DocLayoutV3_safetensors

# Now that weights are baked in, go offline to prevent accidental re-downloads at runtime
ENV HF_HUB_OFFLINE=1

WORKDIR /app

# Copy handler last — changes here don't invalidate the layers above
COPY glmocr_config.yaml /app/glmocr_config.yaml
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
