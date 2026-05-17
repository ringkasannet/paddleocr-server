# GLM-OCR RunPod serverless worker
# Bundles vLLM + glmocr into a single image.
# Model weights are NOT baked in — loaded at runtime from RunPod model cache
# (HF_HOME=/runpod-volume/huggingface-cache) or downloaded on first cold start.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Point HuggingFace at the RunPod model-cache mount point so the platform's
# cached model files are found automatically without re-downloading.
ENV HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    PYTHONUNBUFFERED=1

# Install uv for fast package installs, then vLLM + glmocr
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "vllm>=0.19.0" \
        "transformers>=5.3.0" \
        "glmocr[selfhosted,server]" \
        runpod

WORKDIR /app

# Copy handler last — changes here don't invalidate the pip layer above
COPY glmocr_config.yaml /app/glmocr_config.yaml
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
