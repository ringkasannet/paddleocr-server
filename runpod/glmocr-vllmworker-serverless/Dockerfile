# GLM-OCR RunPod serverless worker — direct Pipeline architecture
#
# Eliminates the glmocr HTTP subprocess (no gunicorn/flask hop).
# handler.py calls glmocr.Pipeline directly; vLLM runs as a subprocess.
# concurrency_modifier=2 allows RunPod to send 2 jobs per worker simultaneously,
# enabling vLLM continuous batching across both (same gain as Modal gthread×2).
#
# Model weights: configure RunPod endpoint with model caching:
#   Cached model: zai-org/GLM-OCR
#   RunPod downloads to /runpod-volume/huggingface-cache/hub/ before any worker starts.
#
# Optional — bake weights into the image for fastest possible cold start
# (eliminates any first-run download but produces a ~3 GB larger image):
#   docker build \
#     --build-arg BAKE_MODEL=1 \
#     --build-arg HF_TOKEN=hf_xxx \
#     -t glm-ocr-pipeline-worker:v1-baked .
#   Then set env var HF_HOME=/app/hf_cache on the RunPod endpoint.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Point HF_HOME at RunPod's pre-downloaded model cache volume.
# Override to /app/hf_cache on the endpoint when using a baked image.
ENV HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    RAY_METRICS_EXPORT_ENABLED=0

# Install vLLM + glmocr (selfhosted = layout detector, server = flask deps for config compat)
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "vllm>=0.19.0" \
        "transformers>=5.3.0" \
        "glmocr[selfhosted,server]" \
        runpod

# Optional: bake model weights into the image.
# Weights land in /app/hf_cache — a separate layer so the pip layer above is
# not invalidated when the token or model revision changes.
ARG BAKE_MODEL=""
ARG HF_TOKEN=""
RUN if [ -n "$BAKE_MODEL" ]; then \
        echo "[bake] Downloading GLM-OCR weights into image..." && \
        HF_HOME=/app/hf_cache \
        HUGGINGFACE_HUB_CACHE=/app/hf_cache/hub \
        hf download zai-org/GLM-OCR ${HF_TOKEN:+--token "${HF_TOKEN}"} && \
        hf download PaddlePaddle/PP-DocLayoutV3_safetensors && \
        echo "[bake] Done — set HF_HOME=/app/hf_cache on the RunPod endpoint"; \
    fi

WORKDIR /app
# Copy handler last — weight / package layers above are not invalidated by code changes
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
