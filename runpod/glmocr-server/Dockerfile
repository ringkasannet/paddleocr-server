# GLM-OCR RunPod serverless worker — model weights via RunPod model cache.
#
# Configure the endpoint with two cached models:
#   zai-org/GLM-OCR
#   PaddlePaddle/PP-DocLayoutV3_safetensors
#
# RunPod downloads both to /runpod-volume/huggingface-cache/hub/ before
# any worker starts; HF_HOME points there so vLLM and glmocr find them
# without a network download at cold start.
#
# Build: bash build_push.sh [tag]

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "vllm>=0.21.0" \
        "transformers>=5.3.0" \
        "glmocr[selfhosted,server]" \
        runpod

WORKDIR /app
COPY handler.py .
CMD ["python", "-u", "handler.py"]
