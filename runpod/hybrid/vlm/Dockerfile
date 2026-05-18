FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# RunPod volume cache — GLM-OCR model pre-downloaded by RunPod.
# If zai-org/GLM-OCR is gated (requires HF_TOKEN), use the baked-in approach:
#   ARG HF_TOKEN=""
#   RUN HF_HOME=/app/hf_cache huggingface-cli download zai-org/GLM-OCR ${HF_TOKEN:+--token "${HF_TOKEN}"}
#   ENV HF_HOME=/app/hf_cache HF_HUB_OFFLINE=1 ...
ENV HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "vllm>=0.9.0" \
        "transformers>=5.3.0" \
        "aiohttp>=3.9.0" \
        runpod

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
