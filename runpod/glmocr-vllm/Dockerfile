FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

# Model weights come from RunPod model caching — configure the endpoint with:
#   Cached model: zai-org/GLM-OCR
# RunPod downloads the model to /runpod-volume/huggingface-cache/hub/ before
# any worker starts. HF_HOME points there so vLLM finds it without downloading.
ENV HF_HOME=/runpod-volume/huggingface-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "vllm>=0.9.0" \
    "transformers>=5.3.0" \
    "pypdfium2" \
    "Pillow" \
    "openai" \
    "runpod"

WORKDIR /app
COPY handler.py .
CMD ["python", "-u", "handler.py"]
