FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

# Model weights come from RunPod model caching — configure the endpoint with:
#   Cached model: PaddlePaddle/PP-DocLayoutV3_safetensors
# RunPod downloads the model to /runpod-volume/huggingface-cache/hub/ before
# any worker starts. HF_HOME points there so from_pretrained finds it.
ENV HF_HOME=/runpod-volume/huggingface-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "transformers>=5.3.0" \
    "pypdfium2" \
    "Pillow" \
    "opencv-python-headless" \
    "runpod"

WORKDIR /app
COPY handler.py .
CMD ["python", "-u", "handler.py"]
