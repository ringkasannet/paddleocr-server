FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# RunPod volume cache — layout model is pre-downloaded by RunPod, not baked in.
# ~600 MB; loads from NVMe in ~10s.
ENV HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    PYTHONUNBUFFERED=1

# Install system deps for PyMuPDF and pdfplumber
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "transformers>=5.3.0" \
        "torch>=2.4.0" \
        "pillow>=10.0.0" \
        "pymupdf>=1.24.0" \
        "pdfplumber>=0.11.0" \
        "glmocr[selfhosted]>=0.4.0" \
        runpod

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
