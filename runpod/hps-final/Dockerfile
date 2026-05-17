FROM ringkasannet/paddleocr-hps:paddlex3.4-gpu

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update -qq && apt-get install -y curl wget -qq \
    && rm -rf /var/lib/apt/lists/*

# ── 1. flash-attn (pre-built — avoids CUDA 11.8 vs torch 2.8 compile mismatch) ──
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"

# ── 2. Pin numpy (flash-attn upgrades it to 2.x, breaking pandas/paddleocr CLI) ──
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "numpy==1.26.4" --force-reinstall --no-deps

# ── 3. paddleocr (provides CLI needed by genai_server) ──────────────────────────
RUN /paddlex/py310/bin/pip install --no-cache-dir paddleocr==3.5.0

# ── 4. vllm deps (direct pip install — bypasses CUDA check that fails during build) ──
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "vllm==0.10.2" \
    "transformers<5.0.0" \
    "einops" \
    "uvloop"

# ── 5. matplotlib ≥3.9 ──────────────────────────────────────────────────────────
RUN /paddlex/py310/bin/pip install --no-cache-dir "matplotlib>=3.9" --upgrade

# ── 6. paddlex[ocr] extras ───────────────────────────────────────────────────────
RUN PADDLEX_VER=$(/paddlex/py310/bin/pip show paddlex | grep ^Version | awk '{print $2}') && \
    /paddlex/py310/bin/pip install --no-cache-dir "paddlex[ocr]==$PADDLEX_VER"

# ── 7. numpy final pin — MUST be last pip install ────────────────────────────────
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "numpy==1.26.4" --force-reinstall --no-deps

# ── 8. HPS SDK (model_repo + pipeline config + client wheel) ─────────────────────
RUN wget -q "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz" \
    -O /tmp/sdk.tar.gz && \
    tar -xf /tmp/sdk.tar.gz -C /tmp/ && \
    mkdir -p /opt/hps/server /opt/hps/client && \
    cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/server/. /opt/hps/server/ && \
    cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/client/. /opt/hps/client/ && \
    rm -rf /tmp/sdk.tar.gz /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk

# Patch pipeline_config.yaml: use vllm-server instead of native
RUN /paddlex/py310/bin/python3 -c "import yaml; path='/opt/hps/server/pipeline_config.yaml'; cfg=yaml.safe_load(open(path)); cfg['SubModules']['VLRecognition']['genai_config']={'backend':'vllm-server','server_url':'http://localhost:8118/v1','max_concurrency':32}; yaml.dump(cfg, open(path,'w'), default_flow_style=False, allow_unicode=True, sort_keys=False); print('patched')"

# Fix missing config.pbtxt and tune Triton batching
RUN cp /opt/hps/server/model_repo/layout-parsing/config_gpu.pbtxt \
    /opt/hps/server/model_repo/layout-parsing/config.pbtxt && \
    sed -i 's/max_batch_size: 8/max_batch_size: 16/' \
    /opt/hps/server/model_repo/layout-parsing/config.pbtxt && \
    sed -i 's/count: 1/count: 2/' \
    /opt/hps/server/model_repo/layout-parsing/config.pbtxt

# ── 9. Gateway app + venv ────────────────────────────────────────────────────────
RUN mkdir -p /opt/gateway
COPY app.py /opt/gateway/app.py

RUN /paddlex/py310/bin/python3 -m venv /opt/.venv_gateway && \
    /opt/.venv_gateway/bin/pip install --upgrade pip -q && \
    /opt/.venv_gateway/bin/pip install --no-cache-dir -q \
    fastapi==0.123.6 uvicorn==0.35.0 "paddlex[serving]>=3.4.0" && \
    /opt/.venv_gateway/bin/pip install --no-cache-dir -q \
    -r /opt/hps/client/requirements.txt && \
    WHL=$(ls /opt/hps/client/paddlex_hps_client-*.whl | head -1) && \
    /opt/.venv_gateway/bin/pip install --no-cache-dir -q "$WHL"

# ── 10. Model weights (~2.2 GB from local cache via --build-context) ─────────────
# Build with: docker build --build-context paddlex_models=$HOME/.paddlex/official_models ...
COPY --from=paddlex_models . /root/.paddlex/official_models/

# ── 11. Start script ─────────────────────────────────────────────────────────
COPY start.sh /opt/start_hps.sh
RUN chmod +x /opt/start_hps.sh

# ── 12. Benchmark test script ─────────────────────────────────────────────────
RUN /paddlex/py310/bin/pip install --no-cache-dir pypdf
COPY test_hps.py /opt/test_hps.py

# ── Verify ───────────────────────────────────────────────────────────────────────
RUN /paddlex/py310/bin/python3 -c "import paddleocr; print('paddleocr OK')" && \
    /paddlex/py310/bin/python3 -c "import vllm; print('vllm OK')" && \
    /paddlex/py310/bin/python3 -c "import numpy; print('numpy', numpy.__version__)" && \
    test -f /root/.paddlex/official_models/PaddleOCR-VL/model.safetensors && echo "model weights OK" && \
    test -f /opt/hps/server/pipeline_config.yaml && echo "HPS SDK OK" && \
    test -f /opt/.venv_gateway/bin/uvicorn && echo "gateway venv OK"

# Start all services automatically on pod boot.
# SSH access is provided by RunPod/Vast.ai infrastructure — unaffected by CMD.
ENTRYPOINT []
CMD ["bash", "/opt/start_hps.sh"]

# NVIDIA Container Toolkit: mount compute + utility libraries so CUDA works at runtime
# Placed at the bottom so edits here don't invalidate any build layers above.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.8"
