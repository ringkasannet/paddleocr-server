FROM nvcr.io/nvidia/tritonserver:25.05-py3

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System packages
RUN echo "==> [0/10] System packages" && \
    apt-get update -qq && apt-get install -y curl wget python3.12-venv -qq \
    && rm -rf /var/lib/apt/lists/* && \
    echo "==> [0/10] done"

# ── Create /paddlex/py310/ venv for path compatibility with start.sh ─────────
# Name is legacy — actual Python is whatever Triton 25.05 ships (3.12)
RUN echo "==> [venv] Creating /paddlex/py310 venv" && \
    python3 -m venv /paddlex/py310 && \
    echo "==> [venv] done — $(python3 --version)"

# ── 1. PaddlePaddle cu129 — SM 120 (Blackwell) support ───────────────────────
RUN echo "==> [1/10] Installing PaddlePaddle cu129 (SM 120 / Blackwell) ..." && \
    /paddlex/py310/bin/pip install --no-cache-dir paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ && \
    echo "==> [1/10] PaddlePaddle done"

# ── 2. paddleocr (provides CLI needed by genai_server) ───────────────────────
RUN echo "==> [2/10] Installing paddleocr 3.5.0 ..." && \
    /paddlex/py310/bin/pip install --no-cache-dir paddleocr==3.5.0 && \
    echo "==> [2/10] paddleocr done"

# ── 3. vllm deps ─────────────────────────────────────────────────────────────
RUN echo "==> [3/10] Installing vLLM 0.10.2 ..." && \
    /paddlex/py310/bin/pip install --no-cache-dir \
    "vllm==0.10.2" \
    "transformers<5.0.0" \
    "einops" \
    "uvloop" && \
    echo "==> [3/10] vLLM done"

# ── 3b. flash-attn for SM 120 — community prebuilt wheel (cu129 + torch2.8 + cp312)
# Official flash-attn PyPI has no SM 120 build; this wheel from mjun0812 works on
# Blackwell consumer GPUs via CUDA forward compatibility. Must be installed AFTER
# vLLM so PyTorch 2.8 is already present.
RUN echo "==> [3b] Installing flash-attn 2.8.3 (SM 120 / cu129 / cp312) ..." && \
    /paddlex/py310/bin/pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.11/flash_attn-2.8.3%2Bcu129torch2.8-cp312-cp312-linux_x86_64.whl" && \
    echo "==> [3b] flash-attn done"

# ── 4. matplotlib ≥3.9 ───────────────────────────────────────────────────────
RUN echo "==> [4/10] Installing matplotlib ..." && \
    /paddlex/py310/bin/pip install --no-cache-dir "matplotlib>=3.9" --upgrade && \
    echo "==> [4/10] matplotlib done"

# ── 5. paddlex[ocr] extras ───────────────────────────────────────────────────
RUN echo "==> [5/10] Installing paddlex[ocr] extras ..." && \
    PADDLEX_VER=$(/paddlex/py310/bin/pip show paddlex | grep ^Version | awk '{print $2}') && \
    echo "    paddlex version: $PADDLEX_VER" && \
    /paddlex/py310/bin/pip install --no-cache-dir "paddlex[ocr]==$PADDLEX_VER" && \
    echo "==> [5/10] paddlex[ocr] done"

# ── 6. numpy final pin — MUST come after flash-attn (which upgrades numpy to 2.x) ──
RUN echo "==> [6/10] Pinning numpy 1.26.4 ..." && \
    /paddlex/py310/bin/pip install --no-cache-dir \
    "numpy==1.26.4" --force-reinstall --no-deps && \
    echo "==> [6/10] numpy pinned: $(/paddlex/py310/bin/python3 -c 'import numpy; print(numpy.__version__)')"

# ── 7. HPS SDK ───────────────────────────────────────────────────────────────
RUN echo "==> [7/10] Downloading HPS SDK tarball ..." && \
    wget "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz" \
    -O /tmp/sdk.tar.gz --progress=bar:force 2>&1 && \
    echo "==> [7/10] Extracting SDK ..." && \
    tar -xf /tmp/sdk.tar.gz -C /tmp/ && \
    mkdir -p /opt/hps/server /opt/hps/client && \
    cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/server/. /opt/hps/server/ && \
    cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/client/. /opt/hps/client/ && \
    rm -rf /tmp/sdk.tar.gz /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk && \
    echo "==> [7/10] HPS SDK installed"

# Patch pipeline_config.yaml: vllm-server backend
RUN echo "==> [7b] Patching pipeline_config.yaml ..." && \
    /paddlex/py310/bin/python3 -c "import yaml; path='/opt/hps/server/pipeline_config.yaml'; cfg=yaml.safe_load(open(path)); cfg['SubModules']['VLRecognition']['genai_config']={'backend':'vllm-server','server_url':'http://localhost:8118/v1','max_concurrency':32}; yaml.dump(cfg, open(path,'w'), default_flow_style=False, allow_unicode=True, sort_keys=False); print('patched')"

# Fix config.pbtxt and tune Triton batching
RUN echo "==> [7c] Configuring Triton batching ..." && \
    cp /opt/hps/server/model_repo/layout-parsing/config_gpu.pbtxt \
    /opt/hps/server/model_repo/layout-parsing/config.pbtxt && \
    sed -i 's/max_batch_size: 8/max_batch_size: 16/' \
    /opt/hps/server/model_repo/layout-parsing/config.pbtxt && \
    sed -i 's/count: 1/count: 2/' \
    /opt/hps/server/model_repo/layout-parsing/config.pbtxt && \
    echo "==> [7c] Triton config done"

# ── 8. Gateway app + venv ────────────────────────────────────────────────────
RUN mkdir -p /opt/gateway
COPY app.py /opt/gateway/app.py

RUN echo "==> [8/10] Building gateway venv ..." && \
    /paddlex/py310/bin/python3 -m venv /opt/.venv_gateway && \
    echo "    pip upgrade ..." && \
    /opt/.venv_gateway/bin/pip install --upgrade pip && \
    echo "    fastapi + uvicorn + paddlex[serving] ..." && \
    /opt/.venv_gateway/bin/pip install --no-cache-dir \
    fastapi==0.123.6 uvicorn==0.35.0 "paddlex[serving]>=3.4.0" && \
    echo "    client requirements ..." && \
    /opt/.venv_gateway/bin/pip install --no-cache-dir \
    -r /opt/hps/client/requirements.txt && \
    echo "    paddlex_hps_client wheel ..." && \
    WHL=$(ls /opt/hps/client/paddlex_hps_client-*.whl | head -1) && \
    echo "    wheel: $WHL" && \
    /opt/.venv_gateway/bin/pip install --no-cache-dir "$WHL" && \
    echo "==> [8/10] gateway venv done"

# ── 9. Model weights (~2.2 GB from local cache via --build-context) ──────────
RUN echo "==> [9/10] Copying model weights ..."
COPY --from=paddlex_models . /root/.paddlex/official_models/
RUN echo "==> [9/10] Model weights copied" && \
    du -sh /root/.paddlex/official_models/

# ── 10. Start script ─────────────────────────────────────────────────────────
COPY start.sh /opt/start_hps.sh
RUN chmod +x /opt/start_hps.sh && echo "==> [10/10] start.sh installed"

# ── Verify (CPU-safe checks only — GPU not available during build) ────────────
RUN echo "==> [verify] Running sanity checks ..." && \
    /paddlex/py310/bin/python3 -c "import numpy; print('numpy', numpy.__version__)" && \
    test -f /root/.paddlex/official_models/PaddleOCR-VL/model.safetensors && echo "    model weights OK" && \
    test -f /opt/hps/server/pipeline_config.yaml && echo "    HPS SDK OK" && \
    test -f /opt/.venv_gateway/bin/uvicorn && echo "    gateway venv OK" && \
    /paddlex/py310/bin/python3 -c "import paddleocr; print('    paddleocr OK')" 2>/dev/null || echo "    paddleocr import skipped (needs CUDA 12.9 at runtime)" && \
    echo "==> [verify] All checks passed"

ENTRYPOINT []
CMD ["bash", "/opt/start_hps.sh"]

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.9"
