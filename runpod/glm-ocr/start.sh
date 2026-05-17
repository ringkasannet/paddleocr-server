#!/usr/bin/env bash
# GLM-OCR provisioning script for vast.ai
# Base image: vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311
#
# Usage on vast.ai:
#   Option A (recommended): set PROVISIONING_SCRIPT to the raw URL of this file
#   Option B: paste contents into the "On-start script" field in the template
#
# Environment variables:
#   INSTANCES=1          (default) – single vLLM at 75 % GPU util
#   INSTANCES=2          – two vLLM instances at 38 % each + nginx LB on :8080
#   VLLM_PORT=8000       base port for vLLM (second instance uses VLLM_PORT+1)
#   GLMOCR_PORT=5002     glmocr server port
#   HF_TOKEN             Hugging Face token (required to download THUDM/GLM-OCR)
#
# Ports to expose in vast.ai template:
#   5002   → glmocr API  (/glmocr/parse)
#   8080   → Jupyter (already exposed by base image)
#   8384   → Syncthing (already exposed by base image)

set -euo pipefail

INSTANCES="${INSTANCES:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
GLMOCR_PORT="${GLMOCR_PORT:-5002}"
MODEL="THUDM/GLM-OCR"
MAX_MODEL_LEN=4096
MTP_JSON='{"method":"mtp","num_speculative_tokens":3}'
VENV=". /venv/main/bin/activate"
CONFIG=/etc/glmocr_config.yaml

# ── Install deps into the base-image venv ─────────────────────────────────────
echo "[glmocr] Installing vllm + glmocr..."
. /venv/main/bin/activate
uv pip install "vllm>=0.9.0" "glmocr[selfhosted,server]"

if [[ "$INSTANCES" == "2" ]]; then
    apt-get install -y -q nginx
fi

# ── Write glmocr config ───────────────────────────────────────────────────────
if [[ "$INSTANCES" == "2" ]]; then
    OCR_PORT=8080   # nginx LB
else
    OCR_PORT=$VLLM_PORT
fi

cat > "$CONFIG" <<YAML
server:
  host: 0.0.0.0
  port: ${GLMOCR_PORT}

maas:
  enabled: false

ocr_api:
  api_host: 127.0.0.1
  api_port: ${OCR_PORT}
  model: glm-ocr
  api_path: /v1/chat/completions
  api_mode: openai
  request_timeout: 120
  max_connections: 128
  max_workers: 32

layout:
  model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors
  batch_size: 4
  cuda_visible_devices: "0"
  device: "cuda:0"
YAML

# ── Write supervisor wrapper scripts ──────────────────────────────────────────
mkdir -p /opt/supervisor-scripts

# vLLM instance 0 (always created)
cat > /opt/supervisor-scripts/vllm-0.sh <<SCRIPT
#!/bin/bash
. /venv/main/bin/activate
exec python -m vllm.entrypoints.openai.api_server \\
    --model "${MODEL}" \\
    --served-model-name glm-ocr \\
    --port ${VLLM_PORT} \\
    --gpu-memory-utilization $([ "$INSTANCES" == "2" ] && echo "0.38" || echo "0.75") \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --tensor-parallel-size 1 \\
    --trust-remote-code \\
    --speculative-config '${MTP_JSON}'
SCRIPT
chmod +x /opt/supervisor-scripts/vllm-0.sh

if [[ "$INSTANCES" == "2" ]]; then
    # vLLM instance 1
    cat > /opt/supervisor-scripts/vllm-1.sh <<SCRIPT
#!/bin/bash
. /venv/main/bin/activate
exec python -m vllm.entrypoints.openai.api_server \\
    --model "${MODEL}" \\
    --served-model-name glm-ocr \\
    --port $((VLLM_PORT + 1)) \\
    --gpu-memory-utilization 0.38 \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --tensor-parallel-size 1 \\
    --trust-remote-code \\
    --speculative-config '${MTP_JSON}'
SCRIPT
    chmod +x /opt/supervisor-scripts/vllm-1.sh

    # nginx LB config
    cat > /etc/nginx/conf.d/vllm_lb.conf <<NGINX
upstream vllm_backend {
    server 127.0.0.1:${VLLM_PORT};
    server 127.0.0.1:$((VLLM_PORT + 1));
}
server {
    listen 8080;
    location / {
        proxy_pass http://vllm_backend;
        proxy_read_timeout 300;
        proxy_connect_timeout 10;
    }
}
NGINX
fi

# glmocr wrapper — waits for vLLM to be healthy before starting
cat > /opt/supervisor-scripts/glmocr.sh <<SCRIPT
#!/bin/bash
. /venv/main/bin/activate
echo "[glmocr] Waiting for vLLM on port ${OCR_PORT}..."
until curl -sf "http://127.0.0.1:${OCR_PORT}/health" > /dev/null 2>&1; do sleep 3; done
echo "[glmocr] vLLM ready, starting glmocr server..."
exec python -m glmocr.server --config ${CONFIG}
SCRIPT
chmod +x /opt/supervisor-scripts/glmocr.sh

# ── Write supervisor configs ──────────────────────────────────────────────────

cat > /etc/supervisor/conf.d/vllm-0.conf <<CONF
[program:vllm-0]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/vllm-0.sh
autostart=true
autorestart=true
startsecs=10
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
CONF

if [[ "$INSTANCES" == "2" ]]; then
    cat > /etc/supervisor/conf.d/vllm-1.conf <<CONF
[program:vllm-1]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/vllm-1.sh
autostart=true
autorestart=true
startsecs=10
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
CONF

    cat > /etc/supervisor/conf.d/nginx-vllm.conf <<CONF
[program:nginx-vllm]
environment=PROC_NAME="%(program_name)s"
command=nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
CONF
fi

cat > /etc/supervisor/conf.d/glmocr.conf <<CONF
[program:glmocr]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/glmocr.sh
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
CONF

# ── Register with supervisor (already running in base image) ──────────────────
deactivate
supervisorctl reread
supervisorctl update

echo "[glmocr] Services registered. Check status with: supervisorctl status"
echo "  glmocr  → http://0.0.0.0:${GLMOCR_PORT}/glmocr/parse"
if [[ "$INSTANCES" == "2" ]]; then
    echo "  vLLM LB → http://0.0.0.0:8080"
fi
