#!/usr/bin/env bash
# GLM-OCR provisioning script for vast.ai
# Base image: vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311
#
# Usage on vast.ai:
#   Option A (recommended): set PROVISIONING_SCRIPT to the raw URL of this file
#   Option B: paste contents into the "On-start script" field in the template
#
# Environment variables:
#   INSTANCES=1          (default) – single vLLM at 80 % GPU util
#   INSTANCES=2          – two vLLM instances at 38 % each + nginx LB on :8080
#   VLLM_PORT=8000       base port for vLLM (second instance uses VLLM_PORT+1)
#   GLMOCR_PORT=5002     glmocr server port
#   HF_TOKEN             Hugging Face token (required to download zai-org/GLM-OCR)
#
# Ports to expose in vast.ai template:
#   5002   → glmocr API  (/glmocr/parse)
#   8080   → Jupyter (already exposed by base image)
#   8384   → Syncthing (already exposed by base image)

set -euo pipefail

INSTANCES="${INSTANCES:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
GLMOCR_PORT="${GLMOCR_PORT:-5002}"
MODEL="zai-org/GLM-OCR"
MAX_MODEL_LEN=4096
MTP_JSON='{"method":"mtp","num_speculative_tokens":3}'
VENV=". /venv/main/bin/activate"
CONFIG=/etc/glmocr_config.yaml

# ── Install deps into the base-image venv ─────────────────────────────────────
echo "[glmocr] Installing vllm + glmocr..."
. /venv/main/bin/activate
uv pip install "vllm==0.20.2" "transformers>=5.3.0" "glmocr[selfhosted,server]"

# ── Patch page_loader to accept data:application/pdf;base64,... URIs ──────────
# glmocr only recognises PDFs via file paths or raw bytes; data URIs with the
# application/pdf MIME type fall through to _load_image() and are silently
# skipped.  This patch adds the missing branch in both _load_source() and
# _iter_source() so the benchmark can send base64-encoded PDFs over HTTP.
python3 - <<'PYEOF'
import pathlib

pkg = pathlib.Path("/venv/main/lib/python3.11/site-packages/glmocr/dataloader/page_loader.py")
src = pkg.read_text()

# ── _load_source ──────────────────────────────────────────────────────────────
OLD_LOAD = '''        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        # Detect PDF
        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            return self._load_pdf(file_path)

        # Otherwise load as a single image page
        return [self._load_image(source)]'''

NEW_LOAD = '''        # Handle PDF data URIs (data:application/pdf;base64,...)
        if source.startswith("data:application/pdf"):
            _, b64data = source.split(",", 1)
            import base64 as _b64
            return self._load_pdf_bytes(_b64.b64decode(b64data))

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        # Detect PDF
        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            return self._load_pdf(file_path)

        # Otherwise load as a single image page
        return [self._load_image(source)]'''

# ── _iter_source ──────────────────────────────────────────────────────────────
OLD_ITER = '''        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            yield from self._iter_pdf(file_path)
        else:
            yield self._load_image(source)'''

NEW_ITER = '''        # Handle PDF data URIs (data:application/pdf;base64,...)
        if source.startswith("data:application/pdf"):
            _, b64data = source.split(",", 1)
            import base64 as _b64
            yield from self._iter_pdf_bytes(_b64.b64decode(b64data))
            return

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            yield from self._iter_pdf(file_path)
        else:
            yield self._load_image(source)'''

patched = src
assert OLD_LOAD in patched, "patch target _load_source not found — glmocr version mismatch"
assert OLD_ITER in patched, "patch target _iter_source not found — glmocr version mismatch"
patched = patched.replace(OLD_LOAD, NEW_LOAD, 1)
patched = patched.replace(OLD_ITER, NEW_ITER, 1)
pkg.write_text(patched)
print("[patch] page_loader.py: data:application/pdf URI support added")
PYEOF

if [[ "$INSTANCES" == "2" ]]; then
    apt-get install -y -q nginx
fi

# ── Write glmocr config ───────────────────────────────────────────────────────
if [[ "$INSTANCES" == "2" ]]; then
    OCR_PORT=8080   # nginx LB
else
    OCR_PORT=$VLLM_PORT
fi

# Start from the package defaults so all layout/formatter fields are preserved,
# then patch only the values we need to override.
cp /venv/main/lib/python3.11/site-packages/glmocr/config.yaml "$CONFIG"
sed -i "s/port: 5002/port: ${GLMOCR_PORT}/"         "$CONFIG"
sed -i 's/enabled: true/enabled: false/'             "$CONFIG"
sed -i "s/api_port: 8080/api_port: ${OCR_PORT}/"     "$CONFIG"
sed -i 's/# device: null/device: "cuda:0"/'            "$CONFIG"
sed -i "s/batch_size: 1/batch_size: 2/"              "$CONFIG"
sed -i "s/max_tokens: 8192/max_tokens: 2048/"        "$CONFIG"

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
    --gpu-memory-utilization $([ "$INSTANCES" == "2" ] && echo "0.45" || echo "0.60") \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --tensor-parallel-size 1 \\
    --trust-remote-code \\
    --max-num-seqs 32 \\
    --no-enable-log-requests \\
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
    --max-num-seqs 32 \\
    --no-enable-log-requests \\
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
stdout_logfile=/var/log/supervisor/%(program_name)s.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=2
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
stdout_logfile=/var/log/supervisor/%(program_name)s.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=2
redirect_stderr=true
CONF

    cat > /etc/supervisor/conf.d/nginx-vllm.conf <<CONF
[program:nginx-vllm]
environment=PROC_NAME="%(program_name)s"
command=nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/%(program_name)s.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=2
redirect_stderr=true
CONF
fi

cat > /etc/supervisor/conf.d/glmocr.conf <<CONF
[program:glmocr]
environment=PROC_NAME="%(program_name)s",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
command=/opt/supervisor-scripts/glmocr.sh
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/var/log/supervisor/%(program_name)s.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=2
redirect_stderr=true
CONF

# ── Register with supervisor ──────────────────────────────────────────────────
deactivate
if ! pgrep -x supervisord > /dev/null; then
    supervisord -c /etc/supervisor/supervisord.conf
    sleep 2
fi
supervisorctl reread
supervisorctl update

echo "[glmocr] Services registered. Check status with: supervisorctl status"
echo "  glmocr  → http://0.0.0.0:${GLMOCR_PORT}/glmocr/parse"
if [[ "$INSTANCES" == "2" ]]; then
    echo "  vLLM LB → http://0.0.0.0:8080"
fi
