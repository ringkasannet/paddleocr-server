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
#   VLLM_SEMAPHORE=1     (default) – serialise vLLM submissions so requests pipeline
#                        through layout concurrently but enter vLLM one at a time;
#                        set to 0 to let all concurrent requests flood vLLM together
#
# Ports to expose in vast.ai template:
#   5002   → glmocr API  (/glmocr/parse)
#   8080   → Jupyter (already exposed by base image)
#   8384   → Syncthing (already exposed by base image)

set -euo pipefail

INSTANCES="${INSTANCES:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
GLMOCR_PORT="${GLMOCR_PORT:-5002}"
VLLM_SEMAPHORE="${VLLM_SEMAPHORE:-1}"
MODEL="zai-org/GLM-OCR"
MAX_MODEL_LEN=4096
MTP_JSON='{"method":"mtp","num_speculative_tokens":3}'
VENV=". /venv/main/bin/activate"
CONFIG=/etc/glmocr_config.yaml

# ── Install deps into the base-image venv ─────────────────────────────────────
echo "[glmocr] Installing vllm + glmocr..."
. /venv/main/bin/activate
uv pip install "vllm==0.20.2" "transformers>=5.3.0" "glmocr[selfhosted,server]" gunicorn

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

# ── Patch _workers.py to serialise concurrent layout GPU calls ────────────────
# Multiple concurrent HTTP requests each spawn their own layout_worker thread,
# all sharing the single PPDocLayoutDetector on GPU.  Without a lock they can
# call layout_detector.process() simultaneously, causing CUDA OOM.
# A Semaphore(1) ensures only one GPU forward pass runs at a time; page loading
# and vLLM OCR calls remain fully concurrent.
python3 - <<'PYEOF'
import pathlib

pkg = pathlib.Path("/venv/main/lib/python3.11/site-packages/glmocr/pipeline/_workers.py")
src = pkg.read_text()

if "_LAYOUT_GPU_SEMAPHORE" in src:
    print("[patch] _workers.py: layout semaphore already present, skipping")
else:
    OLD_IMPORT = '''import queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed'''

    NEW_IMPORT = '''import queue
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# One GPU layout forward pass at a time across all concurrent requests.
_LAYOUT_GPU_SEMAPHORE = threading.Semaphore(1)'''

    OLD_PROCESS = '''    try:
        layout_results, vis_images = layout_detector.process(
            batch_images,
            save_visualization=save_visualization,
            global_start_idx=global_start_idx,
            use_polygon=use_polygon,
        )
        if vis_images:
            state.layout_vis_images.update(vis_images)'''

    NEW_PROCESS = '''    try:
        with _LAYOUT_GPU_SEMAPHORE:
            layout_results, vis_images = layout_detector.process(
                batch_images,
                save_visualization=save_visualization,
                global_start_idx=global_start_idx,
                use_polygon=use_polygon,
            )
        if vis_images:
            state.layout_vis_images.update(vis_images)'''

    assert OLD_IMPORT in src, "patch target (imports) not found in _workers.py — glmocr version mismatch"
    assert OLD_PROCESS in src, "patch target (_flush_layout_batch) not found in _workers.py — glmocr version mismatch"
    patched = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    patched = patched.replace(OLD_PROCESS, NEW_PROCESS, 1)
    pkg.write_text(patched)
    print("[patch] _workers.py: layout GPU semaphore(1) applied")

# ── Patch _workers.py to serialise vLLM submissions (pipeline mode) ──────────
# When enabled, only one request's recognition_worker submits to vLLM at a time.
# Layout for queued requests continues concurrently, buffering into region_queue.
# Result: earlier requests return faster; later ones wait their turn.
# Disable with VLLM_SEMAPHORE=0 to revert to flooding vLLM with all requests.
if [[ "$VLLM_SEMAPHORE" == "1" ]]; then
python3 - <<'PYEOF'
import pathlib

pkg = pathlib.Path("/venv/main/lib/python3.11/site-packages/glmocr/pipeline/_workers.py")
src = pkg.read_text()

if "_VLLM_SEMAPHORE" in src:
    print("[patch] _workers.py: vLLM semaphore already present, skipping")
else:
    OLD_SEM = '''# One GPU layout forward pass at a time across all concurrent requests.
_LAYOUT_GPU_SEMAPHORE = threading.Semaphore(1)'''

    NEW_SEM = '''# One GPU layout forward pass at a time across all concurrent requests.
_LAYOUT_GPU_SEMAPHORE = threading.Semaphore(1)

# One vLLM submission batch at a time across all concurrent requests.
_VLLM_SEMAPHORE = threading.Semaphore(1)'''

    OLD_START = '''    """Consume regions, run parallel OCR, store results."""
    executor = None
    try:'''

    NEW_START = '''    """Consume regions, run parallel OCR, store results."""
    _VLLM_SEMAPHORE.acquire()
    executor = None
    try:'''

    OLD_FINALLY = '''    finally:
        state.drain_queue(state.region_queue)'''

    NEW_FINALLY = '''    finally:
        _VLLM_SEMAPHORE.release()
        state.drain_queue(state.region_queue)'''

    assert OLD_SEM in src, "patch target (_LAYOUT_GPU_SEMAPHORE) not found — glmocr version mismatch"
    assert OLD_START in src, "patch target (recognition_worker start) not found — glmocr version mismatch"
    assert OLD_FINALLY in src, "patch target (recognition_worker finally) not found — glmocr version mismatch"
    patched = src.replace(OLD_SEM, NEW_SEM, 1).replace(OLD_START, NEW_START, 1).replace(OLD_FINALLY, NEW_FINALLY, 1)
    pkg.write_text(patched)
    print("[patch] _workers.py: vLLM semaphore(1) applied")
PYEOF
else
    echo "[patch] vLLM semaphore skipped (VLLM_SEMAPHORE=0)"
fi

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

# ── Write Gunicorn WSGI shim ──────────────────────────────────────────────────
mkdir -p /opt/glmocr
cat > /opt/glmocr/wsgi.py <<'PYEOF'
"""Gunicorn WSGI entry-point for glmocr server.

Config path is read from the GLMOCR_CONFIG environment variable
(default: /etc/glmocr_config.yaml).
"""
import os
import multiprocessing

from glmocr.config import load_config
from glmocr.server import create_app

multiprocessing.set_start_method("spawn", force=True)

_config_path = os.environ.get("GLMOCR_CONFIG", "/etc/glmocr_config.yaml")
config = load_config(_config_path)
app = create_app(config)
app.config["pipeline"].start()

import logging as _logging
_gunicorn_logger = _logging.getLogger("gunicorn.error")
_glmocr_logger   = _logging.getLogger("glmocr")
_glmocr_logger.handlers  = _gunicorn_logger.handlers
_glmocr_logger.setLevel(_gunicorn_logger.level or _logging.INFO)
PYEOF

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
    --gpu-memory-utilization $([ "$INSTANCES" == "2" ] && echo "0.45" || echo "0.80") \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --tensor-parallel-size 1 \\
    --trust-remote-code \\
    --max-num-seqs 32 \\
    --max-num-batched-tokens 32768 \\
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
    --max-num-batched-tokens 32768 \\
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
exec gunicorn wsgi:app \
    --bind 0.0.0.0:${GLMOCR_PORT} \
    --workers 1 \
    --worker-class gthread \
    --threads 2 \
    --timeout 300 \
    --chdir /opt/glmocr \
    --access-logfile - \
    --error-logfile - \
    --capture-output
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
environment=PROC_NAME="%(program_name)s",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",GLMOCR_CONFIG="${CONFIG}"
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
