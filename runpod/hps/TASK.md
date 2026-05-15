# PaddleOCR HPS on RunPod — Handoff Document

## Goal

Run the full PaddleOCR HPS (High-Performance Serving) stack on a single RunPod
pod as three separate processes sharing one GPU:

```
Client → FastAPI Gateway (port 8080)
              ↓ gRPC (paddlex_hps_client)
          Triton Server (ports 8000/8001)   ← PP-DocLayoutV3 layout detection
              ↓ HTTP OpenAI-compatible API
          vLLM Server (port 8118)            ← PaddleOCR-VL-1.5-0.9B
```

Everything lives in `/workspace` (persistent network volume). The pod image is
never customized — on every pod start, the servers launch from `/workspace`.

---

## Pod Environment

| Item | Value |
|------|-------|
| Image | `ringkasannet/paddleocr-hps:paddlex3.4-gpu` (retagged `paddlex/hps:paddlex3.4-gpu`) |
| GPU | RTX A4500, 20 GB VRAM |
| CUDA driver | 580.x, supports CUDA 13.0 |
| OS | Ubuntu 20.04 (inside container) |
| Container Python | `/paddlex/py310/bin/python3` = Python 3.10.4 |

---

## What Is Already Installed and Working

### In the Docker image (read-only, always present)
| Component | Path | Status |
|-----------|------|--------|
| Triton 2.x binary | `/opt/tritonserver/bin/tritonserver` | ✅ Starts, GPU found |
| Triton Python backend + stub | `/opt/tritonserver/backends/python/` | ✅ Verified working (Gate 4 passed) |
| PaddlePaddle + paddlex | `/paddlex/py310/lib/python3.10/site-packages/` | ✅ Present |
| `paddlex_hps_server` package | `/paddlex/py310/lib/python3.10/site-packages/paddlex_hps_server/` | ✅ Present |
| vllm + flash-attn | `/paddlex/py310/lib/python3.10/site-packages/` | ✅ Installed via `install_genai_server_deps` |

### In /workspace (network volume, persistent)
| Component | Path | Status |
|-----------|------|--------|
| HPS SDK model repo | `/workspace/hps/server/model_repo/layout-parsing/` | ✅ Downloaded |
| HPS SDK model repo | `/workspace/hps/server/model_repo/restructure-pages/` | ✅ Downloaded |
| HPS pipeline config | `/workspace/hps/server/pipeline_config.yaml` | ✅ Patched (localhost:8118) |
| HPS client SDK | `/workspace/hps/client/` | ✅ Downloaded |
| `.venv_vlm` | `/workspace/.venv_vlm/` | ✅ paddlepaddle-gpu 3.2.1 + paddleocr 3.5.0 |
| `.venv_gateway` | `/workspace/.venv_gateway/` | ✅ fastapi + uvicorn + paddlex[serving] + paddlex_hps_client 0.3.0 |
| Gateway app | `/workspace/gateway/app.py` | ✅ Downloaded from GitHub |

### Triton test (Gate 4) — PASSED ✅
```
hello_world | 1 | READY
successfully loaded 'hello_world' version 1
```
Triton Python backend works. Command that works:
```bash
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH \
  /opt/tritonserver/bin/tritonserver \
  --model-repository=/tmp/model_repo \
  --backend-directory=/opt/tritonserver/backends \
  --backend-config=python,python-runtime=/paddlex/py310/bin/python3 \
  --allow-metrics=false --log-info=true
```

---

## Current Blocker: vLLM Server Won't Start

### The problem
`paddleocr genai_server` is NOT built into paddleocr by default. It is a
dynamically registered subcommand that only appears after vllm is installed
in the same Python environment.

`install_genai_server_deps vllm` installs vllm into `/paddlex/py310/` (hardcoded
inside paddlex's install script). But `paddleocr` CLI is in `/workspace/.venv_vlm/`,
not `/paddlex/py310/`.

### What has been tried and failed

**Attempt 1**: Run `genai_server` from `.venv_vlm/bin/paddleocr` directly
→ "invalid choice: genai_server" — vllm not visible from that venv

**Attempt 2**: PYTHONPATH mixing — add `/paddlex/py310/lib/python3.10/site-packages`
to PYTHONPATH so `.venv_vlm/bin/paddleocr` can find vllm
→ numpy version conflict: `.venv_vlm/` has numpy 2.2.6, `/paddlex/py310/` has
  matplotlib compiled for numpy 1.x → `ImportError: numpy.core.multiarray failed to import`

**Attempt 3**: Run from `/paddlex/py310/bin/python3 -m paddleocr genai_server`
→ `ModuleNotFoundError: No module named 'paddleocr'` — paddleocr is NOT installed in `/paddlex/py310/`

### Root cause summary
| Environment | Has paddleocr CLI | Has vllm | Has paddlepaddle |
|-------------|------------------|----------|-----------------|
| `/paddlex/py310/` | ❌ No | ✅ Yes | ✅ Yes |
| `/workspace/.venv_vlm/` | ✅ Yes | ❌ No | ✅ Yes |

Neither environment has both paddleocr AND vllm.

---

## What To Do Next

### Step 1 — Find the vllm package names in `/paddlex/py310/`

```bash
/paddlex/py310/bin/pip list 2>/dev/null | grep -iE "vllm|xformers|genai|triton"
```

This tells us exactly which packages `install_genai_server_deps` installed.

### Step 2 — Fix numpy in `/paddlex/py310/` (broken by flash-attn install)

```bash
/paddlex/py310/bin/pip install "numpy==1.26.4" --force-reinstall --no-deps
/paddlex/py310/bin/python3 -c "import matplotlib; print('matplotlib OK')"
```

### Step 3 — Choose one of these approaches

#### Option A: Install vllm packages into `.venv_vlm/` (recommended)
Once you know the package names from Step 1, install the same packages into
`.venv_vlm/`. Then `genai_server` will appear in `.venv_vlm/bin/paddleocr` because
vllm is importable there. No PYTHONPATH needed.

```bash
# Example (replace with actual package names from Step 1)
/workspace/.venv_vlm/bin/pip install vllm==<version>
/workspace/.venv_vlm/bin/paddleocr --help 2>&1 | grep genai_server
```

#### Option B: Install paddleocr into `/paddlex/py310/`
Install paddleocr 3.5.0 directly into the image's Python where vllm already lives.

```bash
/paddlex/py310/bin/pip install paddleocr==3.5.0
/paddlex/py310/bin/python3 -m paddleocr genai_server --help
```

Then run genai_server from there:
```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export HF_HOME=/workspace/models/hf_cache
/paddlex/py310/bin/python3 -m paddleocr genai_server \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --backend vllm \
  --host 0.0.0.0 \
  --port 8118 &
```

### Step 4 — Once vLLM is running, test it

```bash
sleep 120  # wait for model to load (~2 min first run, downloads ~2 GB)
curl -s http://localhost:8118/health && echo "vLLM: OK"
```

### Step 5 — Start Triton with HPS models

```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH \
  /opt/tritonserver/bin/tritonserver \
  --model-repository=/workspace/hps/server/model_repo \
  --backend-directory=/opt/tritonserver/backends \
  --backend-config=python,python-runtime=/paddlex/py310/bin/python3 \
  --backend-config=python,shm-default-byte-size=67108864 \
  --http-port=8000 --grpc-port=8001 \
  --model-control-mode=explicit \
  --load-model=layout-parsing \
  --load-model=restructure-pages \
  --allow-metrics=false --log-info=true 2>&1 | tail -20 &

sleep 30
curl -s http://localhost:8000/v2/health/ready && echo "Triton: OK"
curl -s http://localhost:8000/v2/models/layout-parsing/ready && echo "layout-parsing: READY"
curl -s http://localhost:8000/v2/models/restructure-pages/ready && echo "restructure-pages: READY"
```

### Step 6 — Start gateway

```bash
# Determine which Python has vllm (from Step 3 result)
# Use whichever venv successfully runs genai_server

HPS_TRITON_URL="localhost:8001" \
HPS_VLM_URL="http://localhost:8118" \
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=16 \
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=64 \
HPS_INFERENCE_TIMEOUT=600 \
/workspace/.venv_gateway/bin/uvicorn app:app \
  --app-dir /workspace/gateway \
  --host 0.0.0.0 --port 8080 --workers 4 &

sleep 5
curl -s http://localhost:8080/health && echo "Gateway: OK"
curl -s http://localhost:8080/health/ready && echo "Gateway: READY"
```

---

## Key Paths Reference

| Item | Path |
|------|------|
| Triton binary | `/opt/tritonserver/bin/tritonserver` |
| Triton Python backend | `/opt/tritonserver/backends/python/` |
| Image Python | `/paddlex/py310/bin/python3` |
| vllm location | `/paddlex/py310/lib/python3.10/site-packages/` |
| paddleocr CLI | `/workspace/.venv_vlm/bin/paddleocr` |
| paddlepaddle-gpu | in both `/paddlex/py310/` and `/workspace/.venv_vlm/` |
| HPS model repo | `/workspace/hps/server/model_repo/` |
| HPS pipeline config | `/workspace/hps/server/pipeline_config.yaml` (server_url: http://localhost:8118/v1) |
| Gateway venv | `/workspace/.venv_gateway/bin/` |
| Gateway app | `/workspace/gateway/app.py` |
| Models cache | `/workspace/models/hf_cache/` |
