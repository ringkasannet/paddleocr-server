# PaddleOCR HPS on RunPod — Problems & Solutions

Complete record of every problem encountered building
`ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready` and running it on RunPod.

---

## Final Architecture

```
Client
  ↓  POST /layout-parsing  {file: <base64>, fileType: 0}
FastAPI Gateway       :8080   /opt/.venv_gateway/bin/uvicorn
  ↓  gRPC (paddlex_hps_client)
Triton Server         :8000/:8001   /opt/tritonserver/bin/tritonserver
  │  layout-parsing model  → PP-DocLayoutV3  (GPU)
  │  restructure-pages     → CPU post-processing
  ↓  HTTP OpenAI-compatible  /v1/chat/completions
vLLM Server           :8118   python3 -m paddleocr genai_server
                               model: PaddleOCR-VL-1.5-0.9B
```

**Concurrency model:** Triton dynamic batching (max_batch_size=16, 2 instances).
vLLM continuous batching (max_concurrency=32). Gateway semaphore caps at 16
concurrent inference requests. One pod handles 8–16 simultaneous document requests.

---

## Environment Reference

| Item | Value |
|------|-------|
| Image (working) | `ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready` |
| Base image | `ringkasannet/paddleocr-hps:paddlex3.4-gpu` |
| Container Python | `/paddlex/py310/bin/python3` (Python 3.10.4) |
| Container CUDA toolkit | 11.8 (nvcc) |
| PyTorch build | 2.8.0+**cu128** (CUDA 12.8) — mismatch with nvcc, matters for flash-attn |
| Triton binary | `/opt/tritonserver/bin/tritonserver` |
| HPS SDK | `/opt/hps/server/` |
| Gateway venv | `/opt/.venv_gateway/` |
| Model weights | `/root/.paddlex/official_models/` (baked into image) |
| vLLM CUDA cache | `/root/.cache/vllm` (optional: symlink to network volume) |

---

## All Problems and Fixes

### Problem 1 — `wget`/`curl` missing from base image
**Error:** `bash: wget: command not found`  
**Fix:**
```dockerfile
RUN apt-get update -qq && apt-get install -y curl wget -qq
```

---

### Problem 2 — `config.pbtxt` missing in Triton model repo
**Error:** `failed to open text file .../layout-parsing/config.pbtxt: No such file or directory`  
**Cause:** HPS SDK ships `config_gpu.pbtxt` and `config_cpu.pbtxt` but Triton expects `config.pbtxt`.  
**Fix:**
```bash
cp model_repo/layout-parsing/config_gpu.pbtxt \
   model_repo/layout-parsing/config.pbtxt
```
Added to Dockerfile as a `sed`/`cp` step after SDK extraction.

---

### Problem 3 — `genai_server` subcommand missing
**Error:** `paddleocr: error: argument subcommand: invalid choice: 'genai_server'`  
**Cause:** `genai_server` is dynamically registered only when `vllm` is importable
in the same Python environment as `paddleocr`.  
**Fix:** Install both `paddleocr` and `vllm` into the same environment (`/paddlex/py310/`).

---

### Problem 4 — flash-attn source compilation fails
**Error:** `RuntimeError: The detected CUDA version (11.8) mismatches PyTorch (12.8)`  
**Cause:** Container has CUDA 11.8 nvcc but torch 2.8.0+cu128. Cannot compile from source.  
**Fix:** Use a pre-built wheel:
```dockerfile
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"
```

---

### Problem 5 — numpy 2.x breaks pandas and matplotlib
**Error:** `ValueError: numpy.dtype size changed` / `AttributeError: _ARRAY_API not found`  
**Cause:** flash-attn (and vllm) upgrade numpy to 2.x. Compiled packages (pandas,
old matplotlib) were built against numpy 1.x ABI.  
**Fix:** Pin numpy **after every other install**. Do NOT change this ordering:
```dockerfile
# Step 7 in Dockerfile — MUST be the absolute last pip install
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "numpy==1.26.4" --force-reinstall --no-deps
```
Also upgrade matplotlib so it is numpy-2.x compatible:
```dockerfile
RUN /paddlex/py310/bin/pip install --no-cache-dir "matplotlib>=3.9" --upgrade
```

---

### Problem 6 — `paddleocr install_genai_server_deps vllm` fails during Docker build
**Error:** `RuntimeError: CUDA is not available` / `is_cuda_available()` fails  
**Cause:** The command internally calls PaddlePaddle's `is_cuda_available()` → tries
to load `libcuda.so.1` → not present at image build time (no GPU during build).  
**Fix:** Directly install the packages it would have installed:
```dockerfile
RUN /paddlex/py310/bin/pip install --no-cache-dir \
    "vllm==0.10.2" \
    "transformers<5.0.0" \
    "einops" \
    "uvloop"
```

---

### Problem 7 — `paddlex[ocr]` extras missing
**Error:** `DependencyError: PaddleOCR-VL-1.5 requires additional dependencies. pip install "paddlex[ocr]"`  
**Fix:**
```dockerfile
RUN PADDLEX_VER=$(/paddlex/py310/bin/pip show paddlex | grep ^Version | awk '{print $2}') && \
    /paddlex/py310/bin/pip install --no-cache-dir "paddlex[ocr]==$PADDLEX_VER"
```

---

### Problem 8 — `pipeline_config.yaml` using slow `native` backend (performance killer)
**Error (symptom):** 3-page PDF takes 38 seconds instead of ~1.2 seconds.  
**Cause:** Default `pipeline_config.yaml` sets `backend: native` (slow PaddlePaddle
inference) instead of delegating to the vLLM server.  
**Fix:** Patch the config to use `vllm-server`:
```dockerfile
RUN /paddlex/py310/bin/python3 -c "
import yaml
path = '/opt/hps/server/pipeline_config.yaml'
cfg = yaml.safe_load(open(path))
cfg['SubModules']['VLRecognition']['genai_config'] = {
    'backend': 'vllm-server',
    'server_url': 'http://localhost:8118/v1',
    'max_concurrency': 32,
}
yaml.dump(cfg, open(path, 'w'), default_flow_style=False, allow_unicode=True, sort_keys=False)
"
```

---

### Problem 9 — OOM at `gpu_memory_utilization: 0.75`
**Error:** `ResourceExhaustedError: Out of memory on GPU 0, 19.61 GB allocated`  
**Cause:** Triton's PaddlePaddle layout detection (PP-DocLayoutV3) runs on GPU
alongside vLLM. vLLM at 0.75 leaves only ~87 MB free — not enough for Triton
during concurrent requests.  
**GPU split on RTX A4500 (20 GB):**
| Component | VRAM |
|-----------|------|
| vLLM KV cache (0.50 util) | ~10 GB |
| Triton + PP-DocLayoutV3 | ~4–5 GB |
| Headroom | ~5 GB |
**Fix:** Default `gpu_memory_utilization: 0.50`. Override via env: `GPU_MEMORY_UTILIZATION=0.65`.

---

### Problem 10 — `ln: failed to create symbolic link '/root/.cache/vllm'`
**Error:** `ln: failed to create symbolic link '/root/.cache/vllm': No such file or directory`  
**Cause:** `/root/.cache` does not exist in a fresh container.  
**Fix:** Added `mkdir -p /root/.cache` before the symlink in `start.sh`.

---

### Problem 11 — Port 8080 conflict on pod restart
**Error:** `[Errno 98] Address already in use`  
**Cause:** `CMD ["sleep", "infinity"]` in the Dockerfile keeps the container alive.
Manually running `start.sh` again starts duplicate processes.  
**Fix:** Added a `pkill` + wait loop at the top of `start.sh`:
```bash
pkill -f "tritonserver|uvicorn|genai_server" 2>/dev/null || true
for i in $(seq 1 35); do
    pgrep -f "tritonserver|uvicorn|genai_server" > /dev/null 2>&1 || break
    sleep 1
done
```

---

### Problem 12 — `${PIPESTATUS[0]}` captures exit code, not PID
**Error:** `kill: (0): No such process`  
**Cause:** `cmd | sed ... & PID=$!` captures sed's PID, not the main process.
`${PIPESTATUS[0]}` in bash returns the exit code, not the PID.  
**Fix:** Use `pgrep` after a brief sleep:
```bash
paddleocr genai_server ... 2>&1 | sed -u 's/^/[vllm] /' &
sleep 2
VLLM_PID=$(pgrep -n -f "paddleocr genai_server" 2>/dev/null || true)
```

---

### Problem 13 — `CUDA_VISIBLE_DEVICES` unbound variable with `set -u`
**Error:** `start_hps.sh: line 52: CUDA_VISIBLE_DEVICES: unbound variable`  
**Cause:** Script uses `set -euo pipefail`. Referencing an unset variable aborts.  
**Fix:** Use `:-` default expansion which is safe under `set -u`:
```bash
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    unset CUDA_VISIBLE_DEVICES
fi
```

---

### Problem 14 — `cuInit(0)` returns 999 on RunPod driver 580 nodes
**Error:** `RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment`  
**Symptom:** `libcuda.so.1 → libcuda.so.580.65.06` correctly symlinked. Kernel module
version matches. `nvidia-smi` works. But `torch.cuda.is_available()` returns False.  
**Cause:** RunPod infrastructure issue — the container does not have GPU compute
permissions on some driver 580 nodes despite device files being accessible.  
**Diagnosis:**
```bash
python3 -c "import ctypes; l=ctypes.CDLL('libcuda.so.1'); print(l.cuInit(0))"
# Returns 999 = CUDA_ERROR_UNKNOWN
```
**Fix/Workaround:** This is a RunPod node-level bug. Switch to Vast.ai where CUDA
works correctly. On RunPod, try a different pod/node.  
**Attempted fix that did NOT help:** Adding `ENV NVIDIA_REQUIRE_CUDA="cuda>=11.8"` to
override the base image's `driver>=470,driver<471` constraint. The env var is set
correctly in the container but cuInit still returns 999.

---

### Problem 15 — `NVIDIA_REQUIRE_CUDA` driver constraint blocks libcuda injection
**Symptom:** `nvidia-smi` works, `/dev/nvidia0` exists, but CUDA unavailable.  
**Cause:** Base image sets `NVIDIA_REQUIRE_CUDA=driver>=470,driver<471`. On hosts with
driver 580, the NVIDIA Container Runtime interprets this as a constraint violation and
skips injecting `libcuda.so.1`. Container falls back to the stub library which has no
real GPU access.  
**Fix (added to Dockerfile):**
```dockerfile
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.8"
```
Placed at the bottom of the Dockerfile so edits don't invalidate cached layers above.
This fixed the issue on Vast.ai but NOT on the specific RunPod nodes with the compute
permission bug (Problem 14).

---

### Problem 16 — `gateway wait_healthy` timeout too short
**Error:** `[ERROR] Gateway did not become healthy within 30s`  
**Cause:** Gateway with 4 uvicorn workers takes ~60–90s to start on a loaded pod.  
**Fix:** Increased wait timeout to 60s in `start.sh`.

---

### Problem 17 — Docker BuildKit not available for `--build-context`
**Error:** `ERROR: unknown flag: --build-context`  
**Cause:** Older Docker versions or Docker not in BuildKit mode.  
**Fix:**
```bash
# Models are from official PaddlePaddle HuggingFace repos:
#   PaddlePaddle/PaddleOCR-VL-1.5   (1.93 GB)
#   PaddlePaddle/PP-DocLayoutV2      (214 MB)
# build.sh downloads them automatically if not cached locally.
DOCKER_BUILDKIT=1 docker build --build-context paddlex_models=$HOME/.paddlex/official_models ...
```

---

### Problem 18 — HPS image is 40 GB (unexpectedly large)
**Observation:** `docker images` showed 40+ GB.  
**Reality:** Docker reports uncompressed size. Compressed on Docker Hub is ~20 GB.
The base image alone is 13 GB compressed. This is expected for a CUDA + ML stack.

---

### Problem 19 — PDFium error when sending PDF via URL
**Error:** `[paddlex] Failed to read input file: Failed to decode image bytes` (HTTP 422)  
**Cause 1 (original):** Sending `fileType: 0` with a tmpfiles.org URL — the service
fetched the URL and got an HTML page instead of binary PDF (tmpfiles.org requires `/dl/`
prefix for direct download).  
**Cause 2:** Sending base64-encoded PDF with `fileType: 1` (image) instead of `fileType: 0` (PDF).  
**API clarification:**
| fileType | Meaning |
|----------|---------|
| 0 | PDF file (URL or base64) |
| 1 | Image file (URL or base64) |
**Fix:** Send base64-encoded PDF bytes with `fileType: 0`:
```python
json={"file": base64.b64encode(pdf_bytes).decode(), "fileType": 0, "visualize": False}
```

---

### Problem 20 — RunPod proxy (HTTP port) has ~140s hard timeout
**Error:** `HTTP 524` — Cloudflare cuts the connection after ~140s.  
**Cause:** RunPod's public HTTP proxy (`*.proxy.runpod.net`) routes through Cloudflare
which enforces a ~100–140s timeout. Long OCR requests (multiple queued) exceed this.  
**vLLM logs show 200 OK** — the server completed fine, the proxy just killed the client connection.  
**Fix:** Use **TCP port exposure** instead of HTTP port:
- In RunPod pod template, add `8080` to "Expose TCP Ports" (not HTTP Ports)
- Connect via `http://<public-ip>:<external-port>/layout-parsing`
- TCP ports are direct forwarding with no proxy — no timeout limit
- Firewall to your Elixir server IP only:
  ```bash
  iptables -A INPUT -p tcp --dport 8080 ! -s <your-server-ip> -j DROP
  ```

---

### Problem 21 — Triton `max_batch_size` and `instance_group count` defaults too conservative
**Default:** `max_batch_size: 8`, `count: 1`  
**Fix in Dockerfile:**
```dockerfile
RUN sed -i 's/max_batch_size: 8/max_batch_size: 16/' config.pbtxt && \
    sed -i 's/count: 1/count: 2/' config.pbtxt
```
More batching → higher GPU utilization under concurrent load.

---

## Performance Benchmarks

Tested on **RTX A4500 20 GB** (SM 8.6), `gpu_memory_utilization=0.50`.

### HPS stack throughput

| Concurrency | Total time | Pages/sec | GPU util avg | VRAM peak |
|-------------|-----------|-----------|--------------|-----------|
| 1 (sequential) | 26.3s | 1.18 p/s | 35% | 12 GB (59%) |
| 4 concurrent | 11.4s | **2.72 p/s** | 43% | 12 GB (59%) |
| 8 concurrent | 12.2s | 2.54 p/s | 43% | 13.6 GB (67%) |

Sweet spot: **4 concurrent** — going to 8 slightly over-batches vLLM.

### Section 4 pod (paddlex --serve, no HPS)

| Concurrency | Total time | Pages/sec |
|-------------|-----------|-----------|
| 1 | 29.2s | 1.06 p/s |
| 4 | 26.0s | 1.19 p/s |
| 8 | 25.6s | 1.21 p/s |

Section 4 serializes internally — concurrency above 4 gives nothing.
**HPS is 2.3× faster** than Section 4 at full concurrency.

### GPU compatibility

| GPU | SM | Works? | Notes |
|-----|----|--------|-------|
| RTX A4500 | 8.6 | ✅ | Tested: 2.2 p/s |
| RTX 4070 Ti Super | 8.9 | ✅ | Tested: 1.9 p/s |
| V100 | 7.0 | ❌ | No flash-attn |
| RTX 50xx / Blackwell | 12.0 | ❌ | PaddlePaddle not compiled for SM 120 |

Minimum: **SM 8.0** (flash-attn requirement).

---

## All Tunable Environment Variables

Set these in the RunPod pod template — no rebuild needed.

### start.sh

| Variable | Default | Effect |
|----------|---------|--------|
| `GPU_MEMORY_UTILIZATION` | `0.50` | vLLM KV cache size. Try 0.65 for more batching. |
| `UVICORN_WORKERS` | `4` | Gateway worker processes |
| `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS` | `16` | Gateway semaphore — max in-flight Triton calls |
| `HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS` | `64` | Same for /restructure-pages |
| `HPS_INFERENCE_TIMEOUT` | `600` | Seconds before gateway returns 504 |
| `MODEL_NAME` | `PaddleOCR-VL-1.5-0.9B` | vLLM model to load |
| `GATEWAY_PORT` | `8080` | Exposed port |
| `VLLM_PORT` | `8118` | Internal vLLM port |
| `INIT_TIMEOUT` | `600` | Max seconds to wait for each service to become healthy |

### Triton (requires file edit or rebuild)

| Setting | File | Default | Effect |
|---------|------|---------|--------|
| `max_batch_size` | `config.pbtxt` | 16 | Max pages per Triton inference call |
| `count` (instance_group) | `config.pbtxt` | 2 | Parallel Triton model instances |
| `max_concurrency` | `pipeline_config.yaml` | 32 | Max concurrent Triton → vLLM calls |

---

## Production Connectivity

**Never** expose the pod via the HTTP proxy URL (`*.proxy.runpod.net`) for backend-to-backend calls.
Use TCP port exposure for direct connections with no timeout limit.

### RunPod port types
| Type | URL | Proxy | Timeout |
|------|-----|-------|---------|
| HTTP | `https://<id>-8080.proxy.runpod.net` | Cloudflare | ~140s hard limit |
| TCP | `http://<ip>:<port>` | None (direct) | No limit |

### Elixir + Oban architecture
Users never connect to the pod directly. The Elixir backend manages all connections:

```
10,000 users → Elixir Phoenix API → Oban queue (Postgres rows)
                                          ↓ 16 jobs at a time
                                     Pod TCP port  (max 16 connections ever)
```

Oban queue concurrency = `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`. The pod never
sees more simultaneous connections than this number, regardless of user count.

```elixir
config :myapp, Oban,
  queues: [
    ocr_pod_1: 16,   # maps to HPS_MAX_CONCURRENT_INFERENCE_REQUESTS
    ocr_pod_2: 16,
  ]
```

---

## Quick Reference: Pod Health Checks

```bash
wget -qO- http://localhost:8080/health        # gateway liveness
wget -qO- http://localhost:8080/health/ready  # full stack ready (Triton + vLLM)
wget -qO- http://localhost:8000/v2/health/ready  # Triton direct
wget -qO- http://localhost:8118/health           # vLLM direct
nvidia-smi                                       # GPU state
```

## Quick Reference: Rebuild and Push

```bash
# From WSL:
cd /mnt/d/paddleocr/PaddleOCR-main/runpod/hps-final
bash build.sh                     # → ringkasannet/paddleocr-hps:paddlex3.4-gpu-ready
bash build.sh v2                  # → ringkasannet/paddleocr-hps:v2
```

## Quick Reference: Start Services on Pod

```bash
bash /opt/start_hps.sh
# or if using network volume copy:
bash /workspace/paddleocr-server/runpod/hps-final/start.sh
```
