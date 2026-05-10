# PaddleOCR on RunPod — Complete Deployment Guide

Everything you need to know: what was built, how RunPod serverless works,
how to deploy, how to reduce cold starts, and how to call the API.

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [How the Docker Image Works](#2-how-the-docker-image-works)
3. [Two Ways to Run: Pod vs Serverless](#3-two-ways-to-run-pod-vs-serverless)
4. [How RunPod Serverless Works](#4-how-runpod-serverless-works)
5. [Cold Starts — The Core Problem](#5-cold-starts--the-core-problem)
6. [RunPod Model Caching](#6-runpod-model-caching)
7. [vLLM on RunPod](#7-vllm-on-runpod)
8. [Step-by-Step: Deploy to RunPod Serverless](#8-step-by-step-deploy-to-runpod-serverless)
9. [Calling the API](#9-calling-the-api)
10. [Environment Variables Reference](#10-environment-variables-reference)
11. [Cost & Scaling Guide](#11-cost--scaling-guide)
12. [Local Testing](#12-local-testing)
13. [Troubleshooting](#13-troubleshooting)
14. [Files in This Folder](#14-files-in-this-folder)

---

## 1. What We Built

A single Docker image that combines two services that PaddleOCR requires:

```
┌─────────────────────────────────────────────────────┐
│                Docker Container                      │
│                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   vLLM genai server │  │   PaddleX pipeline  │  │
│  │   port 8118         │──│   port 8080         │  │
│  │                     │  │                     │  │
│  │  PaddleOCR-VL-1.5  │  │  Layout detection   │  │
│  │  model loaded       │  │  + VLM recognition  │  │
│  │  in GPU VRAM        │  │  HTTP API           │  │
│  └─────────────────────┘  └─────────────────────┘  │
│                                    │                │
└────────────────────────────────────┼────────────────┘
                                     │ :8080 (exposed)
                                 Your app calls this
```

**Why two services?** PaddleOCR-VL uses a Vision-Language Model (VLM) for
understanding document content. That VLM runs best under vLLM (a high-performance
inference server). PaddleX handles everything else — layout detection, table
parsing, result formatting — and delegates the VLM part to vLLM.

**Why one container?** RunPod runs a single container per worker. The official
PaddleOCR Docker Compose setup uses two containers. We combine them into one
using a startup script that launches both processes.

---

## 2. How the Docker Image Works

### Base Image

```
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu
```

This is PaddleOCR's official image. It already has:
- CUDA 12.x runtime
- Python 3.10
- vLLM 0.10.2 installed
- `paddleocr` CLI in `/usr/local/bin/paddleocr`

We only add on top of it:
- System libraries needed by PaddlePaddle/OpenCV (`libgl1` etc.)
- A separate Python virtual environment at `/workspace/.paddleocr/` containing:
  - PaddlePaddle GPU 3.2.1
  - paddleocr[doc-parser]
  - paddlex with serving plugin
- The patched `PaddleOCR-VL.yaml` pipeline config
- `handler.py` (serverless) and `start.sh` (pod/local)

### Why a Separate Virtual Environment?

PaddlePaddle and vLLM have **conflicting Python dependencies** (different
versions of transformers, torch, etc.). The official PaddleOCR docs explicitly
say to deploy them in separate environments. We solve this with:

- **System Python** (`/usr/local/`) → vLLM environment (from base image)
- **`/workspace/.paddleocr/`** → PaddlePaddle environment (we install this)

### Layer Cache

Every `RUN` instruction in the Dockerfile creates a cached layer. Layers only
re-upload/re-download when they change. Our ordering is deliberate:

```dockerfile
[1]  FROM base image          ← 13 GB, pulled once, never re-sent to DockerHub
[2]  USER root
[3]  apt-get install          ← rarely changes
[4]  python3 -m venv          ← rarely changes
[5]  pip paddlepaddle-gpu     ← changes if you upgrade PaddlePaddle
[6]  pip paddleocr            ← changes if you upgrade paddleocr
[7]  paddlex --install        ← changes if you upgrade paddlex
[8]  paddlex --get_config     ← changes if pipeline changes
[9]  COPY patch_config.py     ← changes if you edit patch_config.py
[10] pip install runpod       ← rarely changes
[11] COPY start.sh            ← changes when you edit start.sh (~KB)
[12] COPY handler.py          ← changes when you edit handler.py (~KB)
```

Result: editing `handler.py` only re-pushes layers 12+ — a few kilobytes.

---

## 3. Two Ways to Run: Pod vs Serverless

The same Docker image supports both modes.

### Pod Mode (Persistent GPU)

```bash
docker run --gpus all -p 8080:8080 -p 8118:8118 \
  --entrypoint bash \
  ringkasannet/paddleocr-runpod:latest \
  /workspace/start.sh
```

- GPU is reserved 24/7 whether you use it or not
- Both services stay running as HTTP servers
- You call `http://YOUR_POD_IP:8080/` directly
- Billed per hour
- Good for: high sustained traffic, development, debugging

### Serverless Mode (Default CMD)

```bash
# Default CMD in Dockerfile is:
CMD ["python", "-u", "/workspace/handler.py"]
```

- GPU only allocated when a job arrives
- `handler.py` starts both services at worker init, then processes jobs
- You call via RunPod's API proxy
- Billed per second of actual use
- Good for: variable traffic, scale-to-zero overnight, cost savings

---

## 4. How RunPod Serverless Works

### The Job Queue Model

```
Your app                RunPod                    Worker
   │                      │                          │
   ├─POST /run────────────▶│                          │
   │  {"input": {...}}    │──── dispatch job ────────▶│
   │                      │                          │ handler(job) runs
   ├─GET /status/job_id──▶│                          │
   │◀─ {status:RUNNING}───┤                          │
   │                      │◀─── return result ────────┤
   ├─GET /status/job_id──▶│                          │
   │◀─ {status:COMPLETED}─┤                          │
   │  {output: {...}}     │                          │
```

Every request goes through RunPod's queue. You never call your container
directly — RunPod manages routing, retries, and scaling.

### Worker Lifecycle

```
Cold start → Initializing → Ready → Running (job) → Idle → Stopped
                                         ↑               │
                                         └───────────────┘
                                         (next job arrives during idle timeout)
```

| State | Billed? | Description |
|---|---|---|
| Initializing | Yes | Pulling image, starting container, loading model |
| Running | Yes | Your handler is executing |
| Idle | Yes | Between jobs, waiting for next (idle timeout window) |
| Stopped | No | Scaled to zero |

### The Handler Function

`handler.py` contains this pattern:

```python
import runpod

# This runs ONCE when the worker starts (not per job)
initialize()          # starts vLLM + PaddleX, waits for health

# This runs PER JOB
def handler(job):
    image = job["input"]["image"]
    result = call_paddlex_api(image)
    return result

runpod.serverless.start({"handler": handler})
```

`runpod.serverless.start()` is a blocking call that connects your handler to
RunPod's job polling loop. It never returns — the process stays alive waiting
for jobs.

### Job Input/Output

Input (what you send):
```json
{
  "input": {
    "image": "<base64 string or public URL>",
    "use_layout_detection": true,
    "use_doc_preprocessor": false
  }
}
```

Output (what you get back):
```json
{
  "id": "job-abc123",
  "status": "COMPLETED",
  "output": { "...PaddleX OCR results..." },
  "executionTime": 4521
}
```

---

## 5. Cold Starts — The Core Problem

A cold start happens when a job arrives and there is no ready worker. RunPod
must start one from scratch.

### What Happens During Our Cold Start

```
1. Image layers pulled from DockerHub     → 2–30s (FlashBoot caches after first run)
2. Container starts, Python imports       → 5–10s
3. vLLM downloads PaddleOCR-VL model    → 2–5 min (only if not cached!)
4. vLLM loads model weights into GPU     → 60–120s
5. vLLM compiles CUDA kernels            → 30–60s (first time only)
6. PaddleX downloads layout models       → 30–60s (only if not cached)
7. PaddleX starts HTTP server            → 5–10s
```

**Total without caching: 5–10 minutes**
**Total with caching: 60–120 seconds**

### Solutions (Best to Worst)

#### ✅ Solution 1: Active Workers (Eliminates Cold Start)

Set **Active Workers = 1** in the endpoint settings. One worker runs 24/7,
always ready. The cold start never happens.

Cost: ~$0.40/hr for A4000. Worth it for any production service.

#### ✅ Solution 2: RunPod Model Caching (Best for cost)

Set the **Model** field in the endpoint to `PaddleOCR-VL-1.5-0.9B`. RunPod
will pre-cache the model on host machines. When a new worker starts, it loads
from local NVMe instead of downloading. Reduces cold start to ~60–90s. Free.

See section 6 for details.

#### ✅ Solution 3: FlashBoot (Default — automatic)

FlashBoot is RunPod's proprietary system that saves worker container state
between spin-downs. After the first cold start, subsequent starts skip most
of steps 1–3 above. Claims P90 < 2s, real-world for our model ~30–60s.
Enabled by default, no action needed.

#### ✅ Solution 4: Increase Idle Timeout

Default idle timeout is 5 seconds — a worker shuts down 5 seconds after
finishing a job. Set it to 30–60 seconds so burst traffic within that window
hits a warm worker instead of triggering a cold start.

---

## 6. RunPod Model Caching

This is the most impactful free optimization.

### How It Works

When you specify a model in the endpoint's **Model** field:
1. RunPod pre-downloads the model to host machines in your selected region
2. When a worker starts, it mounts the host cache — model is already on local NVMe
3. No internet download needed at worker init

### Where Models Are Stored

```
/runpod-volume/huggingface-cache/hub/
└── models--PaddlePaddle--PaddleOCR-VL-1.5/
    ├── refs/
    │   └── main           ← commit hash
    └── snapshots/
        └── abc123.../     ← actual model files
```

Note: HuggingFace converts `/` in model names to `--`. So `PaddlePaddle/PaddleOCR-VL-1.5`
becomes `models--PaddlePaddle--PaddleOCR-VL-1.5`.

### Required Configuration

In `handler.py` and `Dockerfile`:
```python
os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"
```

This tells vLLM where to find the cached model. If the model isn't there,
vLLM falls back to downloading it (cold start, but it still works).

### Setup Steps

1. In RunPod console → Serverless → Your Endpoint → Edit
2. Find the **Model** field
3. Enter: `PaddleOCR-VL-1.5-0.9B`
4. For gated models: also enter your HuggingFace token in **HF Token** field
5. Save — RunPod starts pre-caching in your region

### Limitation

One cached model per endpoint. You can't cache both the VLM model and the
PaddlePaddle layout models (PP-DocLayoutV2, etc.) this way. The layout models
will still download on first cold start (~200 MB, much smaller).

---

## 7. vLLM on RunPod

### What vLLM Does

vLLM is a high-performance inference server optimized for LLMs. In our setup
it serves the Vision-Language Model part of PaddleOCR — the component that
actually "reads" and interprets document content.

### Why vLLM Instead of Plain PyTorch

- **PagedAttention**: manages GPU memory efficiently for variable-length inputs
- **Continuous batching**: processes multiple requests simultaneously
- **CUDA kernel fusion**: compiled inference is 3–5× faster than naive PyTorch
- **OpenAI-compatible API**: the `/v1/` endpoint format PaddleX expects

### RunPod's Official vLLM Worker

RunPod maintains `runpod/worker-v1-vllm` — a pre-built serverless worker for
pure vLLM deployments. We don't use it because we need PaddleX serving on top.
But it's useful to know it exists for LLM-only use cases.

Key env vars from that worker (our `handler.py` respects the same ones):

| Env Var | Default | Description |
|---|---|---|
| `MODEL_NAME` | `PaddleOCR-VL-1.5-0.9B` | Which model to load |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU VRAM for vLLM |
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `MAX_MODEL_LEN` | auto | Max context length in tokens |

### GPU Memory: What `gpu_memory_utilization` Means

```
Total GPU VRAM × gpu_memory_utilization = memory budget for vLLM

16 GB × 0.90 = 14.4 GB budget
  - Model weights:  ~2 GB
  - KV cache:      ~12 GB  ← more = faster, handles longer docs
```

Too low → `ValueError: No available memory for the cache blocks`
Too high → may interfere with other processes (OS, PaddleX)

**Recommended by GPU size:**

| GPU VRAM | Recommended `GPU_MEMORY_UTILIZATION` |
|---|---|
| 8 GB  | 0.90 (tight, may still fail) |
| 16 GB | 0.85 |
| 24 GB | 0.75 |
| 40 GB | 0.65 (default) |
| 80 GB | 0.50 |

### The `--backend_config` Pattern

`gpu_memory_utilization` and other vLLM engine args cannot be passed as direct
CLI flags to `paddleocr genai_server`. They must go through a YAML config file:

```yaml
# /tmp/vllm_backend.yaml
gpu-memory-utilization: 0.90
max-num-seqs: 128
```

```bash
paddleocr genai_server \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --backend vllm \
  --backend_config /tmp/vllm_backend.yaml
```

This is handled automatically in `handler.py` and `start.sh`.

---

## 8. Step-by-Step: Deploy to RunPod Serverless

### Prerequisites

- Docker Hub account (you have `ringkasannet`)
- RunPod account with API key
- Image pushed: `ringkasannet/paddleocr-runpod:latest`

### Step 1: Create a Network Volume (Optional but Recommended)

Network volumes persist model weights between worker restarts.

1. RunPod console → **Storage** → **New Network Volume**
2. Region: select where you want your workers (e.g., US-TX)
3. Size: 20 GB (model ~2 GB + layout models ~500 MB + room for growth)
4. Name: `paddleocr-models`
5. Create

### Step 2: Create the Serverless Endpoint

1. RunPod console → **Serverless** → **New Endpoint**
2. Fill in:

| Field | Value |
|---|---|
| Name | `paddleocr-vl` |
| Container Image | `ringkasannet/paddleocr-runpod:latest` |
| Container Registry Credentials | (if private repo, add DockerHub login) |
| **Model** | `PaddleOCR-VL-1.5-0.9B` |
| GPU Type | RTX 4000 Ada / A4000 (16 GB minimum) |
| Active Workers | `1` (recommended for production) |
| Max Workers | `3` |
| Idle Timeout | `30` |
| Execution Timeout | `600` |
| Network Volume | `paddleocr-models` → mount at `/runpod-volume` |

### Step 3: Set Environment Variables

In the endpoint's Advanced settings → Environment Variables:

```
MODEL_NAME               = PaddleOCR-VL-1.5-0.9B
GPU_MEMORY_UTILIZATION   = 0.85
HF_HOME                  = /runpod-volume/huggingface-cache
RUNPOD_INIT_TIMEOUT      = 600
PYTHONUNBUFFERED         = 1
```

### Step 4: Deploy

Click **Deploy**. RunPod will:
1. Pull your image (first time: slow, subsequent: FlashBoot cache)
2. Start the active worker
3. Worker runs `handler.py` → starts vLLM → starts PaddleX
4. Worker shows **Ready** in the dashboard

First deployment takes 5–10 minutes. After that, the active worker stays warm.

### Step 5: Get Your Endpoint ID

From the endpoint page, copy the endpoint ID (format: `abc1234defg`).
You'll need this for API calls.

---

## 9. Calling the API

### Your Endpoint URL

```
https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
```

### Authentication

All requests need:
```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

Get your API key: RunPod console → Settings → API Keys.

### Async vs Sync Calls

| Method | Endpoint | Behavior |
|---|---|---|
| `POST /run` | Async | Returns job ID immediately, poll for result |
| `POST /runsync` | Sync | Waits until job completes, returns result directly |

For documents that take < 30s: use `/runsync` (simpler).
For large documents or batch: use `/run` + polling.

### Example: Python with `runpod` SDK

```python
import runpod
import base64

runpod.api_key = "YOUR_API_KEY"

# Load image as base64
with open("document.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Call the endpoint
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

result = endpoint.run_sync(
    {"image": img_b64},
    timeout=300
)

print(result)
```

### Example: Raw HTTP (curl)

```bash
# Sync call with a public image URL
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/document.png"
    }
  }'
```

### Example: Async call + polling

```bash
# Submit job
JOB=$(curl -s -X POST https://api.runpod.ai/v2/ENDPOINT_ID/run \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"image": "https://example.com/doc.png"}}')

JOB_ID=$(echo $JOB | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Job ID: $JOB_ID"

# Poll until complete
while true; do
  STATUS=$(curl -s https://api.runpod.ai/v2/ENDPOINT_ID/status/$JOB_ID \
    -H "Authorization: Bearer $KEY")
  STATE=$(echo $STATUS | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATE"
  [ "$STATE" = "COMPLETED" ] || [ "$STATE" = "FAILED" ] && break
  sleep 5
done
echo $STATUS | python3 -m json.tool
```

### Endpoint Health Check

```bash
curl https://api.runpod.ai/v2/ENDPOINT_ID/health \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

```json
{
  "workers": {"idle": 1, "running": 0, "ready": 1},
  "jobs": {"inQueue": 0, "inProgress": 0}
}
```

---

## 10. Environment Variables Reference

These can be set at runtime (`docker run -e ...`) or in RunPod endpoint settings.

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `PaddleOCR-VL-1.5-0.9B` | vLLM model to load |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU VRAM for vLLM |
| `VLLM_PORT` | `8118` | Internal vLLM server port |
| `PADDLE_PORT` | `8080` | Internal PaddleX API port |
| `PIPELINE_CONFIG` | `/workspace/PaddleOCR-VL.yaml` | PaddleX pipeline YAML path |
| `HF_HOME` | `/workspace/models/hf_cache` | HuggingFace model cache directory |
| `RUNPOD_INIT_TIMEOUT` | `600` | Seconds before RunPod marks worker unhealthy during init |
| `PYTHONUNBUFFERED` | `1` | Show Python logs in real-time |

---

## 11. Cost & Scaling Guide

### Serverless vs Pod Break-Even

Serverless active worker (always-on) vs Dedicated Pod:

| GPU | Active Worker (/hr) | Dedicated Pod (/hr) | Cheaper option |
|---|---|---|---|
| A4000 16GB | ~$0.40 | ~$0.34 | Pod if >85% utilized |
| RTX 3090 24GB | ~$0.47 | ~$0.44 | Pod if >94% utilized |
| A100 80GB | ~$2.16 | ~$1.89 | Pod if >88% utilized |

**Rule of thumb**: If your service runs more than ~20 hours/day at capacity,
a Dedicated Pod is cheaper. For anything less, Serverless saves money.

### Flex Workers (Scale-to-Zero)

With Active Workers = 0, you pay nothing when idle:

```
Cost per OCR request (A4000, flex):
  Cold start: 90s × $0.00011 = $0.0099
  Inference:  10s × $0.00011 = $0.0011
  Total: ~$0.011 per document
```

### Recommended Configuration by Traffic

| Traffic | Active Workers | Max Workers | Idle Timeout | Est. Monthly Cost |
|---|---|---|---|---|
| Testing / dev | 0 | 1 | 5s | Near $0 |
| Light (< 100 docs/day) | 0 | 2 | 30s | < $5 |
| Medium (< 1000 docs/day) | 1 | 3 | 30s | ~$300 (A4000) |
| Heavy (> 1000 docs/day) | 2 | 5 | 60s | ~$600+ |

---

## 12. Local Testing

### Full test (requires NVIDIA GPU with 8+ GB VRAM)

```bash
docker run --gpus all \
  --rm \
  -p 8080:8080 \
  -p 8118:8118 \
  --shm-size=8g \
  -v paddleocr-cache:/workspace/models \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  ringkasannet/paddleocr-runpod:latest
```

### Test the HTTP API (once both services are up)

```bash
curl -X POST http://localhost:8080/ocr-doc-parser \
  -H "Content-Type: application/json" \
  -d '{"image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"}'
```

### Simulate RunPod serverless locally (no GPU needed)

```bash
# RunPod SDK can serve the handler locally for input/output testing
pip install runpod

python handler.py --rp_serve_api
# Serves at http://localhost:8000/runsync

curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"image": "..."}}'
```

Note: `--rp_serve_api` tests the handler function plumbing only.
vLLM won't start without a GPU.

### Pod mode from same image

```bash
docker run --gpus all \
  --rm \
  -p 8080:8080 \
  -p 8118:8118 \
  --shm-size=8g \
  -v paddleocr-cache:/workspace/models \
  --entrypoint bash \
  ringkasannet/paddleocr-runpod:latest \
  /workspace/start.sh
```

---

## 13. Troubleshooting

### `ValueError: No available memory for the cache blocks`

vLLM doesn't have enough VRAM for the KV cache after loading the model.

**Fix**: Increase `GPU_MEMORY_UTILIZATION` (try 0.90, then 0.95).
Or use a GPU with more VRAM. Minimum recommended: 16 GB.

### `FileNotFoundError: paddlex not in PATH`

Some paddleocr/paddlex CLI commands spawn subprocesses and don't inherit the
venv PATH. 

**Fix**: Always call them as:
```bash
PATH="/workspace/.paddleocr/bin:$PATH" paddlex --install serving
```

### `EOFError: EOF when reading a line` during build

`paddlex --get_pipeline_config PaddleOCR-VL` prompts interactively.
Docker build has no stdin.

**Fix**: Always pass `--save_path`:
```bash
paddlex --get_pipeline_config PaddleOCR-VL --save_path /workspace
```

### `E: List directory /var/lib/apt/lists/partial is missing`

The base image runs as a non-root user. `apt-get` requires root.

**Fix**: Add `USER root` before any `apt-get` in the Dockerfile.

### `Permission denied` on apt-get in Docker build

Same as above. The compose.yaml for this project has `user: root` for the
same reason.

### vLLM shows logs but `[ERROR] vLLM exited unexpectedly`

The process started but died before becoming healthy. Check the vLLM logs
for the actual error (usually OOM or missing model).

### No logs showing during container run

Python buffers output by default.

**Fix**: Set `PYTHONUNBUFFERED=1` and use `sed -u` in shell pipes.

### Cold start takes > 10 minutes on RunPod

The model is downloading every time. Enable RunPod model caching:
1. Edit endpoint → set **Model** field to `PaddleOCR-VL-1.5-0.9B`
2. Attach a network volume with `HF_HOME` pointing to it

### `RUNPOD_INIT_TIMEOUT` exceeded

RunPod marked the worker unhealthy because initialization took too long.

**Fix**: Set `RUNPOD_INIT_TIMEOUT=600` (or higher) in endpoint env vars.

---

## 14. Files in This Folder

| File | Purpose | When to edit |
|---|---|---|
| `Dockerfile` | Builds the Docker image | Adding dependencies, changing base image |
| `start.sh` | Pod / local mode startup | Changing service startup args |
| `handler.py` | Serverless mode — RunPod job handler | Changing API input/output format |
| `patch_config.py` | Patches PaddleOCR-VL.yaml at build time | If pipeline config fields change |
| `build.sh` | Helper script: build + push to DockerHub | One-time setup convenience |
| `PaddleOCR-VL.yaml` | Pipeline config (generated at build time) | Don't edit directly — re-run patch_config.py |
| `.dockerignore` | Excludes files from Docker build context | Adding/removing build exclusions |

### Build & Deploy Commands

```bash
# Build
docker build --platform linux/amd64 -t ringkasannet/paddleocr-runpod:latest .

# Push (after docker login)
docker tag paddleocr-runpod:latest ringkasannet/paddleocr-runpod:latest
docker push ringkasannet/paddleocr-runpod:latest

# Inspect what's in the image
docker run --rm ringkasannet/paddleocr-runpod:latest cat /workspace/PaddleOCR-VL.yaml
docker run --rm ringkasannet/paddleocr-runpod:latest python3 --version
docker run --rm ringkasannet/paddleocr-runpod:latest which paddleocr
```
