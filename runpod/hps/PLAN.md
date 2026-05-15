# HPS on RunPod — Plan

## Architecture

```
Client → FastAPI Gateway (port 8080)
              ↓ gRPC (paddlex_hps_client)
          Triton Server (port 8000/8001)    ← layout-parsing + restructure-pages
              ↓ HTTP OpenAI API
          vLLM Server (port 8118)           ← PaddleOCR-VL-1.5-0.9B
```

---

## What Each Component Needs

| Component | Binary/Runtime | Key Packages |
|-----------|---------------|--------------|
| vLLM server | `paddleocr genai_server` | paddlepaddle-gpu + paddleocr + vllm |
| Triton server | `tritonserver` binary | paddlex + paddlex_hps_server (Python backends) |
| Gateway | `uvicorn` | fastapi + paddlex[serving] + **paddlex_hps_client wheel** (from SDK) |

**Critical**: `paddlex_hps_client` is a private wheel inside the HPS SDK
(`paddlex_hps_PaddleOCR-VL-1.5_sdk/client/`). Not on PyPI. Gateway won't work without it.

---

## Open Question: Does paddlex/hps have vllm?

The `paddlex/hps` image has Triton + PaddleX but may NOT have vllm/genai_server.
The original docker-compose uses a SEPARATE image for vLLM:
`paddleocr-genai-vllm-server:latest-nvidia-gpu`

**Must verify before proceeding:**
```bash
docker run --rm ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.4-gpu \
  /paddlex/py310/bin/paddleocr --help 2>&1 | grep -i genai
```

- **If genai_server exists** → paddlex/hps image covers everything, proceed with current plan
- **If genai_server missing** → need vllm installed separately (add to install_volume.sh)

---

## Approach: runpod/pytorch + network volume (pytriton)

Using `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` as base (CUDA 12.8.1).
Everything installed into `/workspace` (network volume, persists across pod restarts).

**Why not paddlex/hps as base:**
- CUDA 11.8 (old, may have compatibility issues)
- Large image (~20 GB, slow cold starts)
- May not have vllm

**Why not a custom Docker image:**
- Slow to build (30-40 min)
- Slow to push (10+ GB)
- Hard to update

**Network volume approach:**
- Install once, reuse forever
- Easy to update individual components
- Proven: Triton 2.51.0 + pytriton already works on this pod

---

## Network Volume Layout

```
/workspace/
├── .triton/                    ← nvidia-pytriton venv (Triton 2.51.0)
├── .venv_paddleocr/            ← paddlepaddle-gpu + paddleocr + paddlex (already installed)
├── .venv_vlm/                  ← vllm venv (to be set up)
├── .venv_gateway/              ← fastapi + paddlex[serving] + paddlex_hps_client
├── hps/
│   ├── server/                 ← HPS SDK server files (model_repo, pipeline_config.yaml)
│   │   └── model_repo/
│   │       ├── layout-parsing/
│   │       └── restructure-pages/
│   └── client/                 ← HPS SDK client files (paddlex_hps_client whl)
├── gateway/
│   └── app.py
├── models/
│   └── hf_cache/               ← PaddleOCR-VL-1.5-0.9B weights
├── install_hps.sh              ← run once
└── start_hps.sh                ← run every pod start
```

---

## Step-by-Step with Verification Gates

### Gate 0 — Verify what's already on the pod

```bash
# Check existing venvs
ls /workspace/.triton/     2>/dev/null && echo "triton venv: OK" || echo "triton venv: MISSING"
ls /workspace/.venv_paddleocr/ 2>/dev/null && echo "paddle venv: OK" || echo "paddle venv: MISSING"

# Check Triton stub is in place
ls /workspace/.triton/lib/python3.12/site-packages/pytriton/tritonserver/backends/python/triton_python_backend_stub \
  2>/dev/null && echo "stub: OK" || echo "stub: MISSING"

# Check paddlex_hps_server
/workspace/.venv_paddleocr/bin/python3 -c \
  "import paddlex_hps_server; print('paddlex_hps_server: OK')" 2>/dev/null || echo "paddlex_hps_server: MISSING"

# Check paddleocr genai_server
/workspace/.venv_paddleocr/bin/paddleocr --help 2>&1 | grep -i genai \
  && echo "genai_server: OK" || echo "genai_server: MISSING"
```

### Gate 1 — HPS SDK download

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz \
  -O /tmp/sdk.tar.gz
tar -xf /tmp/sdk.tar.gz -C /tmp/
mkdir -p /workspace/hps/server /workspace/hps/client
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/server/. /workspace/hps/server/
cp -r /tmp/paddlex_hps_PaddleOCR-VL-1.5_sdk/client/. /workspace/hps/client/
rm /tmp/sdk.tar.gz

# Verify
ls /workspace/hps/server/model_repo/ && echo "model_repo: OK"
ls /workspace/hps/client/*.whl 2>/dev/null && echo "client whl: OK" || echo "client whl: MISSING"

# Patch pipeline config
sed -i 's|http://paddleocr-vlm-server:8080/v1|http://localhost:8118/v1|g' \
  /workspace/hps/server/pipeline_config.yaml
grep server_url /workspace/hps/server/pipeline_config.yaml
```

### Gate 2 — Gateway venv

```bash
/workspace/.venv_paddleocr/bin/python3 -m venv /workspace/.venv_gateway
/workspace/.venv_gateway/bin/pip install --upgrade pip -q

# Core gateway requirements
/workspace/.venv_gateway/bin/pip install --no-cache-dir \
  fastapi==0.123.6 uvicorn==0.35.0 "paddlex[serving]>=3.4.0"

# SDK client requirements (needed by paddlex_hps_client)
/workspace/.venv_gateway/bin/pip install --no-cache-dir \
  -r /workspace/hps/client/requirements.txt

# Private paddlex_hps_client wheel
WHL=$(ls /workspace/hps/client/paddlex_hps_client-*.whl | head -1)
/workspace/.venv_gateway/bin/pip install --no-cache-dir "$WHL"

# Verify
/workspace/.venv_gateway/bin/python3 -c \
  "from paddlex_hps_client import triton_request_async; print('gateway: OK')"
```

### Gate 3 — Gateway app

```bash
mkdir -p /workspace/gateway
wget -q https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/paddleocr_vl_docker/hps/gateway/app.py \
  -O /workspace/gateway/app.py

# Verify
python3 -c "import ast; ast.parse(open('/workspace/gateway/app.py').read()); print('app.py: OK')"
```

### Gate 4 — Triton Python backend test (CPU, no GPU needed)

```bash
# Ensure stub is in place
STUB_DIR=/workspace/.triton/lib/python3.12/site-packages/pytriton/tritonserver/backends/python
STUB_SRC=/workspace/.triton/lib/python3.12/site-packages/pytriton/tritonserver/python_backend_stubs/3.12/triton_python_backend_stub
mkdir -p $STUB_DIR
cp $STUB_SRC $STUB_DIR/triton_python_backend_stub
chmod +x $STUB_DIR/triton_python_backend_stub

# Create test model
mkdir -p /tmp/model_repo/hello_world/1
cat > /tmp/model_repo/hello_world/config.pbtxt << 'EOF'
name: "hello_world"
backend: "python"
max_batch_size: 0
input  [{ name: "INPUT0"  data_type: TYPE_STRING dims: [1] }]
output [{ name: "OUTPUT0" data_type: TYPE_STRING dims: [1] }]
instance_group [{ kind: KIND_CPU }]
EOF
cat > /tmp/model_repo/hello_world/1/model.py << 'EOF'
import triton_python_backend_utils as pb_utils
import numpy as np
class TritonPythonModel:
    def initialize(self, args): print("[test] backend OK")
    def execute(self, requests):
        return [pb_utils.InferenceResponse([
            pb_utils.Tensor("OUTPUT0", np.array(["ok"], dtype=object))
        ]) for _ in requests]
EOF

TRITON_BASE=/workspace/.triton/lib/python3.12/site-packages/pytriton/tritonserver
LD_LIBRARY_PATH=$TRITON_BASE/lib:$LD_LIBRARY_PATH \
  $TRITON_BASE/bin/tritonserver \
  --model-repository=/tmp/model_repo \
  --backend-directory=$TRITON_BASE/backends \
  --backend-config=python,python-runtime=/workspace/.venv_paddleocr/bin/python3 \
  --allow-metrics=false --log-info=true 2>&1 | head -30

# Expected: "successfully loaded 'hello_world'"
```

### Gate 5 — vLLM server test

```bash
# Test if genai_server is available in paddle venv
/workspace/.venv_paddleocr/bin/paddleocr genai_server --help 2>&1 | head -5

# If available, start it (takes ~2 min to load model)
/workspace/.venv_paddleocr/bin/paddleocr genai_server \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --backend vllm \
  --host 0.0.0.0 \
  --port 8118 &

# Wait and check
sleep 120
curl -s http://localhost:8118/health && echo "vLLM: OK"
```

### Gate 6 — HPS Triton models test

Only run after Gates 4 and 5 pass.

```bash
TRITON_BASE=/workspace/.triton/lib/python3.12/site-packages/pytriton/tritonserver
LD_LIBRARY_PATH=$TRITON_BASE/lib:$LD_LIBRARY_PATH \
  $TRITON_BASE/bin/tritonserver \
  --model-repository=/workspace/hps/server/model_repo \
  --backend-directory=$TRITON_BASE/backends \
  --backend-config=python,python-runtime=/workspace/.venv_paddleocr/bin/python3 \
  --backend-config=python,shm-default-byte-size=67108864 \
  --http-port=8000 --grpc-port=8001 \
  --model-control-mode=explicit \
  --load-model=layout-parsing \
  --load-model=restructure-pages \
  --allow-metrics=false --log-info=true 2>&1 | tail -30

# Expected: layout-parsing READY, restructure-pages READY
```

### Gate 7 — End-to-end

```bash
curl -s http://localhost:8000/v2/health/ready && echo "Triton: ready"
curl -s http://localhost:8080/health/ready && echo "Gateway: ready"
```

---

## Decision Points

| Gate | Pass → | Fail → |
|------|--------|--------|
| Gate 0 | Proceed to Gate 1 | Fix missing items first |
| Gate 1 | SDK downloaded | Check Baidu URL is reachable |
| Gate 2 | Gateway ready | Check paddlex_hps_client whl exists |
| Gate 3 | app.py present | Download manually or from repo |
| Gate 4 | Triton Python backend works | Debug stub or pytriton version |
| Gate 5 | vLLM available | Install vllm separately in .venv_paddleocr |
| Gate 6 | HPS models load | Check paddlex_hps_server import in paddle venv |
| Gate 7 | Full stack working | Write start_hps.sh |
