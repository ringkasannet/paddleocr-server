#!/usr/bin/env bash
# ============================================================
# Option C — Official NVIDIA Triton Portability Test
#
# 1. Pulls nvcr.io/nvidia/tritonserver:24.09-py3-min (CUDA 12.6, ~6 GB)
# 2. Verifies Triton version
# 3. Extracts /opt/tritonserver → /tmp/triton_official/
# 4. Runs it inside the RunPod PyTorch container
# 5. Runs a minimal Python backend model to verify Python backend works
#
# Usage (WSL):
#   bash test_triton_official.sh
# ============================================================
set -euo pipefail

TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.09-py3-min"
RUNPOD_IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
TEST_DIR="/tmp/triton_official"

# ── Step 1: Pull Triton image ─────────────────────────────────
echo "============================================"
echo " Step 1: Pulling official NVIDIA Triton"
echo " (24.09-py3-min = CUDA 12.6, ~6 GB)"
echo "============================================"
docker pull "$TRITON_IMAGE"

# ── Step 2: Check version inside Triton image ─────────────────
echo ""
echo "============================================"
echo " Step 2: Triton version + CUDA version"
echo "============================================"
docker run --rm "$TRITON_IMAGE" \
    bash -c "tritonserver --version 2>&1 | head -5; echo '---'; nvcc --version 2>/dev/null | grep release || cat /usr/local/cuda/version.txt 2>/dev/null || echo 'CUDA info not found'"

# ── Step 3: Extract tritonserver ──────────────────────────────
echo ""
echo "============================================"
echo " Step 3: Extracting /opt/tritonserver"
echo "============================================"
rm -rf "$TEST_DIR" && mkdir -p "$TEST_DIR"
docker create --name triton_official_tmp "$TRITON_IMAGE" bash
docker cp triton_official_tmp:/opt/tritonserver "$TEST_DIR/tritonserver"
docker rm triton_official_tmp
echo "Extracted. Size: $(du -sh $TEST_DIR/tritonserver | cut -f1)"

# ── Step 4: Create a minimal Python backend test model ────────
# This verifies the Python backend works WITHOUT needing PaddleX or GPU
echo ""
echo "============================================"
echo " Step 4: Creating minimal Python backend test model"
echo "============================================"
mkdir -p "$TEST_DIR/model_repo/hello_world/1"

cat > "$TEST_DIR/model_repo/hello_world/config.pbtxt" << 'EOF'
name: "hello_world"
backend: "python"
max_batch_size: 0
input [{ name: "INPUT0" data_type: TYPE_STRING dims: [1] }]
output [{ name: "OUTPUT0" data_type: TYPE_STRING dims: [1] }]
instance_group [{ kind: KIND_CPU }]
EOF

cat > "$TEST_DIR/model_repo/hello_world/1/model.py" << 'EOF'
import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        print("[hello_world] initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out = pb_utils.Tensor("OUTPUT0", np.array(["hello from triton"], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        print("[hello_world] finalized")
EOF

echo "Test model created at $TEST_DIR/model_repo/"

# ── Step 5: Run tritonserver inside RunPod image ──────────────
echo ""
echo "============================================"
echo " Step 5: Running tritonserver inside RunPod image"
echo " (no GPU needed — model runs on CPU)"
echo "============================================"
docker run --rm \
    -v "$TEST_DIR/tritonserver:/opt/tritonserver" \
    -v "$TEST_DIR/model_repo:/model_repo" \
    "$RUNPOD_IMAGE" \
    bash -c "
        export LD_LIBRARY_PATH=/opt/tritonserver/lib:\${LD_LIBRARY_PATH:-}
        echo '--- ldd missing libs (empty = good) ---'
        ldd /opt/tritonserver/bin/tritonserver 2>&1 | grep 'not found' || echo 'none missing'
        echo ''
        echo '--- Starting tritonserver (will exit after 10s) ---'
        timeout 15 /opt/tritonserver/bin/tritonserver \
            --model-repository=/model_repo \
            --backend-directory=/opt/tritonserver/backends \
            --backend-config=python,python-runtime=\$(which python3) \
            --log-info=true \
            --allow-metrics=false 2>&1 | head -60 || true
        echo ''
        echo '--- Done ---'
    "

echo ""
echo "============================================"
echo " Results interpretation:"
echo "  'Server started' or 'Started HTTPService' = binary is portable ✓"
echo "  'not found' in ldd = missing libs, fixable with apt"
echo "  Segfault / GLIBC error = ABI mismatch, need different approach"
echo "============================================"
