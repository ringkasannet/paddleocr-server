#!/usr/bin/env bash
# Collect GLM-OCR server diagnostic info for cross-server comparison.
# Usage: bash collect_server_info.sh [output_file]
# Output defaults to stdout; optionally saved to a file.

OUT="${1:-/dev/stdout}"

{
echo "========================================="
echo " GLM-OCR SERVER DIAGNOSTIC REPORT"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================="

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,pstate,temperature.gpu,power.draw,power.limit,compute_cap \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "--- GPU Full Status ---"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "--- CPU ---"
grep "model name" /proc/cpuinfo | head -1
echo "Threads: $(nproc)"
echo "Physical cores: $(grep "^core id" /proc/cpuinfo | sort -u | wc -l)"

echo ""
echo "--- Memory ---"
free -h

echo ""
echo "--- OS / Kernel ---"
uname -a
cat /etc/os-release 2>/dev/null | grep -E "^NAME=|^VERSION=" || true

echo ""
echo "--- Python Packages ---"
if [ -f /venv/main/bin/activate ]; then
    . /venv/main/bin/activate
    pip show vllm glmocr torch flashinfer-python flash-attn 2>/dev/null \
        | grep -E "^Name:|^Version:|^Location:"
    echo ""
    echo "torch CUDA/cuDNN/SM:"
    python3 -c "
import torch
print('  torch:', torch.__version__)
print('  CUDA (torch):', torch.version.cuda)
print('  cuDNN:', torch.backends.cudnn.version())
if torch.cuda.is_available():
    print('  GPU SM:', torch.cuda.get_device_capability())
    print('  GPU name:', torch.cuda.get_device_name(0))
    print('  GPU mem total:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), 'GiB')
"
    echo ""
    echo "flashinfer version:"
    python3 -c "import flashinfer; print(' ', flashinfer.__version__)" 2>/dev/null \
        || python3 -c "import flashinfer_python; print(' ', flashinfer_python.__version__)" 2>/dev/null \
        || echo "  flashinfer: not importable"
else
    echo "venv not found at /venv/main"
fi

echo ""
echo "--- vLLM Launch Command ---"
if [ -f /opt/supervisor-scripts/vllm-0.sh ]; then
    cat /opt/supervisor-scripts/vllm-0.sh
else
    echo "Not found: /opt/supervisor-scripts/vllm-0.sh"
fi

echo ""
echo "--- glmocr Config (key values) ---"
if [ -f /etc/glmocr_config.yaml ]; then
    grep -E "port:|max_tokens:|batch_size:|device:|max_workers:|gpu_memory|api_port:|enabled:" \
        /etc/glmocr_config.yaml | head -20
else
    echo "Not found: /etc/glmocr_config.yaml"
fi

echo ""
echo "--- Supervisor Services ---"
supervisorctl status 2>/dev/null || echo "supervisord not running"

echo ""
echo "--- GPU Memory by Process ---"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null \
    || echo "not available"

echo ""
echo "--- Attention Backend (from vLLM log) ---"
LOG=/var/log/supervisor/vllm-0.log
if [ -f "$LOG" ]; then
    grep -E "Using.*backend|Using FLASH|attention backend out of|enable_flashinfer_autotune" "$LOG" \
        | grep -v "splitting_ops\|cudagraph\|compilation_config" \
        | tail -10
else
    echo "Log not found: $LOG"
fi

echo ""
echo "--- vLLM Throughput (last 5 log entries) ---"
if [ -f "$LOG" ]; then
    grep "Avg prompt throughput" "$LOG" | tail -5
else
    echo "Log not found"
fi

echo ""
echo "--- Torch Compile Cache ---"
CACHE_DIR="$HOME/.cache/vllm/torch_compile_cache"
if [ -d "$CACHE_DIR" ]; then
    echo "Cache exists: $CACHE_DIR"
    du -sh "$CACHE_DIR" 2>/dev/null
    ls "$CACHE_DIR" 2>/dev/null
else
    echo "No torch compile cache found at $CACHE_DIR"
fi

echo ""
echo "========================================="
echo " END OF REPORT"
echo "========================================="
} | tee "$OUT"
