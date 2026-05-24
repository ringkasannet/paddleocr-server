#!/usr/bin/env bash
# Collect GLM-OCR server diagnostic info for cross-server comparison.
# Usage: bash collect_server_info.sh [output_file]
# Output defaults to stdout; optionally saved to a file.

OUT="${1:-}"

collect() {
echo "========================================="
echo " GLM-OCR SERVER DIAGNOSTIC REPORT"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================="

echo ""
echo "--- GPU (summary) ---"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,pstate,temperature.gpu,power.draw,power.limit,compute_cap \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "--- GPU Full Status ---"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "--- PCIe Info ---"
# GPU PCI slot and link speed/width
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
    --format=csv,noheader 2>/dev/null || echo "pcie query not available"
# Full lspci entry for the GPU
GPU_BDF=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | sed 's/^0000://')
if [ -n "$GPU_BDF" ]; then
    lspci -v -s "$GPU_BDF" 2>/dev/null || echo "lspci not available"
fi

echo ""
echo "--- CPU ---"
grep "model name" /proc/cpuinfo | head -1
echo "Threads: $(nproc)"
echo "Physical cores: $(grep "^core id" /proc/cpuinfo | sort -u | wc -l)"
echo "CPU MHz (first core): $(grep "^cpu MHz" /proc/cpuinfo | head -1)"

echo ""
echo "--- Memory ---"
free -h

echo ""
echo "--- OS / Kernel ---"
uname -a
grep -E "^NAME=|^VERSION=" /etc/os-release 2>/dev/null || true

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
    props = torch.cuda.get_device_properties(0)
    print('  GPU SM:', torch.cuda.get_device_capability())
    print('  GPU name:', props.name)
    print('  GPU mem total:', round(props.total_memory/1024**3, 2), 'GiB')
    print('  Multi-processor count:', props.multi_processor_count)
    print('  Memory clock (MHz):', props.memory_clock_rate // 1000)
    print('  Memory bus width (bits):', props.memory_bus_width)
    bw = 2 * props.memory_clock_rate * 1e3 * props.memory_bus_width / 8 / 1e9
    print('  Peak mem bandwidth (GB/s):', round(bw, 1))
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
echo " SUMMARY"
echo "========================================="

# GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)
PCIE_GEN=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader 2>/dev/null | head -1 | xargs)
PCIE_GEN_MAX=$(nvidia-smi --query-gpu=pcie.link.gen.max --format=csv,noheader 2>/dev/null | head -1 | xargs)
PCIE_WIDTH=$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader 2>/dev/null | head -1 | xargs)
PCIE_WIDTH_MAX=$(nvidia-smi --query-gpu=pcie.link.width.max --format=csv,noheader 2>/dev/null | head -1 | xargs)

# CPU
CPU_NAME=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
CPU_THREADS=$(nproc)
CPU_CORES=$(grep "^core id" /proc/cpuinfo | sort -u | wc -l)

# RAM
RAM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')

# Packages
if [ -f /venv/main/bin/activate ]; then
    . /venv/main/bin/activate 2>/dev/null
    VLLM_VER=$(pip show vllm 2>/dev/null | grep "^Version:" | awk '{print $2}')
    GLMOCR_VER=$(pip show glmocr 2>/dev/null | grep "^Version:" | awk '{print $2}')
    TORCH_VER=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}')
    FLASHINFER_VER=$(pip show flashinfer-python 2>/dev/null | grep "^Version:" | awk '{print $2}')
    [ -z "$FLASHINFER_VER" ] && FLASHINFER_VER="not installed"
fi

# Attention backend (most recent)
LOG=/var/log/supervisor/vllm-0.log
ATTN_BACKEND="unknown"
if [ -f "$LOG" ]; then
    ATTN_LINE=$(grep "attention backend out of" "$LOG" | tail -1)
    if [ -n "$ATTN_LINE" ]; then
        ATTN_BACKEND=$(echo "$ATTN_LINE" | grep -oP "Using \K\w+" | head -1)
        ATTN_CANDIDATES=$(echo "$ATTN_LINE" | grep -oP "potential backends: \K\[.*\]" | head -1)
    fi
fi

# GPU memory stats from torch
MEM_BW=""
if [ -f /venv/main/bin/activate ]; then
    MEM_BW=$(python3 -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    bw = 2 * p.memory_clock_rate * 1e3 * p.memory_bus_width / 8 / 1e9
    print(f'{round(bw,1)} GB/s  (clock={p.memory_clock_rate//1000} MHz, bus={p.memory_bus_width}-bit)')
" 2>/dev/null)
fi

# vLLM args
GPU_UTIL=$(grep "\-\-gpu-memory-utilization" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null | grep -oP '[\d.]+' | head -1)
MAX_MODEL_LEN=$(grep "\-\-max-model-len" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null | grep -oP '\d+' | head -1)
MAX_NUM_SEQS=$(grep "\-\-max-num-seqs" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null | grep -oP '\d+' | head -1)
MAX_BATCHED=$(grep "\-\-max-num-batched-tokens" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null | grep -oP '\d+' | head -1)
MTP=$(grep -c "speculative-config" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null && echo "enabled" || echo "disabled")
MTP=$(grep -q "speculative-config" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null && echo "enabled" || echo "disabled")
TRITON=$(grep -q "VLLM_ATTENTION_BACKEND=TRITON" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null && echo "yes" || echo "no")
VLLM_SEM=$(grep -q "VLLM_SEMAPHORE=1" /opt/supervisor-scripts/glmocr.sh 2>/dev/null && echo "1" || echo "0")

printf "%-28s %s\n" "GPU:"               "$GPU_NAME"
printf "%-28s %s\n" "GPU Memory:"        "$GPU_MEM"
printf "%-28s %s\n" "GPU SM:"            "$GPU_SM"
printf "%-28s %s\n" "GPU Mem Bandwidth:" "${MEM_BW:-n/a}"
printf "%-28s %s\n" "PCIe:"              "Gen${PCIE_GEN}/${PCIE_GEN_MAX} x${PCIE_WIDTH}/${PCIE_WIDTH_MAX}"
printf "%-28s %s\n" "Driver:"            "$GPU_DRIVER"
echo  ""
printf "%-28s %s\n" "CPU:"               "$CPU_NAME"
printf "%-28s %s\n" "Cores / Threads:"   "$CPU_CORES / $CPU_THREADS"
printf "%-28s %s\n" "RAM:"               "$RAM_TOTAL"
echo ""
printf "%-28s %s\n" "vLLM:"              "${VLLM_VER:-n/a}"
printf "%-28s %s\n" "glmocr:"            "${GLMOCR_VER:-n/a}"
printf "%-28s %s\n" "torch:"             "${TORCH_VER:-n/a}"
printf "%-28s %s\n" "flashinfer:"        "${FLASHINFER_VER:-n/a}"
echo ""
printf "%-28s %s\n" "Attention backend:" "$ATTN_BACKEND"
printf "%-28s %s\n" "Candidates:"        "${ATTN_CANDIDATES:-n/a}"
printf "%-28s %s\n" "TRITON_ATTN:"       "$TRITON"
echo ""
printf "%-28s %s\n" "gpu-memory-utilization:" "${GPU_UTIL:-n/a}"
printf "%-28s %s\n" "max-model-len:"     "${MAX_MODEL_LEN:-n/a}"
printf "%-28s %s\n" "max-num-seqs:"      "${MAX_NUM_SEQS:-n/a}"
printf "%-28s %s\n" "max-num-batched-tokens:" "${MAX_BATCHED:-n/a}"
printf "%-28s %s\n" "MTP speculative:"   "$MTP"

echo ""
echo "========================================="
echo " END OF REPORT"
echo "========================================="
}

if [ -n "$OUT" ]; then
    collect | tee "$OUT"
else
    collect
fi
