#!/usr/bin/env bash
# Collect GLM-OCR server diagnostic info for cross-server comparison.
# Usage: bash collect_server_info.sh [output_file]

OUT="${1:-}"

collect() {

echo "========================================="
echo " GLM-OCR SERVER DIAGNOSTIC REPORT"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================="

# ── GPU ───────────────────────────────────────────────────────────────────────
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,pstate,temperature.gpu,power.draw,power.limit,compute_cap \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""
nvidia-smi 2>/dev/null

# ── PCIe ──────────────────────────────────────────────────────────────────────
echo ""
echo "--- PCIe ---"
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
    --format=csv,noheader 2>/dev/null || echo "pcie query not available"
GPU_BDF_FULL=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_BDF_SHORT=$(echo "$GPU_BDF_FULL" | sed 's/^0000://')
SYS_PCI="/sys/bus/pci/devices/${GPU_BDF_FULL}"
[ -d "$SYS_PCI" ] || SYS_PCI="/sys/bus/pci/devices/0000:${GPU_BDF_SHORT}"
if [ -d "$SYS_PCI" ]; then
    echo "current_link_speed : $(cat $SYS_PCI/current_link_speed 2>/dev/null)"
    echo "max_link_speed     : $(cat $SYS_PCI/max_link_speed 2>/dev/null)"
    echo "current_link_width : x$(cat $SYS_PCI/current_link_width 2>/dev/null)"
    echo "max_link_width     : x$(cat $SYS_PCI/max_link_width 2>/dev/null)"
fi
if ! command -v lspci &>/dev/null; then
    apt-get install -y -q pciutils 2>/dev/null || true
fi
lspci -v -s "$GPU_BDF_SHORT" 2>/dev/null || echo "lspci not available"

# ── CPU ───────────────────────────────────────────────────────────────────────
echo ""
echo "--- CPU ---"
lscpu 2>/dev/null || grep "model name" /proc/cpuinfo | head -1
echo ""
echo "CPU frequency limits:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null \
    | awk '{printf "  scaling_min_freq: %d MHz\n", $1/1000}' || true
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null \
    | awk '{printf "  scaling_max_freq: %d MHz\n", $1/1000}' || true
cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null \
    | awk '{printf "  cpuinfo_max_freq: %d MHz\n", $1/1000}' || true
echo "  governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "  current MHz (all cores): $(grep '^cpu MHz' /proc/cpuinfo | awk '{print $4}' | tr '\n' ' ')"

# ── Memory ────────────────────────────────────────────────────────────────────
echo ""
echo "--- Memory ---"
free -h
echo ""
# Try to get memory speed via dmidecode
if command -v dmidecode &>/dev/null; then
    dmidecode -t memory 2>/dev/null | grep -E "Speed|Type:|Size:" | grep -v "Unknown\|No Module" | head -20
else
    echo "(dmidecode not available — install with: apt-get install dmidecode)"
fi

# ── Process limits ────────────────────────────────────────────────────────────
echo ""
echo "--- Process Limits (ulimit) ---"
ulimit -a 2>/dev/null

# ── OS / Kernel ───────────────────────────────────────────────────────────────
echo ""
echo "--- OS / Kernel ---"
uname -a
grep -E "^NAME=|^VERSION=" /etc/os-release 2>/dev/null || true

# ── Storage ───────────────────────────────────────────────────────────────────
echo ""
echo "--- Storage ---"
df -h / 2>/dev/null
echo ""
lsblk -d -o NAME,SIZE,ROTA,TYPE,MODEL 2>/dev/null || true

# ── Python Packages ───────────────────────────────────────────────────────────
echo ""
echo "--- Python Packages ---"
if [ -f /venv/main/bin/activate ]; then
    . /venv/main/bin/activate 2>/dev/null
    pip show vllm glmocr torch flashinfer-python flash-attn 2>/dev/null \
        | grep -E "^Name:|^Version:|^Location:"
    echo ""
    echo "GPU properties (torch):"
    python3 -c "
import torch
print('  torch:', torch.__version__)
print('  CUDA (torch):', torch.version.cuda)
print('  cuDNN:', torch.backends.cudnn.version())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    bw = 2 * p.memory_clock_rate * 1e3 * p.memory_bus_width / 8 / 1e9
    print('  GPU SM:', torch.cuda.get_device_capability())
    print('  GPU name:', p.name)
    print('  GPU mem total:', round(p.total_memory/1024**3, 2), 'GiB')
    print('  Multi-processor count:', p.multi_processor_count)
    print('  Memory clock (MHz):', p.memory_clock_rate // 1000)
    print('  Memory bus width (bits):', p.memory_bus_width)
    print('  Peak mem bandwidth (GB/s):', round(bw, 1))
" 2>/dev/null
    echo ""
    echo "flashinfer version:"
    python3 -c "import flashinfer; print(' ', flashinfer.__version__)" 2>/dev/null \
        || echo "  not importable"
else
    echo "venv not found at /venv/main"
fi

# ── vLLM Launch Command ───────────────────────────────────────────────────────
echo ""
echo "--- vLLM Launch Command ---"
cat /opt/supervisor-scripts/vllm-0.sh 2>/dev/null || echo "Not found"

# ── glmocr Config ─────────────────────────────────────────────────────────────
echo ""
echo "--- glmocr Config (key values) ---"
if [ -f /etc/glmocr_config.yaml ]; then
    grep -E "port:|max_tokens:|batch_size:|device:|max_workers:|gpu_memory|api_port:|enabled:" \
        /etc/glmocr_config.yaml | head -20
else
    echo "Not found: /etc/glmocr_config.yaml"
fi

# ── Services ──────────────────────────────────────────────────────────────────
echo ""
echo "--- Supervisor Services ---"
supervisorctl status 2>/dev/null || echo "supervisord not running"

echo ""
echo "--- GPU Memory by Process ---"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null \
    || echo "not available"

# ── vLLM Runtime ─────────────────────────────────────────────────────────────
echo ""
echo "--- Attention Backend (from vLLM log) ---"
LOG=/var/log/supervisor/vllm-0.log
if [ -f "$LOG" ]; then
    grep -E "Using.*backend|Using FLASH|attention backend out of" "$LOG" \
        | grep -v "splitting_ops\|cudagraph\|compilation_config" \
        | tail -6
else
    echo "Log not found: $LOG"
fi

echo ""
echo "--- vLLM Throughput (last 5 entries) ---"
[ -f "$LOG" ] && grep "Avg prompt throughput" "$LOG" | tail -5 || echo "Log not found"

echo ""
echo "--- Torch Compile Cache ---"
CACHE_DIR="$HOME/.cache/vllm/torch_compile_cache"
if [ -d "$CACHE_DIR" ]; then
    echo "Exists: $CACHE_DIR"
    du -sh "$CACHE_DIR" 2>/dev/null
    ls "$CACHE_DIR" 2>/dev/null
else
    echo "No cache at $CACHE_DIR"
fi

# ── SUMMARY ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo " SUMMARY"
echo "========================================="

GPU_NAME=$(nvidia-smi --query-gpu=name            --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_MEM=$(nvidia-smi  --query-gpu=memory.total    --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_SM=$(nvidia-smi   --query-gpu=compute_cap     --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_PWR=$(nvidia-smi  --query-gpu=power.limit     --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)
PCIE_GEN=$(nvidia-smi     --query-gpu=pcie.link.gen.current   --format=csv,noheader 2>/dev/null | head -1 | xargs)
PCIE_GEN_MAX=$(nvidia-smi --query-gpu=pcie.link.gen.max       --format=csv,noheader 2>/dev/null | head -1 | xargs)
PCIE_WIDTH=$(nvidia-smi     --query-gpu=pcie.link.width.current --format=csv,noheader 2>/dev/null | head -1 | xargs)
PCIE_WIDTH_MAX=$(nvidia-smi --query-gpu=pcie.link.width.max     --format=csv,noheader 2>/dev/null | head -1 | xargs)
_BDF_F=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | xargs)
_BDF_S=$(echo "$_BDF_F" | sed 's/^0000://')
_SYS="/sys/bus/pci/devices/${_BDF_F}"
[ -d "$_SYS" ] || _SYS="/sys/bus/pci/devices/0000:${_BDF_S}"
PCIE_SPEED_CUR=$(cat "$_SYS/current_link_speed" 2>/dev/null | xargs)
PCIE_SPEED_MAX=$(cat "$_SYS/max_link_speed"     2>/dev/null | xargs)

CPU_NAME=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
CPU_THREADS=$(nproc)
CPU_CORES=$(grep "^core id" /proc/cpuinfo | sort -u | wc -l)
CPU_BASE=$(lscpu 2>/dev/null | grep "^CPU MHz:" | awk '{print $3}' | xargs)
CPU_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null | awk '{printf "%d MHz", $1/1000}')
CPU_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)
RAM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')

if [ -f /venv/main/bin/activate ]; then
    { . /venv/main/bin/activate; } 2>&1 | grep -v "" > /dev/null || true
    VLLM_VER=$(pip show vllm            2>/dev/null | grep "^Version:" | awk '{print $2}')
    GLMOCR_VER=$(pip show glmocr        2>/dev/null | grep "^Version:" | awk '{print $2}')
    TORCH_VER=$(pip show torch          2>/dev/null | grep "^Version:" | awk '{print $2}')
    FLASHINFER_VER=$(pip show flashinfer-python 2>/dev/null | grep "^Version:" | awk '{print $2}')
    [ -z "$FLASHINFER_VER" ] && FLASHINFER_VER="not installed"
    MEM_BW=$(python3 -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    bw = 2 * p.memory_clock_rate * 1e3 * p.memory_bus_width / 8 / 1e9
    print(f'{round(bw,1)} GB/s  (clock={p.memory_clock_rate//1000} MHz, bus={p.memory_bus_width}-bit)')
" 2>/dev/null)
fi

ATTN_BACKEND="unknown"; ATTN_CANDIDATES=""
if [ -f "$LOG" ]; then
    ATTN_LINE=$(grep "attention backend out of" "$LOG" | tail -1)
    [ -n "$ATTN_LINE" ] && ATTN_BACKEND=$(echo "$ATTN_LINE" | grep -oP "Using \K\w+" | head -1)
    [ -n "$ATTN_LINE" ] && ATTN_CANDIDATES=$(echo "$ATTN_LINE" | grep -oP "potential backends: \K\[.*\]" | head -1)
fi

GPU_UTIL=$(grep "\-\-gpu-memory-utilization" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null | grep -oP '[\d.]+' | head -1)
MAX_MODEL_LEN=$(grep "\-\-max-model-len"         /opt/supervisor-scripts/vllm-0.sh 2>/dev/null | grep -oP '\d+' | head -1)
MAX_NUM_SEQS=$(grep "\-\-max-num-seqs"           /opt/supervisor-scripts/vllm-0.sh 2>/dev/null | grep -oP '\d+' | head -1)
MAX_BATCHED=$(grep "\-\-max-num-batched-tokens"  /opt/supervisor-scripts/vllm-0.sh 2>/dev/null | grep -oP '\d+' | head -1)
MTP=$(grep -q "speculative-config" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null && echo "enabled" || echo "disabled")
TRITON=$(grep -q "VLLM_ATTENTION_BACKEND=TRITON" /opt/supervisor-scripts/vllm-0.sh 2>/dev/null && echo "yes" || echo "no")

echo "[ Hardware ]"
printf "  %-26s %s\n" "GPU:"               "$GPU_NAME"
printf "  %-26s %s\n" "GPU Memory:"        "$GPU_MEM"
printf "  %-26s %s\n" "GPU Compute SM:"    "$GPU_SM"
printf "  %-26s %s\n" "GPU Power Limit:"   "$GPU_PWR"
printf "  %-26s %s\n" "GPU Mem Bandwidth:" "${MEM_BW:-n/a}"
printf "  %-26s %s\n" "PCIe (nvidia-smi):" "Gen${PCIE_GEN}/${PCIE_GEN_MAX} x${PCIE_WIDTH}/${PCIE_WIDTH_MAX}"
printf "  %-26s %s\n" "PCIe speed (sysfs):" "current=${PCIE_SPEED_CUR:-n/a}  max=${PCIE_SPEED_MAX:-n/a}"
printf "  %-26s %s\n" "Driver:"            "$GPU_DRIVER"
echo ""
printf "  %-26s %s\n" "CPU:"               "$CPU_NAME"
printf "  %-26s %s\n" "Cores / Threads:"   "$CPU_CORES / $CPU_THREADS"
printf "  %-26s %s\n" "CPU Max Freq:"      "${CPU_MAX:-n/a}"
printf "  %-26s %s\n" "CPU Governor:"      "$CPU_GOV"
printf "  %-26s %s\n" "RAM:"               "$RAM_TOTAL"

echo ""
echo "[ Software ]"
printf "  %-26s %s\n" "vLLM:"              "${VLLM_VER:-n/a}"
printf "  %-26s %s\n" "glmocr:"            "${GLMOCR_VER:-n/a}"
printf "  %-26s %s\n" "torch:"             "${TORCH_VER:-n/a}"
printf "  %-26s %s\n" "flashinfer:"        "${FLASHINFER_VER:-n/a}"

echo ""
echo "[ Runtime ]"
printf "  %-26s %s\n" "Attention backend:" "$ATTN_BACKEND"
printf "  %-26s %s\n" "Candidates:"        "${ATTN_CANDIDATES:-n/a}"
printf "  %-26s %s\n" "TRITON_ATTN:"       "$TRITON"

echo ""
echo "[ vLLM Settings ]"
printf "  %-26s %s\n" "gpu-memory-utilization:" "${GPU_UTIL:-n/a}"
printf "  %-26s %s\n" "max-model-len:"          "${MAX_MODEL_LEN:-n/a}"
printf "  %-26s %s\n" "max-num-seqs:"           "${MAX_NUM_SEQS:-n/a}"
printf "  %-26s %s\n" "max-num-batched-tokens:" "${MAX_BATCHED:-n/a}"
printf "  %-26s %s\n" "MTP speculative:"        "$MTP"

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
