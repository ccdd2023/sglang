#!/bin/bash
# =============================================================================
# Multi-Workflow KVFlow Benchmark - Server Launch Script
#
# Launches SGLang server with configurable eviction policy and HiCache settings.
#
# Usage:
#   ./run_server.sh priority   30000   # Priority eviction, port 30000
#   ./run_server.sh lru       30001   # LRU eviction, port 30001
#   ./run_server.sh priority_wb 30002  # Priority + write_back HiCache, port 30002
#
# The script kills any existing server on the target port before starting.
# =============================================================================

set -euo pipefail

EVICTION="${1:-priority}"
PORT="${2:-30000}"
MODEL_PATH="${MODEL_PATH:-/home/gfy/models/Qwen2.5-3B-Instruct}"
LOG_DIR="${LOG_DIR:-/home/gfy/CodeMAS_Project/logs/kvflow-multi-workflow}"

# Try to find conda environment
if [[ -f "/usr/local/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh" ]]; then
    source "/usr/local/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh"
    if conda env list | grep -q "^sglang-kvflow "; then
        conda activate sglang-kvflow
    elif conda env list | grep -q "^sglang "; then
        conda activate sglang
    fi
fi

SGLANG_ROOT_DIR="/home/gfy/CodeMAS_Project/sglang-kvflow"
mkdir -p "$LOG_DIR"

# Port availability check and cleanup
cleanup_port() {
    local p="$1"
    echo "[INFO] Checking port $p..."
    if pid=$(lsof -ti :"$p" 2>/dev/null); then
        echo "[INFO] Killing existing process on port $p (PID=$pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
    fi
}

cleanup_port "$PORT"

# Derive configuration from eviction mode
RADIX_POLICY=""
HICACHE_ENABLE=""
HICACHE_RATIO=""
HICACHE_WRITE=""
PREFETCH_ENABLE=""
MAX_TOTAL_TOKENS=""

case "$EVICTION" in
    lru)
        RADIX_POLICY="lru"
        HICACHE_ENABLE="--enable-hierarchical-cache"
        HICACHE_RATIO="--hicache-ratio 1.5"
        HICACHE_WRITE="--hicache-write-policy write_through"
        PREFETCH_ENABLE=""
        MAX_TOTAL_TOKENS="--max-total-tokens 60000"
        ;;
    priority)
        RADIX_POLICY="priority"
        HICACHE_ENABLE="--enable-hierarchical-cache"
        HICACHE_RATIO="--hicache-ratio 1.5"
        HICACHE_WRITE="--hicache-write-policy write_through"
        PREFETCH_ENABLE=""
        MAX_TOTAL_TOKENS="--max-total-tokens 60000"
        ;;
    priority_wb)
        # Priority + write_back HiCache (the key fix for the prefetch deadlock)
        # write_back defers CPU writes, avoiding the evictable_host_leaves lock issue
        RADIX_POLICY="priority"
        HICACHE_ENABLE="--enable-hierarchical-cache"
        HICACHE_RATIO="--hicache-ratio 2.0"
        HICACHE_WRITE="--hicache-write-policy write_back"
        PREFETCH_ENABLE="--enable-hicache-prefetch"
        MAX_TOTAL_TOKENS="--max-total-tokens 90000"
        ;;
    lru_wb)
        # LRU + write_back HiCache for fair comparison
        RADIX_POLICY="lru"
        HICACHE_ENABLE="--enable-hierarchical-cache"
        HICACHE_RATIO="--hicache-ratio 2.0"
        HICACHE_WRITE="--hicache-write-policy write_back"
        PREFETCH_ENABLE="--enable-hicache-prefetch"
        MAX_TOTAL_TOKENS="--max-total-tokens 90000"
        ;;
    *)
        echo "[ERROR] Unknown eviction mode: $EVICTION"
        echo "Valid modes: lru, priority, priority_wb, lru_wb"
        exit 1
        ;;
esac

SERVER_LOG="$LOG_DIR/server_${EVICTION}_${PORT}.log"

echo "=============================================="
echo "SGLang KVFlow Server Launch"
echo "=============================================="
echo "  Eviction policy : $RADIX_POLICY"
echo "  Port            : $PORT"
echo "  HiCache         : $HICACHE_ENABLE (ratio=$HICACHE_RATIO, write=$HICACHE_WRITE)"
echo "  Prefetch        : ${PREFETCH_ENABLE:-disabled}"
echo "  Max total tokens : $MAX_TOTAL_TOKENS"
echo "  Log file        : $SERVER_LOG"
echo "  Model           : $MODEL_PATH"
echo "=============================================="

cd "$SGLANG_ROOT_DIR"

# Find and activate conda environment
if [[ -f "/usr/local/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh" ]]; then
    source "/usr/local/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh"
    if conda env list | grep -q "^sglang-kvflow "; then
        conda activate sglang-kvflow
    elif conda env list | grep -q "^sglang "; then
        conda activate sglang
    fi
fi

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tokenizer-path "$MODEL_PATH" \
    --tokenizer-mode auto \
    --trust-remote-code \
    \
    --mem-fraction-static 0.85 \
    --max-total-tokens 60000 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 8192 \
    \
    --radix-eviction-policy "$RADIX_POLICY" \
    \
    $HICACHE_ENABLE \
    $HICACHE_RATIO \
    $HICACHE_WRITE \
    --hicache-io-backend direct \
    --hicache-mem-layout layer_first \
    $PREFETCH_ENABLE \
    \
    --enable-cache-report \
    \
    --attention-backend flashinfer \
    --sampling-backend flashinfer \
    \
    --tensor-parallel-size 2 \
    --disable-cuda-graph \
    \
    --log-level info \
    2>&1 | tee "$SERVER_LOG"
