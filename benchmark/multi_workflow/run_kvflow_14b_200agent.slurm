#!/bin/bash
# =============================================================================
# KVFlow 200-Agent Eviction War Benchmark
# Tests Priority vs LRU under extreme cache pressure with Qwen3-14B
#
# Key improvements over previous 8B experiments:
# 1. 14B model has longer prefill time → larger absolute TTFT savings
# 2. 14B KV cache per token is ~1.75x larger → more cache pressure
# 3. KV pressure ~1.5M tokens/round with 14B → extreme cache pressure
#
# Two configs tested:
#   hicache90k: LRU eviction + write_back (baseline)
#   kvflow:    Priority eviction + write_back + prefetch
#
# Usage:
#   sbatch run_kvflow_14b_200agent.slurm
#   sbatch --export=MODEL=Qwen3-8b run_kvflow_14b_200agent.slurm
# =============================================================================

#SBATCH --job-name=kvflow-14b-200ag
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kvflow-14b-200ag/slurm-%j.out
#SBATCH --error=logs/kvflow-14b-200ag/slurm-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Load miniconda module so conda env is available on compute nodes
module load miniconda/py312_24.7.1-0

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
# NOTE: Use ABSOLUTE path. Using relative path (e.g. "$SCRIPT_DIR/../..") resolves
# to /var/spool/slurm/ on compute nodes because SLURM copies the script there.
SGLANG_ROOT_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow"

# Model selection: Qwen3-14B (default) or Qwen3-8B
MODEL_NAME="${MODEL:-Qwen3-14B}"
if [[ "$MODEL_NAME" == "Qwen3-8b" ]]; then
    MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
    TP_SIZE=2
    MEM_FRAC=0.80
    MODEL_LABEL="8b"
else
    MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-14B"
    TP_SIZE=2
    MEM_FRAC=0.72
    MODEL_LABEL="14b"
fi

# Log directories
LOG_DIR="${LOG_DIR:-/home/comp/25480812/CodeMAS_Project/logs/kvflow-14b-200ag}"
RESULT_DIR="$LOG_DIR/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Conda configuration
CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PATH/bin/python}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$MODEL_NAME] $*" | tee -a "$LOG_DIR/slurm.log"; }

# =============================================================================
# Helper Functions
# =============================================================================

kill_port() {
    local p="$1"
    if pid=$(lsof -ti :"$p" 2>/dev/null); then
        log "Killing port $p (PID=$pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 8
    fi
}

wait_ready() {
    local config="$1"
    local port="$2"
    local max_wait="${3:-1200}"
    local log_file="$LOG_DIR/server_${config}_${port}.log"
    log "Waiting for server on port $port (polling log file)..."
    for i in $(seq 1 "$max_wait"); do
        # Check if server log shows readiness
        if grep -q "ready to roll" "$log_file" 2>/dev/null; then
            log "Server on port $port ready after $((i*5))s!"
            return 0
        fi
        # Also check if server process died
        if ! kill -0 "$(cat "$LOG_DIR/.server_${config}_${port}.pid" 2>/dev/null)" 2>/dev/null; then
            log "ERROR: Server process died!"
            tail -20 "$log_file" 2>/dev/null || true
            return 1
        fi
        [ $((i % 12)) -eq 0 ] && log "  ...still waiting, $((i*5))s..."
        sleep 5
    done
    log "ERROR: Server on port $port did not start (timeout)"
    tail -20 "$log_file" 2>/dev/null || true
    return 1
}

wait_gpu_free() {
    local max_wait="${1:-180}"
    log "Waiting for GPU memory to free (threshold: 5000MiB)..."
    for i in $(seq 1 "$max_wait"); do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -lt 5000 ]; then
            log "GPU memory freed: ${used}MiB"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && log "  GPU memory still: ${used}MiB..."
        sleep 5
    done
    log "WARNING: GPU memory still high after ${max_wait}s"
    return 1
}

# =============================================================================
# Server Management
# =============================================================================

start_server() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    log "=========================================="
    log "Starting server: config=$config, port=$port, model=$MODEL_NAME"
    log "=========================================="
    kill_port "$port"

    : "${CUDA_HOME:=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda)))}"
    export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    # Add conda env bin to PATH so tvm_ffi JIT can find ninja
    export PATH="/home/comp/25480812/.conda/envs/sglang-kvflow/bin:$PATH"
    # Use system GCC 8 for JIT kernel compilation. A <version> stub header
    # (jit_kernel/include/version) defines __cpp_lib_source_location=0 so the
    # source_location.h fallback compiles cleanly with GCC 8's C++17 mode.
    export PYTHONPATH="${SGLANG_ROOT_DIR}/benchmark:${SGLANG_ROOT_DIR}/python${PYTHONPATH:+:$PYTHONPATH}"

    local RADIX="lru"
    local HICACHE_FLAGS=""
    local MAX_TOKENS="--max-total-tokens 90000"

    if [[ "$config" == "kvflow" ]]; then
        RADIX="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"
    elif [[ "$config" == "hicache90k" ]]; then
        RADIX="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back"
    fi

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size $TP_SIZE \
        --mem-fraction-static $MEM_FRAC \
        $MAX_TOKENS \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        --radix-eviction-policy "$RADIX" \
        $HICACHE_FLAGS \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
        --skip-server-warmup \
        --log-level info \
        > "$log_file" 2>&1 &

    log "Server PID=$! (log: $log_file)"
    echo "$!" > "$LOG_DIR/.server_${config}_${port}.pid"
}

stop_server() {
    local config="$1"
    local port="$2"
    kill_port "$port"
    rm -f "$LOG_DIR/.server_${config}_${port}.pid"
    sleep 15
    pkill -9 -f "sglang.launch_server.*--port $port" 2>/dev/null || true
    sleep 10
}

# =============================================================================
# Benchmark Runner
# =============================================================================

run_benchmark() {
    local config="$1"
    local port="$2"
    shift 2
    local extra_args="$*"
    local bench_log="$LOG_DIR/bench_${config}_${port}.log"

    log "Running $config benchmark on port $port..."
    cd "$SGLANG_ROOT_DIR"

    # Build result filename based on config
    local result_prefix="mwf_${config}_${MODEL_LABEL}_200ag_4096shr_64uni"
    local result_file="$RESULT_DIR/${result_prefix}.json"

    # Export PYTHONPATH so Python can find the benchmark module.
    # Also add conda bin to PATH for tvm_ffi JIT (ninja).
    export PYTHONNOUSERSITE=1
    export PATH="/home/comp/25480812/.conda/envs/sglang-kvflow/bin:$PATH"
    export CXX="/home/comp/25480812/.conda/envs/sglang-kvflow/bin/x86_64-conda-linux-gnu-g++"
    export CC="/home/comp/25480812/.conda/envs/sglang-kvflow/bin/x86_64-conda-linux-gnu-gcc"
    export PYTHONPATH="${SGLANG_ROOT_DIR}/benchmark:${SGLANG_ROOT_DIR}/python:${PYTHONPATH:-}"

    # Quick diagnostic: verify module is reachable before running
    if ! "$PYTHON_BIN" -c "import benchmark; print('benchmark OK')" 2>&1 | tee -a "$bench_log"; then
        log "ERROR: Cannot import benchmark module. PYTHONPATH=$PYTHONPATH"
        log "       SGLANG_ROOT_DIR=$SGLANG_ROOT_DIR"
        log "       Benchmark dir: $SGLANG_ROOT_DIR/benchmark/"
        ls -la "$SGLANG_ROOT_DIR/benchmark/" 2>&1 | tee -a "$bench_log" || true
        return 1
    fi

    "$PYTHON_BIN" -m benchmark.multi_workflow.bench_multi_workflow \
        --config "$config" \
        --host 127.0.0.1 \
        --port "$port" \
        --agents-seed 42 \
        --seed 42 \
        --output-dir "$RESULT_DIR" \
        --num-workflows 20 \
        --agents-per-workflow 10 \
        --shared-p-len 4096 \
        --unique-p-len 64 \
        --suffix-len 32 \
        --output-len 32 \
        --num-rounds 10 \
        --warmup-rounds 1 \
        $extra_args \
        >> "$bench_log" 2>&1

    log "$config benchmark completed. Results in $RESULT_DIR"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "=========================================="
    log "KVFlow 200-Agent Eviction War"
    log "Model: $MODEL_NAME ($MODEL_PATH)"
    log "TP: $TP_SIZE, MEM_FRAC: $MEM_FRAC"
    log "LOG_DIR: $LOG_DIR"
    log "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
    log "=========================================="
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
    log "=========================================="

    local PORT_LRU=30081
    local PORT_KVFLOW=30082

    # ---- hicache90k (LRU baseline) ----
    log "=========================================="
    log "STEP 1: hicache90k (LRU baseline)"
    log "Config: 20wf x 10ag, shared=4096, unique=64, rounds=10"
    log "=========================================="
    start_server "hicache90k" "$PORT_LRU"
    wait_ready "hicache90k" "$PORT_LRU" 1200 || { log "ERROR: hicache90k server failed"; exit 1; }
    sleep 5
    run_benchmark "hicache90k" "$PORT_LRU"
    stop_server "hicache90k" "$PORT_LRU"
    wait_gpu_free 120
    sleep 30
    sync
    sleep 15

    # ---- kvflow (Priority eviction) ----
    log "=========================================="
    log "STEP 2: kvflow (Priority eviction)"
    log "Config: 20wf x 10ag, shared=4096, unique=64, rounds=10"
    log "=========================================="
    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "kvflow" "$PORT_KVFLOW" 1200 || { log "ERROR: kvflow server failed"; exit 1; }
    sleep 5

    # Reference hicache90k result as baseline
    local LRU_JSON=$(ls -t "$RESULT_DIR"/mwf_hicache90k_${MODEL_LABEL}_200ag*.json 2>/dev/null | head -1 || true)
    if [ -n "$LRU_JSON" ]; then
        log "Baseline from hicache90k: $LRU_JSON"
        run_benchmark "kvflow" "$PORT_KVFLOW" --baseline-json "$LRU_JSON"
    else
        run_benchmark "kvflow" "$PORT_KVFLOW"
    fi
    stop_server "kvflow" "$PORT_KVFLOW"

    # ---- Summary ----
    log "=========================================="
    log "BENCHMARK COMPLETE"
    log "=========================================="
    log "Results: $RESULT_DIR"
    log "JSON files:"
    ls -lh "$RESULT_DIR"/mwf_*.json 2>/dev/null | awk '{print "  "$NF}'
    log ""
    log "KV pressure per round (estimated for 14B):"
    log "  20wf x 10ag x (4096+64) = ~830k tokens"
    log "  90k cache: ~9x pressure"
    log "  Expect Priority advantage in TTFT (prefill savings)"
    log "=========================================="

    # Copy this script for reproducibility
    cp "$0" "$LOG_DIR/benchmark_script_${MODEL_LABEL}.sh"
}

main "$@"
