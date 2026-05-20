#!/bin/bash
# =============================================================================
# KVFlow Optimal Scenario Benchmark with Qwen3-8B
#
# This script:
# 1. Downloads Qwen3-8B model (if not already present)
# 2. Starts SGLang server with hicache (LRU) configuration
# 3. Runs benchmark with hicache
# 4. Starts SGLang server with kvflow (Priority) configuration
# 5. Runs benchmark with kvflow
# 6. Compares and analyzes results
# =============================================================================

#SBATCH --job-name=kvflow-8b-benchmark
#SBATCH --output=/home/comp/25480812/logs/kvflow-8b/slurm-%j.out
#SBATCH --error=/home/comp/25480812/logs/kvflow-8b/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
SGLANG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-/home/comp/25480812/logs/kvflow-8b}"
RESULT_DIR="$LOG_DIR/results"

# Model configuration - try 8B first, fall back to 1.7B
MODEL_8B_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
MODEL_1_7B_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-1.7B"

# Determine which model to use
if [ -d "$MODEL_8B_PATH" ]; then
    MODEL_PATH="$MODEL_8B_PATH"
    MODEL_NAME="Qwen3-8B"
elif [ -d "$MODEL_1_7B_PATH/snapshots" ]; then
    # Find actual snapshot
    SNAPSHOT=$(ls -d $MODEL_1_7B_PATH/snapshots/*/ 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "")
    if [ -n "$SNAPSHOT" ]; then
        MODEL_PATH="$MODEL_1_7B_PATH/snapshots/$SNAPSHOT"
        MODEL_NAME="Qwen3-1.7B"
    else
        MODEL_PATH="$MODEL_1_7B_PATH"
        MODEL_NAME="Qwen3-1.7B"
    fi
else
    echo "ERROR: No model found at $MODEL_1_7B_PATH"
    exit 1
fi

# Python environment
CONDA_ENV="/home/comp/25480812/.conda/envs/sglang-kvflow"
PYTHON_BIN="$CONDA_ENV/bin/python"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

log() { echo "[$(date '+%H:%M:%S')] [SLURM:${SLURM_JOB_ID:-local}] $*" | tee -a "$LOG_DIR/slurm.log"; }

# =============================================================================
# Model Download (if needed)
# =============================================================================
download_model() {
    local model_id="$1"
    local target_dir="$2"

    if [ -d "$target_dir" ]; then
        log "Model already exists at $target_dir"
        return 0
    fi

    log "Downloading model $model_id to $target_dir..."

    # Create parent directory
    mkdir -p "$(dirname "$target_dir")"

    # Download using huggingface-cli or python
    $PYTHON_BIN -c "
from huggingface_hub import snapshot_download
import os

os.makedirs('$target_dir', exist_ok=True)
snapshot_download(
    repo_id='$model_id',
    local_dir='$target_dir',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('Download complete!')
"

    if [ -d "$target_dir" ]; then
        log "Model downloaded successfully to $target_dir"
    else
        log "ERROR: Model download failed"
        return 1
    fi
}

# =============================================================================
# Server Management
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
    local port="$1"
    local max_wait="${2:-120}"
    log "Waiting for server on port $port..."
    for i in $(seq 1 "$max_wait"); do
        if curl -sf --max-time 2 "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            log "Server on port $port ready after $((i*5))s!"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && log "  ...still waiting, $((i*5))s..."
        sleep 5
    done
    log "ERROR: Server on port $port did not start"
    tail -20 "$LOG_DIR"/server_*.log 2>/dev/null || true
    return 1
}

wait_gpu_free() {
    local max_wait="${1:-300}"
    log "Waiting for GPU memory to free..."
    for i in $(seq 1 "$max_wait"); do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -lt 4000 ]; then
            log "GPU memory freed: ${used}MiB"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && log "  GPU memory still: ${used}MiB..."
        sleep 5
    done
    log "WARNING: GPU memory still high after ${max_wait}s wait"
    return 1
}

start_server() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    log "Starting server: config=$config, port=$port"
    kill_port "$port"

    # Configure based on config
    local RADIX="lru"
    local HICACHE_FLAGS=""
    local MAX_TOKENS="--max-total-tokens 60000"
    local MEM_FRAC="0.80"

    if [[ "$config" == "kvflow" ]]; then
        RADIX="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"
        MAX_TOKENS="--max-total-tokens 90000"
        MEM_FRAC="0.75"  # Slightly less to accommodate larger cache
    elif [[ "$config" == "hicache" ]]; then
        RADIX="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through"
    elif [[ "$config" == "hicache90k" ]]; then
        RADIX="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"
        MAX_TOKENS="--max-total-tokens 90000"
        MEM_FRAC="0.75"
    fi

    # Determine TP size based on model
    local TP_SIZE=2
    if [[ "$MODEL_NAME" == "Qwen3-1.7B" ]]; then
        TP_SIZE=1
        MEM_FRAC="0.85"
    fi

    export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size $TP_SIZE \
        --mem-fraction-static "$MEM_FRAC" \
        $MAX_TOKENS \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        --radix-eviction-policy "$RADIX" \
        $HICACHE_FLAGS \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
        --attention-backend flashinfer \
        --sampling-backend flashinfer \
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
}

# =============================================================================
# Benchmark Runner
# =============================================================================

run_benchmark() {
    local config="$1"
    local port="$2"
    shift 2
    local extra_args="$@"
    local bench_log="$LOG_DIR/bench_${config}_${port}.log"

    log "Running $config benchmark on port $port..."

    cd "$SGLANG_ROOT"

    "$PYTHON_BIN" -m benchmark.multi_workflow.bench_kvflow_optimal \
        --config "$config" \
        --host 127.0.0.1 \
        --port "$port" \
        --agents-seed 42 \
        --seed 42 \
        --model "$MODEL_PATH" \
        --output-dir "$RESULT_DIR" \
        $extra_args \
        2>&1 | tee "$bench_log"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "=========================================="
    log "KVFlow Optimal Scenario Benchmark"
    log "=========================================="
    log "Model: $MODEL_NAME"
    log "Model path: $MODEL_PATH"
    log "Log dir: $LOG_DIR"
    log "Result dir: $RESULT_DIR"
    log "SLURM_JOB_ID: $SLURM_JOB_ID"
    log "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
    log "GPUs: $CUDA_VISIBLE_DEVICES"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    log "=========================================="

    # Ports for different configs
    local PORT_HICACHE=30001
    local PORT_KVFLOW=30002
    local PORT_HICACHE90K=30003

    # =============================================================================
    # EXPERIMENT 1: Small scale (4 workflows × 4 agents) - Sanity check
    # =============================================================================
    log "=========================================="
    log "EXPERIMENT 1: Small Scale (4wf × 4ag)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 120 || exit 1
    sleep 5

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 4 --agents-per-workflow 4 \
        --system-prompt-tokens 2048 \
        --group-prefix-tokens 1024 \
        --unique-prefix-tokens 512 \
        --output-len 64 \
        --num-rounds 5 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 120

    # =============================================================================
    # EXPERIMENT 2: Medium scale (8 workflows × 6 agents) - Main comparison
    # =============================================================================
    log "=========================================="
    log "EXPERIMENT 2: Medium Scale (8wf × 6ag)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 120 || exit 1
    sleep 5

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 8 --agents-per-workflow 6 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 64 \
        --num-rounds 8 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 120

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 120 || exit 1
    sleep 5

    local LRU_JSON=$(ls -t "$RESULT_DIR"/kvflow_opt_hicache_*8wf*6ag*.json 2>/dev/null | head -1 || true)

    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 8 --agents-per-workflow 6 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 64 \
        --num-rounds 8 --warmup-rounds 1 \
        ${LRU_JSON:+--baseline-json "$LRU_JSON"}

    stop_server "kvflow" "$PORT_KVFLOW"
    wait_gpu_free 120

    # =============================================================================
    # EXPERIMENT 3: Heavy scale (8 workflows × 8 agents) - Stress test
    # =============================================================================
    log "=========================================="
    log "EXPERIMENT 3: Heavy Scale (8wf × 8ag)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 120 || exit 1
    sleep 5

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 8 --agents-per-workflow 8 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 64 \
        --num-rounds 10 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 120

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 120 || exit 1
    sleep 5

    LRU_JSON=$(ls -t "$RESULT_DIR"/kvflow_opt_hicache_*8wf*8ag*.json 2>/dev/null | head -1 || true)

    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 8 --agents-per-workflow 8 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 64 \
        --num-rounds 10 --warmup-rounds 1 \
        ${LRU_JSON:+--baseline-json "$LRU_JSON"}

    stop_server "kvflow" "$PORT_KVFLOW"

    # =============================================================================
    # Summary
    # =============================================================================
    log "=========================================="
    log "BENCHMARK COMPLETE"
    log "=========================================="
    log "Results: $RESULT_DIR"
    log ""
    log "JSON files:"
    ls -lh "$RESULT_DIR"/*.json 2>/dev/null | awk '{print "  "$NF}'
    log ""
    log "To analyze results:"
    log "  python $LOG_DIR/analyze_results.py"

    # Copy this script to log dir for reproducibility
    cp "$0" "$LOG_DIR/benchmark_script.sh"
}

# Run main
main "$@"
