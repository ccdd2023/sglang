#!/bin/bash
# =============================================================================
# KVFlow Optimal Scenario Benchmark - RTX 4080 Optimized
# 
# Optimized for 4x RTX 4080 (16GB each) with Qwen3-8B
# =============================================================================

#SBATCH --job-name=kvflow-8b-rtx
#SBATCH --output=/home/comp/25480812/logs/kvflow-8b/slurm-%j.out
#SBATCH --error=/home/comp/25480812/logs/kvflow-8b/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

# Load GCC 11 for C++17 support
module load gcc/11.2.0 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-/home/comp/25480812/logs/kvflow-8b}"
RESULT_DIR="$LOG_DIR/results"

MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
CONDA_ENV="/home/comp/25480812/.conda/envs/sglang-kvflow"
PYTHON_BIN="$CONDA_ENV/bin/python"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/slurm.log"; }

kill_port() {
    local p="$1"
    if pid=$(lsof -ti :"$p" 2>/dev/null); then
        log "Killing port $p..."
        kill "$pid" 2>/dev/null || true
        sleep 5
    fi
}

wait_ready() {
    local port="$1"
    local max_wait="${2:-180}"
    log "Waiting for server on port $port..."
    # Bypass proxy for localhost
    for i in $(seq 1 "$max_wait"); do
        if curl -sf --max-time 3 --noproxy '*' "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            log "Server ready after $((i*5))s!"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && log "  Still waiting... ${i}s"
        sleep 5
    done
    log "ERROR: Server did not start"
    tail -30 "$LOG_DIR"/server_*.log 2>/dev/null || true
    return 1
}

wait_gpu_free() {
    local max_wait="${1:-180}"
    log "Waiting for GPU memory..."
    for i in $(seq 1 "$max_wait"); do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -lt 6000 ]; then
            log "GPU freed: ${used}MiB"
            return 0
        fi
        [ $((i % 6)) -eq 0 ] && log "  GPU memory: ${used}MiB"
        sleep 10
    done
    log "WARNING: GPU still in use"
    return 1
}

start_server() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    log "Starting $config server on port $port..."
    kill_port "$port"

    local RADIX="lru"
    local HICACHE_FLAGS=""
    local MAX_TOKENS="--max-total-tokens 60000"
    local MEM_FRAC="0.85"
    local TP_SIZE=1

    if [[ "$config" == "kvflow" ]]; then
        RADIX="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_back --enable-hicache-prefetch"
        MAX_TOKENS="--max-total-tokens 80000"
    elif [[ "$config" == "hicache" ]]; then
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through"
    fi

    export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"

    # Reinstall sgl-kernel for current GPU architecture
    log "Reinstalling sgl-kernel for A100..."
    "$PYTHON_BIN" -m pip install --quiet --force-reinstall sglang-kernel 2>&1 | tail -5

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size $TP_SIZE \
        --mem-fraction-static "$MEM_FRAC" \
        $MAX_TOKENS \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 4096 \
        --radix-eviction-policy "$RADIX" \
        $HICACHE_FLAGS \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
        --attention-backend flashinfer \
        --sampling-backend flashinfer \
        --skip-server-warmup \
        --log-level info \
        > "$log_file" 2>&1 &

    log "Server PID=$! started"
    echo "$!" > "$LOG_DIR/.server_${config}_${port}.pid"
}

stop_server() {
    local config="$1"
    local port="$2"
    kill_port "$port"
    rm -f "$LOG_DIR/.server_${config}_${port}.pid"
    sleep 10
}

run_benchmark() {
    local config="$1"
    local port="$2"
    shift 2
    local extra_args="$@"
    local bench_log="$LOG_DIR/bench_${config}_${port}.log"

    log "Running $config benchmark..."

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
    log "Model: Qwen3-8B on 4x RTX 4080"
    log "=========================================="
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

    local PORT_HICACHE=30001
    local PORT_KVFLOW=30002

    # =========================================================================
    # EXPERIMENT 1: 4wf × 4ag - Quick validation
    # =========================================================================
    log "=========================================="
    log "EXP 1: 4wf × 4ag (Light)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 180 || exit 1
    sleep 3

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 4 --agents-per-workflow 4 \
        --system-prompt-tokens 2048 \
        --group-prefix-tokens 1024 \
        --unique-prefix-tokens 512 \
        --output-len 32 \
        --num-rounds 3 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 120

    # =========================================================================
    # EXPERIMENT 2: 8wf × 6ag - Medium pressure
    # =========================================================================
    log "=========================================="
    log "EXP 2: 8wf × 6ag (Medium)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 180 || exit 1
    sleep 3

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 8 --agents-per-workflow 6 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 32 \
        --num-rounds 5 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 120

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 180 || exit 1
    sleep 3

    local LRU_JSON=$(ls -t "$RESULT_DIR"/kvflow_opt_hicache_*8wf*6ag*.json 2>/dev/null | head -1 || true)

    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 8 --agents-per-workflow 6 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        ${LRU_JSON:+--baseline-json "$LRU_JSON"}

    stop_server "kvflow" "$PORT_KVFLOW"
    wait_gpu_free 120

    # =========================================================================
    # EXPERIMENT 3: 8wf × 8ag - Heavy pressure
    # =========================================================================
    log "=========================================="
    log "EXP 3: 8wf × 8ag (Heavy)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 180 || exit 1
    sleep 3

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 8 --agents-per-workflow 8 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 32 \
        --num-rounds 5 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 120

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 180 || exit 1
    sleep 3

    LRU_JSON=$(ls -t "$RESULT_DIR"/kvflow_opt_hicache_*8wf*8ag*.json 2>/dev/null | head -1 || true)

    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 8 --agents-per-workflow 8 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        ${LRU_JSON:+--baseline-json "$LRU_JSON"}

    stop_server "kvflow" "$PORT_KVFLOW"

    # =========================================================================
    # Summary
    # =========================================================================
    log "=========================================="
    log "BENCHMARK COMPLETE"
    log "=========================================="
    log "Results: $RESULT_DIR"
    ls -lh "$RESULT_DIR"/*.json 2>/dev/null || echo "No results yet"
}

main "$@"
