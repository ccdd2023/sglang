#!/bin/bash
# =============================================================================
# KVFlow Local Test Script for RTX 4090
#
# This script runs a local KVFlow benchmark on a single RTX 4090 (24GB)
# using Qwen2.5-3B-Instruct model.
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/kvflow-local}"
RESULT_DIR="$LOG_DIR/results"

# Model - use HF model ID directly (sglang will download if needed)
MODEL_DIR="Qwen/Qwen2.5-3B-Instruct"

# Server configuration
PORT_HICACHE=30001
PORT_KVFLOW=30002

mkdir -p "$LOG_DIR" "$RESULT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/experiment.log"; }

# =============================================================================
# Environment Check
# =============================================================================

check_environment() {
    log "=========================================="
    log "Environment Check"
    log "=========================================="
    
    # GPU
    log "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
    
    # Python
    log "Python: $(python3 --version)"
    
    # PyTorch
    log "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
    
    # Transformers
    log "Transformers: $(python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not installed')"
    
    # Check if sglang is installed
    if python3 -c "import sglang" 2>/dev/null; then
        log "SGLang: installed ($(python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo 'ok'))"
    else
        log "SGLang: NOT installed"
        log "Installing sglang..."
        cd "$PROJECT_DIR/sglang-kvflow/python"
        pip install -e . -q
    fi
    
    log "=========================================="
}

# =============================================================================
# Model Download
# =============================================================================

download_model() {
    # Check if model directory exists and has valid config.json
    if [ -f "$MODEL_DIR/config.json" ]; then
        log "Model already exists at $MODEL_DIR"
        # Verify config.json has model_type
        if grep -q "model_type" "$MODEL_DIR/config.json" 2>/dev/null; then
            log "Model config.json is valid"
            return 0
        else
            log "Model config.json is incomplete/corrupted, re-downloading..."
            rm -rf "$MODEL_DIR"
        fi
    elif [ -d "$MODEL_DIR" ]; then
        log "Model directory exists but config.json is missing, re-downloading..."
        rm -rf "$MODEL_DIR"
    fi
    
    log "Downloading model Qwen/Qwen2.5-3B-Instruct..."
    python3 -c "
from huggingface_hub import snapshot_download
import os

os.makedirs('$MODEL_DIR', exist_ok=True)
snapshot_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('Download complete!')
"
}

# =============================================================================
# Server Management
# =============================================================================

kill_port() {
    local p="$1"
    if pid=$(lsof -ti :"$p" 2>/dev/null); then
        log "Killing port $p (PID=$pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 5
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
    return 1
}

wait_gpu_free() {
    local max_wait="${1:-120}"
    log "Waiting for GPU memory to free..."
    for i in $(seq 1 "$max_wait"); do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -lt 8000 ]; then
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

    log "=========================================="
    log "Starting server: config=$config, port=$port"
    log "=========================================="
    kill_port "$port"

    # Configure based on config
    local RADIX="lru"
    local HICACHE_FLAGS=""
    local MAX_TOKENS="--max-total-tokens 40000"
    local MEM_FRAC="0.75"

    if [[ "$config" == "kvflow" ]]; then
        RADIX="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_back --enable-hicache-prefetch"
        MAX_TOKENS="--max-total-tokens 60000"
        MEM_FRAC="0.70"
    elif [[ "$config" == "hicache" ]]; then
        RADIX="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through"
    fi

    export PYTHONPATH="$PROJECT_DIR/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"

    python3 -m sglang.launch_server \
        --model-path "$MODEL_DIR" \
        --port "$port" \
        --tp-size 1 \
        --mem-fraction-static "$MEM_FRAC" \
        $MAX_TOKENS \
        --chunked-prefill-size 2048 \
        --max-prefill-tokens 4096 \
        --radix-eviction-policy "$RADIX" \
        $HICACHE_FLAGS \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
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
    sleep 10
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

    cd "$PROJECT_DIR/sglang-kvflow"

    python3 -m benchmark.multi_workflow.bench_kvflow_optimal \
        --config "$config" \
        --host 127.0.0.1 \
        --port "$port" \
        --agents-seed 42 \
        --seed 42 \
        --model "$MODEL_DIR" \
        --output-dir "$RESULT_DIR" \
        $extra_args \
        2>&1 | tee "$bench_log"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "=========================================="
    log "KVFlow Local Benchmark (RTX 4090)"
    log "=========================================="
    
    # Environment
    check_environment
    
    # Download model if needed
    download_model
    
    # Show GPU memory
    log "GPU Memory before experiments:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

    # =============================================================================
    # EXPERIMENT 1: Light pressure (4 workflows × 4 agents)
    # This should show minimal difference as cache pressure is low
    # =============================================================================
    log "=========================================="
    log "EXPERIMENT 1: Light Pressure (4wf × 4ag)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 180 || exit 1
    sleep 5

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 4 --agents-per-workflow 4 \
        --system-prompt-tokens 2048 \
        --group-prefix-tokens 1024 \
        --unique-prefix-tokens 512 \
        --output-len 64 \
        --num-rounds 5 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 60

    # =============================================================================
    # EXPERIMENT 2: Medium pressure (8 workflows × 6 agents)
    # This is where we should start seeing some differences
    # =============================================================================
    log "=========================================="
    log "EXPERIMENT 2: Medium Pressure (8wf × 6ag)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 180 || exit 1
    sleep 5

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 8 --agents-per-workflow 6 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 64 \
        --num-rounds 8 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 60

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 180 || exit 1
    sleep 5

    # Find baseline for comparison
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
    wait_gpu_free 60

    # =============================================================================
    # EXPERIMENT 3: Heavy pressure (8 workflows × 8 agents)
    # This is where KVFlow should show its advantage
    # =============================================================================
    log "=========================================="
    log "EXPERIMENT 3: Heavy Pressure (8wf × 8ag)"
    log "=========================================="

    start_server "hicache" "$PORT_HICACHE"
    wait_ready "$PORT_HICACHE" 180 || exit 1
    sleep 5

    run_benchmark "hicache" "$PORT_HICACHE" \
        --num-workflows 8 --agents-per-workflow 8 \
        --system-prompt-tokens 4096 \
        --group-prefix-tokens 2048 \
        --unique-prefix-tokens 1024 \
        --output-len 64 \
        --num-rounds 10 --warmup-rounds 1

    stop_server "hicache" "$PORT_HICACHE"
    wait_gpu_free 60

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 180 || exit 1
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
    log "GPU Memory after experiments:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    
    # Analyze results
    log ""
    log "To analyze results, run:"
    log "  python $PROJECT_DIR/sglang-kvflow/benchmark/multi_workflow/analyze_kvflow_optimal.py --results-dir $RESULT_DIR"
}

# Run main
main "$@"
