#!/bin/bash
# =============================================================================
# KVFlow Benchmark Pipeline
#
# Compares LRU+HiCache vs Priority+HiCache+Prefetch for multi-agent
# code generation workloads.
#
# Usage:
#   ./run_pipeline.sh              # Run all experiments
#   ./run_pipeline.sh exp1          # Run only exp1 (single workflow, light pressure)
#   ./run_pipeline.sh exp2          # Run only exp2 (4 workflows, moderate pressure)
#   ./run_pipeline.sh exp3          # Run only exp3 (8 workflows, heavy pressure)
#   ./run_pipeline.sh quick         # Fast smoke test (1 round, 1 workflow)
#
# After running:
#   python /home/comp/csgfyu/logs/kvflow-multi-workflow/analyze_results.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
SGLANG_ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # sglang-kvflow root
LOG_DIR="${LOG_DIR:-/home/comp/csgfyu/logs/kvflow-multi-workflow}"
RESULT_DIR="$LOG_DIR/results"
MODEL_PATH="${MODEL_PATH:-/home/comp/csgfyu/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"

# Use absolute path to Python binary in the target conda env
# Override with PYTHON_BIN env var if needed
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="/home/comp/csgfyu/miniconda3/envs/sglang-kvflow/bin/python"
fi

mkdir -p "$LOG_DIR" "$RESULT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

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
    local max_wait="${2:-60}"
    log "Waiting for server on port $port..."
    for i in $(seq 1 "$max_wait"); do
        if curl -sf --max-time 2 "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            log "Server on port $port ready after $((i*5))s!"
            return 0
        fi
        [ $((i % 6)) -eq 0 ] && log "  ...still waiting, $((i*5))s..."
        sleep 5
    done
    log "ERROR: Server on port $port did not start"
    tail -5 "$LOG_DIR"/server_*.log 2>/dev/null || true
    return 1
}

wait_gpu_free() {
    # Wait for GPU memory to drop below threshold before starting next server.
    # With HiCache enabled, host memory may take a long time to release.
    # We wait until GPU memory is below 4GB (enough for one HiCache server).
    local max_wait="${1:-600}"
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

# -----------------------------------------------------------------------
# Server launch configurations
#
# hicache  — LRU eviction + HiCache write_through
#             This is the "baseline" for comparison.
#
# kvflow    — Priority eviction + HiCache write_back + proactive prefetch
#             This is the KVFlow experimental config.
# -----------------------------------------------------------------------

start_server() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    log "Starting server: config=$config, port=$port"
    kill_port "$port"

    # Build server command with absolute path to python binary
    # No cd needed - we use absolute paths everywhere

    local RADIX="lru"
    local HICACHE_FLAGS=""
    local MAX_TOKENS="--max-total-tokens 60000"
    local MEM_FRAC="0.85"

    if [[ "$config" == "kvflow" ]]; then
        RADIX="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"
        MAX_TOKENS="--max-total-tokens 90000"
    elif [[ "$config" == "hicache" ]]; then
        RADIX="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through"
    fi

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size 2 \
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
    # Wait longer for GPU memory to be freed after HiCache server exits
    sleep 15
}

run_benchmark() {
    local config="$1"
    local port="$2"
    shift 2
    local extra_args="$@"
    local bench_log="$LOG_DIR/bench_${config}_${port}.log"

    log "Running $config benchmark on port $port..."

    # cd to sglang-kvflow root so the benchmark module is importable
    # PYTHON_BIN already points to the right python in the conda env (no activate needed)
    cd "$SGLANG_ROOT_DIR"

    "$PYTHON_BIN" -m benchmark.multi_workflow.bench_multi_workflow \
        --config "$config" \
        --host 127.0.0.1 \
        --port "$port" \
        --agents-seed 42 \
        --seed 42 \
        --output-dir "$RESULT_DIR" \
        $extra_args \
        2>&1 | tee "$bench_log"
}

# -----------------------------------------------------------------------
# Experiment definitions
# -----------------------------------------------------------------------

run_exp_quick() {
    log "=========================================="
    log "QUICK SMOKE TEST"
    log "=========================================="

    local PORT_LRU=30001
    local PORT_KVFLOW=30002

    # hicache (LRU baseline)
    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 90 || return 1
    sleep 5
    run_benchmark "hicache" "$PORT_LRU" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 3 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 5

    # kvflow
    wait_gpu_free 120
    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 90 || return 1
    sleep 5
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 3 --warmup-rounds 1 \
        --baseline-json "$(ls -t "$RESULT_DIR"/mwf_hicache_*.json 2>/dev/null | head -1)"
    stop_server "kvflow" "$PORT_KVFLOW"

    log "QUICK TEST COMPLETE"
}

run_exp1_single() {
    log "=========================================="
    log "EXPERIMENT 1: Single Workflow (Light Pressure)"
    log "=========================================="
    # 1 workflow × 4 agents × 2560 = 10,240 tokens/round
    # 60k cache fits everything; both configs should perform well.
    # This is a sanity check.

    local PORT_LRU=30011
    local PORT_KVFLOW=30012

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 90 || return 1
    sleep 5
    run_benchmark "hicache" "$PORT_LRU" \
        --num-workflows 1 --agents-per-workflow 4 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 5

    wait_gpu_free 120
    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 90 || return 1
    sleep 5
    local LRU_JSON=$(ls -t "$RESULT_DIR"/mwf_hicache_*1wf*.json 2>/dev/null | head -1 || true)
    local BASELINE_ARG=""
    [ -n "$LRU_JSON" ] && BASELINE_ARG="--baseline-json $LRU_JSON"
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 1 --agents-per-workflow 4 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_ARG
    stop_server "kvflow" "$PORT_KVFLOW"

    log "EXP 1 COMPLETE"
}

run_exp2_multi() {
    log "=========================================="
    log "EXPERIMENT 2: 4 Concurrent Workflows (Moderate Pressure)"
    log "=========================================="
    # 4 workflows × 5 agents × 2560 = 51,200 tokens/round
    # 60k cache: ~120% of one round; pressure builds up.
    # Key question: does Priority preserve shared prefixes better?

    local PORT_LRU=30021
    local PORT_KVFLOW=30022

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 90 || return 1
    sleep 5
    run_benchmark "hicache" "$PORT_LRU" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 5

    wait_gpu_free 120
    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 90 || return 1
    sleep 5
    local LRU_JSON=$(ls -t "$RESULT_DIR"/mwf_hicache_*4wf*.json 2>/dev/null | head -1 || true)
    local BASELINE_ARG=""
    [ -n "$LRU_JSON" ] && BASELINE_ARG="--baseline-json $LRU_JSON"
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_ARG
    stop_server "kvflow" "$PORT_KVFLOW"

    log "EXP 2 COMPLETE"
}

run_exp3_heavy() {
    log "=========================================="
    log "EXPERIMENT 3: 8 Concurrent Workflows (Heavy Pressure)"
    log "=========================================="
    # 8 workflows × 8 agents × 3072 = 196,608 tokens/round
    # 60k cache: ~220% pressure → severe eviction
    # Key question: does Priority preserve shared prefixes better than LRU?
    # To isolate the effect, we use a "no-eviction" warmup baseline.

    local PORT_LRU=30031
    local PORT_KVFLOW=30032

    # Run both servers in PARALLEL on different ports so they don't compete.
    # kvflow uses larger max_total_tokens (90k) vs hicache (60k).
    # Both use same GPU memory fraction so they can coexist (though throughput halves).

    # hicache server
    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 90 || return 1
    sleep 3

    # kvflow server (on different port, same GPU)
    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 90 || return 1
    sleep 3

    # Run hicache first
    run_benchmark "hicache" "$PORT_LRU" \
        --num-workflows 8 --agents-per-workflow 8 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1

    # Run kvflow second (concurrent with hicache server still running)
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 8 --agents-per-workflow 8 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        --baseline-json "$(ls -t "$RESULT_DIR"/mwf_hicache_*64agents*8wf*.json 2>/dev/null | head -1)"

    stop_server "hicache" "$PORT_LRU"
    stop_server "kvflow" "$PORT_KVFLOW"

    log "EXP 3 COMPLETE"
}

print_summary() {
    log "=========================================="
    log "RESULTS"
    log "=========================================="
    log ""
    log "Results: $RESULT_DIR"
    log "Logs:    $LOG_DIR"
    log ""
    log "JSON files:"
    ls -lh "$RESULT_DIR"/mwf_*.json 2>/dev/null | awk '{print "  "$NF}'
    log ""
    log "Analyze results:"
    log "  python $LOG_DIR/analyze_results.py"
}

usage() {
    cat <<'EOF'
Usage: ./run_pipeline.sh [exp|all|quick]

Arguments:
    exp1    Single workflow, light pressure (sanity check)
    exp2    4 concurrent workflows, moderate pressure (main comparison)
    exp3    8 concurrent workflows, heavy pressure (stress test)
    quick   Fast smoke test
    all     Run all experiments (default)
    help    Show this message

Configs compared:
    hicache  — LRU eviction + HiCache write_through (60k cache)
    kvflow   — Priority eviction + HiCache write_back + prefetch (90k cache)
EOF
}

PYTHON_BIN="${PYTHON_BIN:-$(which python3 2>/dev/null || echo python)}"

TARGET="${1:-all}"
case "$TARGET" in
    exp1)  run_exp1_single ;;
    exp2)  run_exp2_multi ;;
    exp3)  run_exp3_heavy ;;
    quick) run_exp_quick ;;
    all)   run_exp1_single && run_exp2_multi && run_exp3_heavy ;;
    help|--help|-h) usage; exit 0 ;;
    *)     usage; exit 1 ;;
esac

print_summary
