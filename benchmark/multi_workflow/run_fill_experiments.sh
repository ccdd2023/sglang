#!/bin/bash
# =============================================================================
# Fill-in Missing Experiments
#
# Missing experiments identified by audit (2026-03-27):
#   1. kvflow exp3   — 8wf × 8agent, 2048shr, 1024uni (critical comparison)
#   2. hicache90k    — LRU + write_back + 90k cache (fair comparison vs kvflow)
#   3. kvflow exp2   — 4wf × 5agent, 2048shr, 512uni  (moderate pressure)
#
# Each experiment runs hicache (baseline) and kvflow/hicache90k sequentially,
# waiting for GPU memory to free between runs.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
SGLANG_ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-/home/comp/csgfyu/logs/kvflow-multi-workflow}"
RESULT_DIR="$LOG_DIR/results"
MODEL_PATH="${MODEL_PATH:-/home/comp/csgfyu/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"

PYTHON_BIN="/home/comp/csgfyu/miniconda3/envs/sglang-kvflow/bin/python"

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
    local max_wait="${2:-90}"
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
    local threshold="${1:-4000}"
    local max_wait="${2:-600}"
    log "Waiting for GPU memory to free below ${threshold}MiB..."
    for i in $(seq 1 "$max_wait"); do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -lt "$threshold" ]; then
            log "GPU memory freed: ${used}MiB"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && log "  GPU memory still: ${used}MiB..."
        sleep 5
    done
    log "WARNING: GPU memory still high after ${max_wait}s"
    return 1
}

start_server() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    log "Starting server: config=$config, port=$port"
    kill_port "$port"

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
    elif [[ "$config" == "hicache90k" ]]; then
        # Fair comparison: same cache size (90k) as kvflow, only eviction differs
        RADIX="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"
        MAX_TOKENS="--max-total-tokens 90000"
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
    sleep 15
}

run_benchmark() {
    local config="$1"
    local port="$2"
    local bench_log="$LOG_DIR/bench_${config}_${port}.log"
    shift 2
    local extra_args="$@"

    log "Running $config benchmark on port $port..."
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

# =============================================================================
# MISSING EXPERIMENT 1: kvflow exp3 (8wf × 8agent)
# =============================================================================
run_missing_kvflow_exp3() {
    log "=========================================="
    log "MISSING EXP 1: kvflow 8wf × 8agent (Heavy Pressure)"
    log "=========================================="
    # The hicache exp3 result already exists:
    #   mwf_hicache_64agents_2048shr_1024uni_5rounds_8wf.json
    # We run kvflow with identical parameters for direct comparison.

    local PORT=30032

    wait_gpu_free 120 || true
    start_server "kvflow" "$PORT"
    wait_ready "$PORT" 90 || return 1
    sleep 5

    local BASELINE="$(ls -t "$RESULT_DIR"/mwf_hicache_*64agents*8wf*.json 2>/dev/null | head -1 || true)"
    local BASELINE_ARG=""
    [ -n "$BASELINE" ] && BASELINE_ARG="--baseline-json $BASELINE"

    run_benchmark "kvflow" "$PORT" \
        --num-workflows 8 --agents-per-workflow 8 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_ARG

    stop_server "kvflow" "$PORT"
    log "MISSING EXP 1 COMPLETE"
}

# =============================================================================
# MISSING EXPERIMENT 2: hicache90k (LRU + write_back + 90k cache)
# =============================================================================
run_missing_hicache90k() {
    log "=========================================="
    log "MISSING EXP 2: hicache90k 8wf × 8agent (Fair Comparison)"
    log "=========================================="
    # hicache90k = hicache config BUT with same cache size (90k) and write_back
    # as kvflow. This isolates the effect of Priority vs LRU eviction.

    local PORT=30041

    wait_gpu_free 120 || true
    start_server "hicache90k" "$PORT"
    wait_ready "$PORT" 90 || return 1
    sleep 5

    # Baseline: hicache standard exp3 result
    local BASELINE="$(ls -t "$RESULT_DIR"/mwf_hicache_*64agents*8wf*.json 2>/dev/null | head -1 || true)"
    local BASELINE_ARG=""
    [ -n "$BASELINE" ] && BASELINE_ARG="--baseline-json $BASELINE"

    run_benchmark "hicache90k" "$PORT" \
        --num-workflows 8 --agents-per-workflow 8 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_ARG

    stop_server "hicache90k" "$PORT"
    log "MISSING EXP 2 COMPLETE"
}

# =============================================================================
# MISSING EXPERIMENT 3: kvflow exp2 (4wf × 5agent)
# =============================================================================
run_missing_kvflow_exp2() {
    log "=========================================="
    log "MISSING EXP 3: kvflow 4wf × 5agent (Moderate Pressure)"
    log "=========================================="
    # hicache exp2 result already exists (4wf × 5agent, 2048shr, 512uni)

    local PORT=30022

    wait_gpu_free 120 || true
    start_server "kvflow" "$PORT"
    wait_ready "$PORT" 90 || return 1
    sleep 5

    local BASELINE="$(ls -t "$RESULT_DIR"/mwf_hicache_*4wf*.json 2>/dev/null | head -1 || true)"
    local BASELINE_ARG=""
    [ -n "$BASELINE" ] && BASELINE_ARG="--baseline-json $BASELINE"

    run_benchmark "kvflow" "$PORT" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_ARG

    stop_server "kvflow" "$PORT"
    log "MISSING EXP 3 COMPLETE"
}

# =============================================================================
# MISSING EXPERIMENT 4: hicache exp2 repeat (fair 4wf × 5agent, 2048shr, 1024uni)
# Currently exp2 uses unique=512, but the 64agent exp uses unique=1024.
# For a consistent scaling study, we need exp2 with unique=1024 too.
# =============================================================================
run_missing_hicache_exp2_heavy() {
    log "=========================================="
    log "MISSING EXP 4: hicache 4wf × 5agent, 2048shr, 1024uni (moderate-heavy)"
    log "=========================================="
    # Same pressure as 64agent exp but fewer workflows.
    # 4wf × 5agent × 3072 = 61,440 tokens/round (same as 20agent exp)
    # But with 2048+1024=3072 per agent (same as 64agent config)

    local PORT=30043

    wait_gpu_free 120 || true
    start_server "hicache" "$PORT"
    wait_ready "$PORT" 90 || return 1
    sleep 5

    run_benchmark "hicache" "$PORT" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1

    stop_server "hicache" "$PORT"
    log "MISSING EXP 4 COMPLETE"
}

# =============================================================================
usage() {
    cat <<'EOF'
Usage: ./run_fill_experiments.sh [all|miss1|miss2|miss3|miss4]

Targets:
    all     Run all 4 missing experiments (default)
    miss1   kvflow exp3 (8wf × 8agent, critical comparison)
    miss2   hicache90k (LRU + write_back + 90k, fair eviction comparison)
    miss3   kvflow exp2 (4wf × 5agent, moderate pressure)
    miss4   hicache 4wf × 5agent, 2048shr, 1024uni (scaling study)
EOF
}

TARGET="${1:-all}"
case "$TARGET" in
    all)   run_missing_kvflow_exp3 && run_missing_hicache90k && run_missing_kvflow_exp2 && run_missing_hicache_exp2_heavy ;;
    miss1) run_missing_kvflow_exp3 ;;
    miss2) run_missing_hicache90k ;;
    miss3) run_missing_kvflow_exp2 ;;
    miss4) run_missing_hicache_exp2_heavy ;;
    help|--help|-h) usage; exit 0 ;;
    *)     usage; exit 1 ;;
esac

log "=========================================="
log "ALL MISSING EXPERIMENTS COMPLETE"
log "=========================================="
log "Results: $RESULT_DIR"
ls -lh "$RESULT_DIR"/mwf_*.json 2>/dev/null | awk '{print "  "$NF}'
log "Run analysis: python $LOG_DIR/analyze_results.py"
