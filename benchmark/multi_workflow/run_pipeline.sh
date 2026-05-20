#!/bin/bash
# =============================================================================
# KVFlow Benchmark Pipeline
#
# Compares LRU+HiCache vs Priority+HiCache for multi-agent
# code generation workloads.
#
# Experiment types:
#   Fair linear:  exp1-fair, exp2-fair
#   DAG workflow: exp1-dag, exp2-dag
#   Ablation:     ablation-8wf, ablation-16wf, ablation-32wf
#
# All output written to files -- NO stdout echo/print (NFS I/O safety)
# =============================================================================

set -uo pipefail  # Use -uo instead of -euo to allow error handling

export PYTHONPATH="${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
SGLANG_ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-/home/gfy/CodeMAS_Project/logs/kvflow-multi-workflow}"
RESULT_DIR="$LOG_DIR/results"
MODEL_PATH="${MODEL_PATH:-/home/gfy/models/Qwen2.5-3B-Instruct}"

CONDA_ENV_PATH="/home/gfy/.conda/envs/sglang-kvflow"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# All output goes to a local pipeline log (no NFS stdout)
PIPELINE_LOG="$LOG_DIR/pipeline.log"
exec >> "$PIPELINE_LOG" 2>&1

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# Python binary resolution
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="/home/gfy/.conda/envs/sglang-kvflow/bin/python"
fi
export PYTHONPATH="/home/gfy/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_DISABLE_FLEX_ATTENTION=1

log "PYTHON_BIN: $PYTHON_BIN"
log "PYTHONPATH: ${PYTHONPATH:-<not set>}"
log "MODEL_PATH: $MODEL_PATH"

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
        if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            log "Server on port $port ready after $((i*5))s!"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && log "  ...still waiting, $((i*5))s..."
        sleep 5
    done
    log "ERROR: Server on port $port did not start"
    tail -5 "$LOG_DIR"/server_*.log 2>/dev/null || true
    return 1
}

# ---------------------------------------------------------------------------
# Server launch configurations
#
# Fair comparison design:
#   hicache90k -- LRU eviction + HiCache (ratio=2.5, write_back, 90k tokens)
#   kvflow     -- Priority eviction + HiCache (ratio=2.5, write_back, 90k tokens)
# Both servers use IDENTICAL HiCache params -- only eviction (LRU vs Priority) differs.
# ---------------------------------------------------------------------------

start_server() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    log "Starting server: config=$config, port=$port"
    kill_port "$port"

    : "${CUDA_HOME:=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda)))}"
    export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

    local RADIX="lru"
    local HICACHE_FLAGS=""

    if [[ "$config" == "kvflow" ]]; then
        RADIX="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back"
    elif [[ "$config" == "kvflow_tiered" ]]; then
        RADIX="tiered"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"
    elif [[ "$config" == "hicache" || "$config" == "hicache90k" ]]; then
        RADIX="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back"
    fi

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size 1 \
        --mem-fraction-static 0.85 \
        --max-total-tokens 90000 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        --radix-eviction-policy "$RADIX" \
        $HICACHE_FLAGS \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
        --disable-pie \
        --skip-server-warmup \
        --log-level info \
        > "$log_file" 2>&1 &

    log "Server PID=$! (log: $log_file)"
    printf '%s\n' "$!" > "$LOG_DIR/.server_${config}_${port}.pid"
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

run_benchmark() {
    local config="$1"
    local port="$2"
    shift 2
    local extra_args="$@"
    local bench_log="$LOG_DIR/bench_${config}_${port}.log"

    log "Running $config benchmark on port $port..."
    cd "$SGLANG_ROOT_DIR"

    local LOCAL_SITE="$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo "")"
    local attempt=1
    local max_attempts=3

    log "Syntax check bench_multi_workflow.py..."
    if ! "$PYTHON_BIN" -B -m py_compile "$SGLANG_ROOT_DIR/benchmark/multi_workflow/bench_multi_workflow.py" 2>&1; then
        log "ERROR: bench_multi_workflow.py has syntax errors!"
        return 1
    fi

    while true; do
        if PYTHONNOUSERSITE=1 \
            PYTHONPATH="$LOCAL_SITE${LOCAL_SITE:+:}$SGLANG_ROOT_DIR" \
            "$PYTHON_BIN" -B "$SGLANG_ROOT_DIR/benchmark/multi_workflow/bench_multi_workflow.py" \
            --config "$config" \
            --host 127.0.0.1 \
            --port "$port" \
            --agents-seed 42 \
            --seed 42 \
            --output-dir "$RESULT_DIR" \
            $extra_args \
            >> "$bench_log" 2>&1; then
            log "$config benchmark on port $port succeeded on attempt $attempt."
            break
        else
            local exit_code=$?
            if [[ $attempt -ge $max_attempts ]]; then
                log "ERROR: $config benchmark FAILED after $max_attempts attempts (exit=$exit_code)"
                return 1
            fi
            log "WARNING: $config benchmark failed (attempt $attempt/$max_attempts), retrying in 30s..."
            sleep 30
            attempt=$((attempt + 1))
        fi
    done
}

print_summary() {
    log "=========================================="
    log "RESULTS"
    log "=========================================="
    log "Results: $RESULT_DIR"
    log "Logs:    $LOG_DIR"
    log "JSON files:"
    ls -lh "$RESULT_DIR"/mwf_*.json 2>/dev/null | awk '{print "  "$NF}'
}

# =============================================================================
# Experiment definitions
# =============================================================================

run_exp1_fair() {
    # FAIR EXP1: 1 Workflow x 10 Agents, Low Pressure
    log "=========================================="
    log "FAIR EXP1: 1wf x 10ag (low pressure ~0.5x)"
    log "Config: Fixed=4096, Dynamic=32, Output=32, 5 rounds + 1 warmup"
    log "Fair: same HiCache (90k, write_back), only eviction differs"
    log "=========================================="

    local PORT_LRU=30200
    local PORT_KVFLOW=30201

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 360 || return 1
    sleep 3
    run_benchmark "hicache90k" "$PORT_LRU" \
        --num-workflows 1 --agents-per-workflow 10 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 60
    sync

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 360 || return 1
    sleep 3
    local BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_hicache90k_*1wf*10ag*.json 2>/dev/null | head -1)"
    [ -n "$BASELINE_JSON" ] && BASELINE_JSON="--baseline-json $BASELINE_JSON" || BASELINE_JSON=""
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 1 --agents-per-workflow 10 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_JSON
    stop_server "kvflow" "$PORT_KVFLOW"

    log "FAIR EXP1 COMPLETE"
}

run_exp2_fair() {
    # FAIR EXP2: 4 Workflows x 10 Agents, Medium Pressure
    log "=========================================="
    log "FAIR EXP2: 4wf x 10ag (medium pressure ~2x)"
    log "Config: Fixed=4096, Dynamic=32, Output=32, 5 rounds + 1 warmup"
    log "Fair: same HiCache (90k, write_back), only eviction differs"
    log "=========================================="

    local PORT_LRU=30210
    local PORT_KVFLOW=30211

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 360 || return 1
    sleep 3
    run_benchmark "hicache90k" "$PORT_LRU" \
        --num-workflows 4 --agents-per-workflow 10 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 60
    sync

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 360 || return 1
    sleep 3
    local BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_hicache90k_*4wf*10ag*.json 2>/dev/null | head -1)"
    [ -n "$BASELINE_JSON" ] && BASELINE_JSON="--baseline-json $BASELINE_JSON" || BASELINE_JSON=""
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 4 --agents-per-workflow 10 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_JSON
    stop_server "kvflow" "$PORT_KVFLOW"

    log "FAIR EXP2 COMPLETE"
}

# =============================================================================
# DAG Workflow Experiments
# DAG structure: PLANNER -> [ARCHITECT, REVIEWER] -> IMPLEMENTER -> TESTER
# Key difference from linear:汇聚节点等待多个上游，Priority 驱逐更有效
# =============================================================================
run_exp1_dag() {
    log "=========================================="
    log "DAG EXP1: 4 parallel DAG workflows"
    log "DAG: PLANNER -> [ARCHITECT, REVIEWER] -> IMPLEMENTER -> TESTER"
    log "Fair: same HiCache (90k, write_back), only eviction differs"
    log "=========================================="

    local PORT_LRU=30300
    local PORT_KVFLOW=30301
    local DAG_CONFIG="$BENCHMARK_DIR/configs/dag_parallel_dev.json"

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 360 || return 1
    sleep 3
    run_benchmark "hicache90k" "$PORT_LRU" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows 4 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 60
    sync

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 360 || return 1
    sleep 3
    local BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_hicache90k_*_dag*_4wf.json 2>/dev/null | head -1)"
    [ -n "$BASELINE_JSON" ] && BASELINE_JSON="--baseline-json $BASELINE_JSON" || BASELINE_JSON=""
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows 4 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_JSON
    stop_server "kvflow" "$PORT_KVFLOW"

    log "DAG EXP1 COMPLETE"
}

run_exp2_dag() {
    log "=========================================="
    log "DAG EXP2: 16 parallel DAG workflows (high pressure)"
    log "DAG: PLANNER -> [ARCHITECT, REVIEWER] -> IMPLEMENTER -> TESTER"
    log "Fair: same HiCache (90k, write_back), only eviction differs"
    log "=========================================="

    local PORT_LRU=30310
    local PORT_KVFLOW=30311
    local DAG_CONFIG="$BENCHMARK_DIR/configs/dag_parallel_dev.json"

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 360 || return 1
    sleep 3
    run_benchmark "hicache90k" "$PORT_LRU" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows 16 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 60
    sync

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 360 || return 1
    sleep 3
    local BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_hicache90k_*_dag*_16wf.json 2>/dev/null | head -1)"
    [ -n "$BASELINE_JSON" ] && BASELINE_JSON="--baseline-json $BASELINE_JSON" || BASELINE_JSON=""
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows 16 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_JSON
    stop_server "kvflow" "$PORT_KVFLOW"

    log "DAG EXP2 COMPLETE"
}

# =============================================================================
# Ablation experiment
# =============================================================================
start_server_ablation() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_ablation_${config}_${port}.log"

    log "Starting ablation server: config=$config, port=$port"
    kill_port "$port"

    : "${CUDA_HOME:=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda)))}"
    export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

    local RADIX="lru"
    local HICACHE_FLAGS=""

    case "$config" in
        lru_nocache)       RADIX="lru";       HICACHE_FLAGS="" ;;
        lru_wb_only)       RADIX="lru";       HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back" ;;
        lru_wb_pf)         RADIX="lru";       HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch" ;;
        priority_wb_only)  RADIX="priority";  HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back" ;;
        kvflow)            RADIX="priority";  HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch" ;;
        tiered)             RADIX="tiered";    HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch" ;;
        tiered_nopf)       RADIX="tiered";    HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back" ;;
    esac

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size 1 \
        --mem-fraction-static 0.85 \
        --max-total-tokens 90000 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        --radix-eviction-policy "$RADIX" \
        $HICACHE_FLAGS \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
        --disable-pie \
        --skip-server-warmup \
        --log-level info \
        > "$log_file" 2>&1 &

    log "Server PID=$! (log: $log_file)"
    printf '%s\n' "$!" > "$LOG_DIR/.server_${config}_${port}.pid"
}

run_exp_ablation() {
    local nwf="${1:-8}"

    log "=========================================="
    log "ABLATION EXP: ${nwf} workflows x 5 agents"
    log "Configs: lru_nocache / lru_wb_only / lru_wb_pf / priority_wb_only / kvflow"
    log "=========================================="

    local CONFIGS="lru_nocache lru_wb_only lru_wb_pf priority_wb_only kvflow"
    local port=30100

    for cfg in $CONFIGS; do
        log "--- Ablation config: $cfg (port=$port) ---"
        start_server_ablation "$cfg" "$port"
        wait_ready "$port" 360 || { log "ERROR: server $cfg failed to start"; return 1; }
        sleep 3

        if [[ "$cfg" != "lru_nocache" ]]; then
            BASELINE_JSON=""
            for candidate in "$(ls -t "$RESULT_DIR"/mwf_lru_nocache_*_${nwf}wf.json 2>/dev/null | head -1)" \
                            "$(ls -t "$RESULT_DIR"/mwf_lru_nocache_*${nwf}wf*.json 2>/dev/null | head -1)"; do
                if [[ -n "$candidate" ]] && [[ -f "$candidate" ]]; then
                    BASELINE_JSON="$candidate"
                    break
                fi
            done
            [ -n "$BASELINE_JSON" ] && BASELINE_JSON="--baseline-json $BASELINE_JSON" || BASELINE_JSON=""
        else
            BASELINE_JSON=""
        fi

        run_benchmark "$cfg" "$port" \
            --num-workflows "$nwf" --agents-per-workflow 5 \
            --tier0-len 512 --tier1-len 1024 --tier2-len 512 \
            --suffix-len 64 --output-len 64 \
            --num-rounds 5 --warmup-rounds 1 \
            $BASELINE_JSON

        stop_server "$cfg" "$port"
        sleep 30
        port=$((port + 10))
    done

    log "ABLATION EXP COMPLETE"
}

usage() {
    printf '%s\n' \
        "Usage: ./run_pipeline.sh [exp]" \
        "" \
        "Fair linear experiments:" \
        "  exp1-fair  1wf x 10ag (low pressure)" \
        "  exp2-fair  4wf x 10ag (medium pressure)" \
        "" \
        "DAG workflow experiments:" \
        "  exp1-dag   4 DAG workflows (parallel_dev, low pressure)" \
        "  exp2-dag   16 DAG workflows (parallel_dev, high pressure)" \
        "  dag-all     run exp1-dag && exp2-dag" \
        "" \
        "Ablation experiments:" \
        "  ablation-8wf   8wf x 5ag" \
        "  ablation-16wf  16wf x 5ag" \
        "  ablation-32wf  32wf x 5ag" \
        "" \
        "All configs use same HiCache (90k, write_back), only eviction differs."
}

PYTHON_BIN="${PYTHON_BIN:-$(which python3 2>/dev/null || echo python)}"

TARGET="${1:-all}"
case "$TARGET" in
    exp1-fair)  run_exp1_fair ;;
    exp2-fair)  run_exp2_fair ;;
    exp1-dag)   run_exp1_dag ;;
    exp2-dag)   run_exp2_dag ;;
    dag-all)    run_exp1_dag && run_exp2_dag ;;
    ablation-8wf)  run_exp_ablation 8 ;;
    ablation-16wf) run_exp_ablation 16 ;;
    ablation-32wf) run_exp_ablation 32 ;;
    all)         run_exp1_fair && run_exp2_fair ;;
    help|--help|-h) usage; exit 0 ;;
    *)           usage; exit 1 ;;
esac

print_summary
