#!/bin/bash
# =============================================================================
# vLLM vs SGLang vs KVFlow Triple Comparison Pipeline
#
# Compares 4 configurations:
#   vllm           - vLLM baseline with LRU + CPU offload
#   sglang         - SGLang with LRU + CPU offload (no HiCache)
#   sglang_hicache - SGLang with LRU + HiCache
#   kvflow         - SGLang with Priority + HiCache (full KVFlow)
#
# Test configuration: DAG 4wf × 5ag
# Tier: tier0=2048, tier1=1024, tier2=1024
# Rounds: 5 + 1 warmup
#
# All output written to files -- NO stdout echo/print (NFS I/O safety)
# =============================================================================

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
SGLANG_ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LOG_DIR:-/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow}"
RESULT_DIR="$LOG_DIR/results"
MODEL_PATH="${MODEL_PATH:-/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B}"

SGLANG_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# All output goes to a local pipeline log (no NFS stdout)
PIPELINE_LOG="$LOG_DIR/pipeline_triple.log"
exec >> "$PIPELINE_LOG" 2>&1

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# Python binary resolution for SGLang
SGLANG_PYTHON="$SGLANG_ENV_PATH/bin/python"
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_DISABLE_FLEX_ATTENTION=1

log "PYTHON_BIN: $SGLANG_PYTHON"
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
    local health_path="${3:-/health_generate}"
    log "Waiting for server on port $port..."
    for i in $(seq 1 "$max_wait"); do
        if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$port${health_path}" > /dev/null 2>&1; then
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

# =============================================================================
# Server launch configurations
# =============================================================================

start_server_vllm() {
    local port="$1"
    local log_file="$LOG_DIR/server_vllm_${port}.log"

    log "Starting vLLM server on port $port"
    kill_port "$port"

    export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda)))}"
    export LD_LIBRARY_PATH="$SGLANG_ENV_PATH/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

    # Use sglang-kvflow Python for vLLM server
    local VLLM_PY="$SGLANG_PYTHON"

    # Check if vLLM is installed, if not install it
    if ! "$VLLM_PY" -c "import vllm" 2>/dev/null; then
        log "Installing vLLM..."
        "$VLLM_PY" -m pip install vllm --quiet 2>&1 || {
            log "ERROR: Failed to install vLLM"
            return 1
        }
    fi

    # vLLM server with CPU offload
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$port" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --enforce-eager \
        > "$log_file" 2>&1 &

    log "vLLM server PID=$! (log: $log_file)"
    printf '%s\n' "$!" > "$LOG_DIR/.server_vllm_${port}.pid"
}

start_server_sglang() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    log "Starting SGLang server: config=$config, port=$port"
    kill_port "$port"

    export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda)))}"
    export LD_LIBRARY_PATH="$SGLANG_ENV_PATH/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

    local RADIX="lru"
    local HICACHE_FLAGS=""

    case "$config" in
        kvflow)
            RADIX="priority"
            HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back"
            ;;
        sglang_hicache)
            RADIX="lru"
            HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back"
            ;;
        sglang)
            RADIX="lru"
            HICACHE_FLAGS=""
            ;;
    esac

    "$SGLANG_PYTHON" -m sglang.launch_server \
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

    log "SGLang server PID=$! (log: $log_file)"
    printf '%s\n' "$!" > "$LOG_DIR/.server_${config}_${port}.pid"
}

stop_server() {
    local name="$1"
    local port="$2"
    kill_port "$port"
    rm -f "$LOG_DIR/.server_${name}_${port}.pid"
    sleep 15
    pkill -9 -f "sglang.launch_server.*--port $port" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.*--port $port" 2>/dev/null || true
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

    local LOCAL_SITE="$("$SGLANG_PYTHON" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo "")"

    log "Syntax check bench_multi_workflow.py..."
    if ! "$SGLANG_PYTHON" -B -m py_compile "$SGLANG_ROOT_DIR/benchmark/multi_workflow/bench_multi_workflow.py" 2>&1; then
        log "ERROR: bench_multi_workflow.py has syntax errors!"
        return 1
    fi

    local attempt=1
    local max_attempts=3

    while true; do
        if PYTHONNOUSERSITE=1 \
            PYTHONPATH="$LOCAL_SITE${LOCAL_SITE:+:}$SGLANG_ROOT_DIR" \
            "$SGLANG_PYTHON" -B "$SGLANG_ROOT_DIR/benchmark/multi_workflow/bench_multi_workflow.py" \
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
    log "TRIPLE COMPARE RESULTS"
    log "=========================================="
    log "Results: $RESULT_DIR"
    log "Logs:    $LOG_DIR"
    log "JSON files:"
    ls -lh "$RESULT_DIR"/mwf_*_triple_*.json 2>/dev/null | awk '{print "  "$NF}'
}

# =============================================================================
# Experiment definitions
# =============================================================================

run_vllm() {
    log "=========================================="
    log "TRIPLE: Running vLLM baseline"
    log "Config: LRU + CPU offload 32GB"
    log "=========================================="

    local PORT=31000

    start_server_vllm "$PORT"
    wait_ready "$PORT" 600 "/health" || return 1
    sleep 3

    local DAG_CONFIG="$BENCHMARK_DIR/configs/dag_parallel_dev.json"
    run_benchmark "vllm_triple" "$PORT" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows 4 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1

    stop_server "vllm" "$PORT"
    sleep 60
    sync

    log "vLLM baseline COMPLETE"
}

run_sglang() {
    log "=========================================="
    log "TRIPLE: Running SGLang baseline (no HiCache)"
    log "Config: LRU + CPU offload, no HiCache"
    log "=========================================="

    local PORT=31010

    start_server_sglang "sglang" "$PORT"
    wait_ready "$PORT" 360 "/health_generate" || return 1
    sleep 3

    local BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_vllm_triple_*_4wf*.json 2>/dev/null | head -1)"
    [ -n "$BASELINE_JSON" ] && BASELINE_JSON="--baseline-json $BASELINE_JSON" || BASELINE_JSON=""

    local DAG_CONFIG="$BENCHMARK_DIR/configs/dag_parallel_dev.json"
    run_benchmark "sglang_triple" "$PORT" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows 4 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_JSON

    stop_server "sglang" "$PORT"
    sleep 60
    sync

    log "SGLang baseline COMPLETE"
}

run_sglang_hicache() {
    log "=========================================="
    log "TRIPLE: Running SGLang with HiCache"
    log "Config: LRU + HiCache + CPU offload"
    log "=========================================="

    local PORT=31020

    start_server_sglang "sglang_hicache" "$PORT"
    wait_ready "$PORT" 360 "/health_generate" || return 1
    sleep 3

    local BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_vllm_triple_*_4wf*.json 2>/dev/null | head -1)"
    [ -n "$BASELINE_JSON" ] && BASELINE_JSON="--baseline-json $BASELINE_JSON" || BASELINE_JSON=""

    local DAG_CONFIG="$BENCHMARK_DIR/configs/dag_parallel_dev.json"
    run_benchmark "sglang_hicache_triple" "$PORT" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows 4 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_JSON

    stop_server "sglang_hicache" "$PORT"
    sleep 60
    sync

    log "SGLang HiCache COMPLETE"
}

run_kvflow() {
    log "=========================================="
    log "TRIPLE: Running KVFlow (Priority + HiCache)"
    log "Config: Priority + HiCache + CPU offload"
    log "=========================================="

    local PORT=31030

    start_server_sglang "kvflow" "$PORT"
    wait_ready "$PORT" 360 "/health_generate" || return 1
    sleep 3

    local BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_vllm_triple_*_4wf*.json 2>/dev/null | head -1)"
    [ -n "$BASELINE_JSON" ] && BASELINE_JSON="--baseline-json $BASELINE_JSON" || BASELINE_JSON=""

    local DAG_CONFIG="$BENCHMARK_DIR/configs/dag_parallel_dev.json"
    run_benchmark "kvflow_triple" "$PORT" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows 4 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_JSON

    stop_server "kvflow" "$PORT"

    log "KVFlow COMPLETE"
}

usage() {
    printf '%s\n' \
        "Usage: ./run_pipeline_triple_compare.sh [config]" \
        "" \
        "Configurations:" \
        "  vllm           - vLLM baseline (LRU + CPU offload)" \
        "  sglang         - SGLang baseline (LRU, no HiCache)" \
        "  sglang_hicache - SGLang with HiCache (LRU + HiCache)" \
        "  kvflow         - KVFlow (Priority + HiCache)" \
        "  all            - Run all 4 configurations (default)" \
        "" \
        "Test: DAG 4wf × 5ag, tier0=2048, tier1=1024, tier2=1024, 5 rounds + 1 warmup"
}

TARGET="${1:-all}"
case "$TARGET" in
    vllm)           run_vllm ;;
    sglang)         run_sglang ;;
    sglang_hicache) run_sglang_hicache ;;
    kvflow)         run_kvflow ;;
    all)            run_vllm && run_sglang && run_sglang_hicache && run_kvflow ;;
    help|--help|-h) usage; exit 0 ;;
    *)              usage; exit 1 ;;
esac

print_summary
