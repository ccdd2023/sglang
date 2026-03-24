#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Priority Benchmark Runner
#
# Starts an SGLang server in Docker (with LRU or priority eviction), runs the
# bench_priority.py client against it, collects results, and cleans up.
#
# Usage:
#   ./benchmark/priority/run_bench.sh [--model qwen3-0.6b|qwen3-4b-awq] \
#                                   [--config baseline|priority|all] \
#                                   [--port 30200] \
#                                   [-- extra bench args]
#
# Requirements:
#   - Docker with --runtime=nvidia support
#   - NVIDIA GPU visible via nvidia-smi
#   - HF model cache at ~/.cache/huggingface
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/benchmark_results"

IMAGE="lmsysorg/sglang:dev"
SERVER_CONTAINER="priority-bench-server"
CLIENT_CONTAINER="priority-bench-client"
HEALTH_TIMEOUT=300
HEALTH_INTERVAL=5

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

detect_gpu() {
    local gpu_arch
    gpu_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    GPU_MAJOR="${gpu_arch%%.*}"
    GPU_ARCH="$gpu_arch"

    if [ "$GPU_MAJOR" -lt 8 ]; then
        SM75_FLAGS="--sampling-backend pytorch --disable-cuda-graph --disable-piecewise-cuda-graph"
    else
        SM75_FLAGS=""
    fi

    echo "[gpu] Detected compute capability $GPU_ARCH (major=$GPU_MAJOR)"
    # No --attention-backend needed: SGLang auto-selects flashinfer (works on sm75+)
    if [ -n "$SM75_FLAGS" ]; then
        echo "[gpu] SM75 compatibility flags: $SM75_FLAGS"
    fi
}

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

# Sets: MODEL_HF, MODEL_SHORT, SERVER_MODEL_FLAGS
set_model_preset() {
    local preset="$1"
    case "$preset" in
        qwen3-0.6b)
            MODEL_HF="Qwen/Qwen3-0.6B"
            MODEL_SHORT="qwen3-0.6b"
            SERVER_MODEL_FLAGS="--mem-fraction-static 0.3"
            ;;
        qwen3-4b-awq)
            MODEL_HF="Qwen/Qwen3-4B-AWQ"
            MODEL_SHORT="qwen3-4b-awq"
            SERVER_MODEL_FLAGS="--dtype half --context-length 4096 --max-total-tokens 4096 --mem-fraction-static 0.5 --chunked-prefill-size 512 --max-prefill-tokens 2048"
            ;;
        *)
            echo "[error] Unknown model preset: $preset"
            echo "  Supported: qwen3-0.6b, qwen3-4b-awq"
            exit 1
            ;;
    esac
    echo "[model] Preset: $preset -> $MODEL_HF"
}

# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

cleanup() {
    echo "[cleanup] Stopping containers..."
    docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
    docker rm -f "$CLIENT_CONTAINER" >/dev/null 2>&1 || true
}

wait_for_server() {
    local port="$1"
    local url="http://127.0.0.1:${port}/health_generate"
    local elapsed=0

    echo "[health] Waiting for server at $url (timeout=${HEALTH_TIMEOUT}s)..."
    while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "[health] Server ready (${elapsed}s)"
            return 0
        fi
        sleep "$HEALTH_INTERVAL"
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done

    echo "[health] TIMEOUT after ${HEALTH_TIMEOUT}s"
    echo "[health] Server logs (last 40 lines):"
    docker logs --tail 40 "$SERVER_CONTAINER" 2>&1 || true
    return 1
}

start_server() {
    local config="$1"
    local port="$2"

    # Map config to eviction policy + HiCache flags
    local eviction_policy hicache_flags bench_config
    case "$config" in
        baseline)
            eviction_policy="lru"
            hicache_flags=""
            bench_config="baseline"
            ;;
        hicache)
            eviction_policy="lru"
            hicache_flags="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-io-backend direct"
            bench_config="baseline"
            ;;
        priority-gpu)
            eviction_policy="priority"
            hicache_flags=""
            bench_config="priority"
            ;;
        priority-full)
            eviction_policy="priority"
            hicache_flags="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-io-backend direct"
            bench_config="priority"
            ;;
        *)
            echo "[error] Unknown config: $config"
            exit 1
            ;;
    esac

    echo "[server] Starting SGLang server: model=$MODEL_HF, eviction=$eviction_policy, hicache=${hicache_flags:-none}, port=$port"

    # Build the server launch command
    local server_cmd="
        pip install sglang-kernel --force-reinstall -q 2>/dev/null || true
        pip install -U transformers -q 2>/dev/null || true
        cd /sgl-workspace/sglang && pip install -e 'python[all]' -q 2>/dev/null || true
        FLASHINFER_DISABLE_VERSION_CHECK=1 python -m sglang.launch_server \
            --model-path $MODEL_HF \
            --port $port \
            --host 0.0.0.0 \
            --radix-eviction-policy $eviction_policy \
            --enable-cache-report \
            $hicache_flags \
            $SERVER_MODEL_FLAGS \
            $SM75_FLAGS
    "

    docker run -d \
        --name "$SERVER_CONTAINER" \
        --runtime=nvidia --gpus all \
        --shm-size 16g --ipc=host \
        -p "${port}:${port}" \
        -v "${PROJECT_DIR}:/sgl-workspace/sglang" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -e "FLASHINFER_DISABLE_VERSION_CHECK=1" \
        "$IMAGE" \
        bash -c "$server_cmd"

    echo "[server] Container $SERVER_CONTAINER started"
}

run_benchmark_client() {
    local config="$1"
    local port="$2"
    shift 2
    local extra_args=("$@")

    # Map 4-config name to bench_priority.py config (baseline or priority)
    local bench_config
    case "$config" in
        baseline|hicache) bench_config="baseline" ;;
        priority-gpu|priority-full) bench_config="priority" ;;
        *) bench_config="$config" ;;
    esac

    echo "[bench] Running benchmark client: config=$config (bench=$bench_config), port=$port"

    local bench_args="--config $bench_config --host 127.0.0.1 --port $port --model $MODEL_HF"

    # Append extra args
    for arg in "${extra_args[@]}"; do
        bench_args+=" $arg"
    done

    # Find baseline JSON for speedup comparison (if running priority config)
    if [ "$config" = "priority" ]; then
        local baseline_json
        baseline_json=$(ls -t "$RESULTS_DIR"/priority_bench_baseline_*.json 2>/dev/null | head -1 || true)
        if [ -n "$baseline_json" ]; then
            echo "[bench] Found baseline JSON for comparison: $(basename "$baseline_json")"
            bench_args+=" --baseline-json /sgl-workspace/results/$(basename "$baseline_json")"
        fi
    fi

    echo "[bench] python -m benchmark.priority.bench_priority $bench_args"

    docker rm -f "$CLIENT_CONTAINER" >/dev/null 2>&1 || true
    docker run --rm \
        --name "$CLIENT_CONTAINER" \
        --network host \
        --runtime=nvidia --gpus all \
        --shm-size 16g --ipc=host \
        -v "${PROJECT_DIR}:/sgl-workspace/sglang" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${RESULTS_DIR}:/sgl-workspace/results" \
        "$IMAGE" \
        bash -c "
            cd /sgl-workspace/sglang
            pip install -e 'python[all]' -q 2>/dev/null || true
            pip install aiohttp -q 2>/dev/null || true
            FLASHINFER_DISABLE_VERSION_CHECK=1 python -m benchmark.priority.bench_priority \
                --output-dir /sgl-workspace/results \
                $bench_args
        "
}

# ---------------------------------------------------------------------------
# Single config run
# ---------------------------------------------------------------------------

run_single_config() {
    local config="$1"
    local port="$2"
    shift 2
    local extra_args=("$@")

    echo "================================================================"
    echo "  Config: $config ($MODEL_SHORT)"
    echo "================================================================"

    # 1. Cleanup
    cleanup

    # 2. Start server
    start_server "$config" "$port"

    # 3. Wait for health
    if ! wait_for_server "$port"; then
        echo "[FAIL] Server did not start for config=$config"
        cleanup
        return 1
    fi

    # 4. Run benchmark
    if ! run_benchmark_client "$config" "$port" "${extra_args[@]}"; then
        echo "[FAIL] Benchmark failed for config=$config"
        echo "[FAIL] Server logs (last 30 lines):"
        docker logs --tail 30 "$SERVER_CONTAINER" 2>&1 || true
        cleanup
        return 1
    fi

    # 5. Show result
    local output_json
    output_json=$(ls -t "$RESULTS_DIR"/priority_bench_${config}_*.json 2>/dev/null | head -1 || true)
    if [ -n "$output_json" ]; then
        echo "[result] Saved: $output_json"
    else
        echo "[warn] No output JSON found"
    fi

    # 6. Cleanup
    cleanup
    echo ""
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
    ./benchmark/priority/run_bench.sh [options] [-- extra bench_priority.py args]

Options:
    --model MODEL     Model preset: qwen3-0.6b (default), qwen3-4b-awq
    --config CONFIG   Config: baseline, hicache, priority-gpu, priority-full, or all (default: all)
    --port PORT       Server port (default: 30200)
    --help, -h        Show this help

Configs:
    baseline    LRU eviction, GPU-only cache
    hicache     LRU eviction + HiCache CPU offloading
    priority-gpu  Priority eviction (Priority), GPU-only
    priority-full  Priority eviction + HiCache (Priority Full)
    all         Run all 4 configs sequentially

Extra args after -- are forwarded to bench_priority.py (e.g., --num-agents 8).

Examples:
    # Run all 4 configs with default model
    ./benchmark/priority/run_bench.sh

    # Run only priority-full (Priority + HiCache)
    ./benchmark/priority/run_bench.sh --config priority-full

    # Run with E1 parameters
    ./benchmark/priority/run_bench.sh --config all -- --prefix-len 2048 --num-rounds 5 --warmup-rounds 2

    # Use larger model
    ./benchmark/priority/run_bench.sh --model qwen3-4b-awq --config all
EOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    local model_preset="qwen3-0.6b"
    local config_arg="all"
    local port=30200
    local extra_args=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --model)  model_preset="${2:?'--model requires a value'}"; shift 2 ;;
            --config) config_arg="${2:?'--config requires a value'}"; shift 2 ;;
            --port)   port="${2:?'--port requires a value'}"; shift 2 ;;
            --help|-h) usage; exit 0 ;;
            --)       shift; extra_args=("$@"); break ;;
            *)        echo "[error] Unknown arg: $1"; usage; exit 1 ;;
        esac
    done

    # Setup
    mkdir -p "$RESULTS_DIR"
    detect_gpu
    set_model_preset "$model_preset"

    # Ensure clean state
    trap cleanup EXIT

    echo "================================================================"
    echo "Priority Benchmark Suite"
    echo "  Model:   $MODEL_HF ($MODEL_SHORT)"
    echo "  Config:  $config_arg"
    echo "  Port:    $port"
    echo "  Output:  $RESULTS_DIR/"
    echo "  GPU:     sm$GPU_ARCH (attention=auto-flashinfer)"
    echo "================================================================"
    echo ""

    local failed=0

    case "$config_arg" in
        all)
            # Run all 4 configs: baseline → hicache → priority-gpu → priority-full
            run_single_config "baseline"  "$port" "${extra_args[@]}" || failed=$((failed + 1))
            run_single_config "hicache"   "$port" "${extra_args[@]}" || failed=$((failed + 1))
            run_single_config "priority-gpu" "$port" "${extra_args[@]}" || failed=$((failed + 1))
            run_single_config "priority-full"    "$port" "${extra_args[@]}" || failed=$((failed + 1))
            ;;
        baseline|hicache|priority-gpu|priority-full)
            run_single_config "$config_arg" "$port" "${extra_args[@]}" || failed=$((failed + 1))
            ;;
        *)
            echo "[error] Unknown config: $config_arg (expected: baseline, hicache, priority-gpu, priority-full, or all)"
            exit 1
            ;;
    esac

    echo "================================================================"
    if [ "$failed" -eq 0 ]; then
        echo "All benchmarks complete. Results in: $RESULTS_DIR/"
    else
        echo "Benchmarks finished with $failed failure(s). Results in: $RESULTS_DIR/"
    fi
    ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "(no JSON files found)"
    echo "================================================================"

    return "$failed"
}

main "$@"
