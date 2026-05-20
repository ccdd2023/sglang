#!/bin/bash
# =============================================================================
# MISSING EXPERIMENT 3: kvflow exp2 (4wf × 5agent)
# Moderate pressure comparison
# =============================================================================

#SBATCH --job-name=kvflow-miss3
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --mail-type=END,FAIL

set -euo pipefail

OUT="/tmp/slurm-${SLURM_JOB_ID:-$$}.out"
exec >"$OUT" 2>&1

echo "[$(date)] MISSING EXP 3: kvflow 4wf × 5agent (Moderate Pressure)"
echo "================================================================"

SCRIPT_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow"
RESULT_DIR="$LOG_DIR/results"
MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
PORT=30022

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Environment setup
for conda_sh in \
    "/usr/local/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh" \
    "/home/comp/25480812/.conda/etc/profile.d/conda.sh" \
    "$HOME/.conda/etc/profile.d/conda.sh"
do
    [[ -f "$conda_sh" ]] && source "$conda_sh" && break
done

CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
PYTHON_BIN="$CONDA_ENV_PATH/bin/python"

export CC="/usr/local/gcc/gcc-11.2.0/bin/gcc"
export CXX="/usr/local/gcc/gcc-11.2.0/bin/g++"
for cuda_ver in cuda-13 cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1; do
    [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]] && \
        export CUDA_HOME="/usr/local/$cuda_ver" && break
done
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME:-/usr/local/cuda}/lib64"
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python"

echo "[$(date)] Environment ready"
echo "[$(date)] PYTHON_BIN: $PYTHON_BIN"

# Clear flashinfer JIT cache
rm -rf /home/comp/25480812/.cache/flashinfer 2>/dev/null || true

# GPU info
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# Kill existing server on port
kill_port() {
    local p="$1"
    if pid=$(lsof -ti :"$p" 2>/dev/null); then
        echo "[$(date)] Killing port $p (PID=$pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 8
    fi
}

wait_ready() {
    local port="$1"
    local max_wait="${2:-90}"
    echo "[$(date)] Waiting for server on port $port..."
    for i in $(seq 1 "$max_wait"); do
        if curl -sf --max-time 2 "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            echo "[$(date)] Server ready after $((i*5))s!"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && echo "[$(date)] ...still waiting, $((i*5))s..."
        sleep 5
    done
    echo "[$(date)] ERROR: Server on port $port did not start"
    tail -5 "$LOG_DIR"/server_*.log 2>/dev/null || true
    return 1
}

wait_gpu_free() {
    local threshold="${1:-4000}"
    local max_wait="${2:-600}"
    echo "[$(date)] Waiting for GPU memory to free below ${threshold}MiB..."
    for i in $(seq 1 "$max_wait"); do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -lt "$threshold" ]; then
            echo "[$(date)] GPU memory freed: ${used}MiB"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && echo "[$(date)] GPU memory still: ${used}MiB..."
        sleep 5
    done
    echo "[$(date)] WARNING: GPU memory still high after ${max_wait}s"
    return 1
}

start_server() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    echo "[$(date)] Starting server: config=$config, port=$port"
    kill_port "$port"

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size 1 \
        --mem-fraction-static 0.85 \
        --max-total-tokens 90000 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        --radix-eviction-policy priority \
        --enable-hierarchical-cache \
        --hicache-ratio 2.5 \
        --hicache-write-policy write_back \
        --enable-hicache-prefetch \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
        --disable-pie \
        --log-level info \
        > "$log_file" 2>&1 &

    echo "[$(date)] Server PID=$! (log: $log_file)"
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

    echo "[$(date)] Running $config benchmark on port $port..."
    cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow

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
# Main execution
# =============================================================================

wait_gpu_free 4000 || true
start_server "kvflow" "$PORT"
wait_ready "$PORT" 90 || exit 1
sleep 5

# Baseline: find hicache 4wf result
BASELINE="$(ls -t "$RESULT_DIR"/mwf_hicache_*4wf*.json 2>/dev/null | head -1 || true)"
BASELINE_ARG=""
[ -n "$BASELINE" ] && BASELINE_ARG="--baseline-json $BASELINE"

echo "[$(date)] Running kvflow 4wf × 5agent benchmark..."
run_benchmark "kvflow" "$PORT" \
    --num-workflows 4 --agents-per-workflow 5 \
    --shared-p-len 2048 --unique-p-len 512 \
    --suffix-len 64 --output-len 64 \
    --num-rounds 5 --warmup-rounds 1 \
    $BASELINE_ARG

stop_server "kvflow" "$PORT"

echo "[$(date)] ========================================"
echo "[$(date)] MISSING EXP 3 COMPLETE"
echo "[$(date)] ========================================"
ls -lh "$RESULT_DIR"/mwf_kvflow_*4wf*.json 2>/dev/null || ls -lh "$RESULT_DIR"/mwf_*.json | tail -5

# Copy output to home
cp "$OUT" "$LOG_DIR/slurm-${SLURM_JOB_ID}.out" 2>/dev/null || true
