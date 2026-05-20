#!/bin/bash
# =============================================================================
# SLURM Job Submission Script for KVFlow 14B Experiment
#
# CRITICAL FIX: Qwen3-14B requires TP=2 (2 GPUs) but old script used --gres=gpu:a100:1
# This caused the server to wait indefinitely for the second GPU.
#
# Fixes applied:
#  1. --gres=gpu:a100:2 (TP=2 needs 2 GPUs)
#  2. --mem=512G (14B model needs more host memory for HiCache)
#  3. Extended wait_gpu_free timeout (300s instead of 120s)
#  4. Uses mem_fraction_static=0.72 to leave room for KV cache
#  5. Explicit GPU reset between hicache and kvflow steps
#
# Usage:
#   sbatch --parsable run_slurm_14b.sh           # Submit job, get JobID
#   sbatch run_slurm_14b.sh 14b-eviction-war     # Full eviction war (hicache then kvflow)
#   sbatch run_slurm_14b.sh 14b-exp1-large       # Single 10-agent workflow
# =============================================================================

#SBATCH --job-name=kvflow-14b
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=hkbugpusrv08,hkbugpusrv15
#SBATCH --gres=gpu:a100:2          # TP=2 requires 2 GPUs
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G                  # 14B model needs more host memory
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@example.com

set -euo pipefail

OUT="/tmp/slurm-${SLURM_JOB_ID:-$$}.out"
exec >"$OUT" 2>&1

BENCHMARK_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-14b-200ag"
RESULT_DIR="$LOG_DIR/results"
EXP_TYPE="${1:-14b-eviction-war}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

echo "[14B-FIX] =========================================="
echo "[14B-FIX] KVFlow 14B Experiment (FIXED GPU allocation)"
echo "[14B-FIX] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[14B-FIX] GPUs allocated: $SLURM_JOB_CONSTRAINT"
echo "[14B-FIX] EXP_TYPE: $EXP_TYPE"
echo "[14B-FIX] LOG_DIR: $LOG_DIR"
echo "[14B-FIX] =========================================="

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 || true

# Clear flashinfer JIT cache
rm -rf /home/comp/25480812/.cache/flashinfer 2>/dev/null || true

# Load modules
module load gcc/11.2.0 2>/dev/null || true
module load conda 2>/dev/null || true

for conda_sh in \
    "/usr/local/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh" \
    "/home/comp/25480812/.conda/etc/profile.d/conda.sh" \
    "$HOME/.conda/etc/profile.d/conda.sh"
do
    if [[ -f "$conda_sh" ]]; then
        source "$conda_sh"
        break
    fi
done

CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
if [[ ! -d "$CONDA_ENV_PATH" ]]; then
    echo "[14B-FIX] ERROR: conda env not found at $CONDA_ENV_PATH"
    exit 1
fi

export CC="/usr/local/gcc/gcc-11.2.0/bin/gcc"
export CXX="/usr/local/gcc/gcc-11.2.0/bin/g++"

for cuda_ver in cuda-13 cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1; do
    if [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/$cuda_ver"
        export PATH="$CUDA_HOME/bin:$PATH"
        break
    fi
done

export TORCHINDUCTOR_DISABLE_FLEX_ATTENTION=1
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"

export PYTHON_BIN="$CONDA_ENV_PATH/bin/python"

echo "[14B-FIX] Python: $("$PYTHON_BIN" --version 2>&1)"
"$PYTHON_BIN" -c "import sglang; print(f'sglang {sglang.__version__}')" 2>&1 || true
"$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1 || {
    echo "[14B-FIX] Rebuilding sgl_kernel..."
    cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/sgl-kernel
    "$PYTHON_BIN" -m pip install scikit-build-core cmake ninja --upgrade --quiet 2>&1 || true
    "$PYTHON_BIN" -m pip install -e . --no-build-isolation 2>&1 || true
    cd "$BENCHMARK_DIR"
}
"$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1 || true

# =============================================================================
# Core experiment functions
# =============================================================================

kill_port() {
    local p="$1"
    if pid=$(lsof -ti :"$p" 2>/dev/null); then
        echo "[14B-FIX] Killing port $p (PID=$pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 10
    fi
}

wait_ready() {
    local port="$1"
    local max_wait="${2:-600}"
    echo "[14B-FIX] Waiting for server on port $port (max ${max_wait}s)..."
    for i in $(seq 1 "$max_wait"); do
        if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            echo "[14B-FIX] Server on port $port ready after $((i*5))s!"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && echo "[14B-FIX]   ...still waiting, $((i*5))s..."
        sleep 5
    done
    echo "[14B-FIX] ERROR: Server on port $port did not start"
    tail -10 "$LOG_DIR"/server_*.log 2>/dev/null || true
    return 1
}

wait_gpu_free() {
    local threshold="${1:-10000}"
    local max_wait="${2:-300}"
    echo "[14B-FIX] Waiting for GPU memory to free (threshold: ${threshold}MiB, max ${max_wait}s)..."
    for i in $(seq 1 "$max_wait"); do
        local mem_free
        mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [[ -n "$mem_free" ]] && [[ "$mem_free" -gt "$threshold" ]]; then
            echo "[14B-FIX] GPU memory freed: ${mem_free}MiB after $((i*5))s"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && echo "[14B-FIX]   GPU memory still: ${mem_free}MiB..."
        sleep 5
    done
    echo "[14B-FIX] WARNING: GPU memory still high after ${max_wait}s"
    return 1
}

reset_gpu_memory() {
    echo "[14B-FIX] Attempting GPU memory reset..."
    kill_port 30081
    kill_port 30082
    kill_port 30091
    kill_port 30092
    sleep 15
    # Try to free any stuck Python processes
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 10
    wait_gpu_free 50000 300 || true
}

run_server() {
    local config="$1"
    local port="$2"
    local model="$3"
    local tp_size="${4:-2}"
    local mem_frac="${5:-0.72}"
    local hicache_flags="$6"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    echo "[14B-FIX] =========================================="
    echo "[14B-FIX] Starting server: $config on port $port"
    echo "[14B-FIX]   Model: $model"
    echo "[14B-FIX]   TP: $tp_size, MEM_FRAC: $mem_frac"
    echo "[14B-FIX]   HiCache: $hicache_flags"
    echo "[14B-FIX] =========================================="

    kill_port "$port"

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$model" \
        --port "$port" \
        --tp-size "$tp_size" \
        --mem-fraction-static "$mem_frac" \
        --max-total-tokens 90000 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        $hicache_flags \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
        --disable-pie \
        --skip-server-warmup \
        --log-level info \
        > "$log_file" 2>&1 &

    local server_pid=$!
    echo "[14B-FIX] Server PID=$server_pid (log: $log_file)"
}

run_benchmark() {
    local config="$1"
    local port="$2"
    local bench_log="$LOG_DIR/bench_${config}_${port}.log"
    shift 3
    local extra_args="$@"

    echo "[14B-FIX] Running $config benchmark on port $port..."

    PYTHONNOUSERSITE=1 \
    PYTHONPATH="/home/comp/25480812/.conda/envs/sglang-kvflow/lib/python3.12/site-packages:/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python" \
    "$PYTHON_BIN" -m benchmark.multi_workflow.bench_multi_workflow \
        --config "$config" \
        --host 127.0.0.1 \
        --port "$port" \
        --agents-seed 42 \
        --seed 42 \
        --output-dir "$RESULT_DIR" \
        $extra_args \
        >> "$bench_log" 2>&1

    echo "[14B-FIX] Benchmark complete."
}

stop_server() {
    local port="$1"
    echo "[14B-FIX] Stopping server on port $port..."
    kill_port "$port"
    sleep 20
    pkill -9 -f "sglang.launch_server.*--port $port" 2>/dev/null || true
    sleep 15
}

# =============================================================================
# Experiment 1: 14B Eviction War
# =============================================================================
run_eviction_war() {
    echo "[14B-FIX] =========================================="
    echo "[14B-FIX] 14B EVICTION WAR: KVFlow vs LRU"
    echo "[14B-FIX] 20 workflows × 10 agents = 200 agents"
    echo "[14B-FIX] Config: 20wf x 10ag, shared=4096, unique=64, rounds=10"
    echo "[14B-FIX] =========================================="

    local MODEL="/home/comp/25480812/models/hub/models--Qwen--Qwen3-14B"
    local PORT_LRU=30091
    local PORT_KVFLOW=30092
    local BASELINE_JSON=""

    # STEP 1: hicache90k (LRU baseline)
    echo "[14B-FIX] =========================================="
    echo "[14B-FIX] STEP 1: hicache90k (LRU baseline)"
    echo "[14B-FIX] =========================================="
    run_server "hicache90k" "$PORT_LRU" "$MODEL" 2 0.72 \
        "--radix-eviction-policy lru --enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back"

    wait_ready "$PORT_LRU" 600 || { echo "[14B-FIX] ERROR: LRU server failed to start"; return 1; }
    sleep 5

    run_benchmark "hicache90k" "$PORT_LRU" \
        --num-workflows 20 --agents-per-workflow 10 \
        --tier0-len 2048 --tier1-len 1024 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 10 --warmup-rounds 1

    BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_hicache90k_*20wf*.json 2>/dev/null | head -1)"
    [[ -n "$BASELINE_JSON" ]] && echo "[14B-FIX] Baseline: $BASELINE_JSON"

    stop_server "$PORT_LRU"
    sleep 30
    reset_gpu_memory

    # STEP 2: kvflow (Priority)
    echo "[14B-FIX] =========================================="
    echo "[14B-FIX] STEP 2: kvflow (Priority)"
    echo "[14B-FIX] =========================================="
    run_server "kvflow" "$PORT_KVFLOW" "$MODEL" 2 0.72 \
        "--radix-eviction-policy priority --enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"

    wait_ready "$PORT_KVFLOW" 600 || { echo "[14B-FIX] ERROR: KVFlow server failed to start"; return 1; }
    sleep 5

    local BASELINE_ARG=""
    [[ -n "$BASELINE_JSON" ]] && BASELINE_ARG="--baseline-json $BASELINE_JSON"

    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 20 --agents-per-workflow 10 \
        --tier0-len 2048 --tier1-len 512 --tier2-len 1024 \
        --suffix-len 32 --output-len 32 \
        --num-rounds 10 --warmup-rounds 1 \
        $BASELINE_ARG

    stop_server "$PORT_KVFLOW"

    echo "[14B-FIX] =========================================="
    echo "[14B-FIX] 14B EVICTION WAR COMPLETE"
    echo "[14B-FIX] =========================================="
}

# =============================================================================
# Experiment 2: 14B Exp1-Large (single workflow, large prefix)
# =============================================================================
run_exp1_large() {
    echo "[14B-FIX] =========================================="
    echo "[14B-FIX] 14B EXP1-LARGE: Single 10-Agent Workflow"
    echo "[14B-FIX] =========================================="

    local MODEL="/home/comp/25480812/models/hub/models--Qwen--Qwen3-14B"
    local PORT_LRU=30091
    local PORT_KVFLOW=30092
    local BASELINE_JSON=""

    echo "[14B-FIX] STEP 1: hicache90k (LRU)"
    run_server "hicache90k" "$PORT_LRU" "$MODEL" 2 0.72 \
        "--radix-eviction-policy lru --enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back"

    wait_ready "$PORT_LRU" 600 || { echo "[14B-FIX] ERROR: LRU server failed"; return 1; }
    sleep 5

    run_benchmark "hicache90k" "$PORT_LRU" \
        --num-workflows 1 --agents-per-workflow 10 \
        --tier0-len 2048 --tier1-len 512 --tier2-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1

    BASELINE_JSON="$(ls -t "$RESULT_DIR"/mwf_hicache90k_*1wf*10ag*.json 2>/dev/null | head -1)"
    [[ -n "$BASELINE_JSON" ]] && echo "[14B-FIX] Baseline: $BASELINE_JSON"

    stop_server "$PORT_LRU"
    sleep 30
    reset_gpu_memory

    echo "[14B-FIX] STEP 2: kvflow (Priority)"
    run_server "kvflow" "$PORT_KVFLOW" "$MODEL" 2 0.72 \
        "--radix-eviction-policy priority --enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"

    wait_ready "$PORT_KVFLOW" 600 || { echo "[14B-FIX] ERROR: KVFlow server failed"; return 1; }
    sleep 5

    local BASELINE_ARG=""
    [[ -n "$BASELINE_JSON" ]] && BASELINE_ARG="--baseline-json $BASELINE_JSON"

    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 1 --agents-per-workflow 10 \
        --tier0-len 2048 --tier1-len 512 --tier2-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_ARG

    stop_server "$PORT_KVFLOW"

    echo "[14B-FIX] =========================================="
    echo "[14B-FIX] 14B EXP1-LARGE COMPLETE"
    echo "[14B-FIX] =========================================="
}

# =============================================================================
# Dispatch
# =============================================================================
case "$EXP_TYPE" in
    14b-eviction-war)  run_eviction_war ;;
    14b-exp1-large)    run_exp1_large ;;
    all)                run_eviction_war && run_exp1_large ;;
    *)                  echo "[14B-FIX] Unknown exp: $EXP_TYPE"; exit 1 ;;
esac

# Copy output to home
cp "$OUT" "$LOG_DIR/slurm-${SLURM_JOB_ID:-$$}.out" 2>/dev/null || true
echo "[14B-FIX] Results:"
ls -lh "$RESULT_DIR"/mwf_*.json 2>/dev/null | awk '{print "  "$NF}' || true
echo "[14B-FIX] Done."
