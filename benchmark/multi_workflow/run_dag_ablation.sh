#!/bin/bash
# =============================================================================
# DAG Ablation Experiment: Priority vs Prefetch Contribution Analysis
#
# Tests 4 configurations to isolate the contribution of each component:
#   1. lru_wb_only : LRU + HiCache write_back (no prefetch)  ← baseline
#   2. priority_wb_only: Priority + HiCache write_back (no prefetch)
#   3. lru_wb_pf   : LRU + HiCache write_back + prefetch
#   4. kvflow      : Priority + HiCache write_back + prefetch
#
# At two pressure levels:
#   - Low:  4 workflows × 5 agents = 20 agents
#   - High: 16 workflows × 5 agents = 80 agents
#
# Usage:
#   ./run_dag_ablation.sh          # Run all (low + high pressure)
#   ./run_dag_ablation.sh low      # Low pressure only (4wf)
#   ./run_dag_ablation.sh high     # High pressure only (16wf)
#   ./run_dag_ablation.sh quick    # Quick smoke test (1wf, 2 rounds)
# =============================================================================

#SBATCH --job-name=kvflow-dag-ablation
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow"
RESULT_DIR="$LOG_DIR/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# CRITICAL: redirect stdout/stderr to shared home dir for monitoring
OUT="$LOG_DIR/slurm-ablation-${SLURM_JOB_ID:-$$}.out"
exec >"$OUT" 2>&1

MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
DAG_CONFIG="$SCRIPT_DIR/configs/dag_parallel_dev.json"

echo "[$(date)] DAG Ablation Experiment: Priority vs Prefetch"
echo "================================================================"

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

# Add conda env bin to PATH so ninja is found
export PATH="$CONDA_ENV_PATH/bin:$PATH"

export CC="/usr/local/gcc/gcc-11.2.0/bin/gcc"
export CXX="/usr/local/gcc/gcc-11.2.0/bin/g++"
# Find available CUDA version
CUDA_FOUND=""
for cuda_ver in cuda-13 cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1; do
    if [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/$cuda_ver"
        CUDA_FOUND="yes"
        break
    fi
done
if [[ -z "$CUDA_FOUND" ]]; then
    echo "[$(date)] WARNING: No CUDA found, using /usr/local/cuda"
    export CUDA_HOME="/usr/local/cuda"
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME}/lib64"
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python"

# ninja JIT cache permissions fixed by setting cache dir to home
# Also unset proxy vars so Python/curl bypass proxy for localhost health checks
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy
export NINJA_CACHE_DIR="/home/comp/25480812/.cache/ninja"
export TORCHINDUCTOR_CACHE_DIR="/home/comp/25480812/.cache/torchinductor"
export TORCH_COMPILE_DIR="/home/comp/25480812/.cache/torch_compile"
mkdir -p "$NINJA_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_COMPILE_DIR"

echo "[$(date)] Environment ready"

# =============================================================================
# Ablation configs: (bench_config_name, server_evict_policy, hicache_enabled, enable_prefetch, port)
# =============================================================================
# bench_config_name must match keys in CONFIGS dict in bench_multi_workflow.py:
#   lru_wb_only, priority_wb_only, lru_wb_pf, kvflow

ABLATION_CONFIGS=(
    "lru_wb_only:lru:true:false:30410"
    "priority_wb_only:priority:true:false:30411"
    "lru_wb_pf:lru:true:true:30412"
    "kvflow:priority:true:true:30413"
)

# =============================================================================
# Helper functions
# =============================================================================

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
    local max_wait="${2:-600}"
    echo "[$(date)] Waiting for server on port $port (timeout=$((max_wait*5))s)..."
    for i in $(seq 1 "$max_wait"); do
        if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            echo "[$(date)] Server ready after $((i*5))s!"
            return 0
        fi
        # Check if server process died
        if ! lsof -ti :"$port" > /dev/null 2>&1; then
            echo "[$(date)] WARNING: No process on port $port yet..."
        fi
        [ $((i % 12)) -eq 0 ] && echo "[$(date)] ...still waiting, $((i*5))s..."
        sleep 5
    done
    echo "[$(date)] ERROR: Server on port $port did not start (timeout)"
    for lf in "$LOG_DIR"/server_*_${port}.log; do
        echo "=== Log: $lf ==="
        tail -20 "$lf" 2>/dev/null || echo "(empty or missing)"
    done
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
    local bench_config="$1"
    local evict="$2"
    local hicache_enabled="$3"
    local enable_prefetch="$4"
    local port="$5"
    local log_file="$LOG_DIR/server_${bench_config}_${port}.log"

    echo "[$(date)] Starting server: config=$bench_config, port=$port, evict=$evict, hicache=$hicache_enabled, prefetch=$enable_prefetch"
    kill_port "$port"

    local SERVER_FLAGS=""
    if [[ "$hicache_enabled" == "true" ]]; then
        SERVER_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --hicache-io-backend direct"
        if [[ "$enable_prefetch" == "true" ]]; then
            SERVER_FLAGS="$SERVER_FLAGS --enable-hicache-prefetch"
        fi
    fi

    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size 1 \
        --mem-fraction-static 0.85 \
        --max-total-tokens 90000 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        --radix-eviction-policy "$evict" \
        $SERVER_FLAGS \
        --enable-cache-report \
        --disable-cuda-graph \
        --disable-pie \
        --log-level info \
        > "$log_file" 2>&1 &

    echo "[$(date)] Server PID=$! (log: $log_file)"
}

stop_server() {
    local port="$1"
    kill_port "$port"
    sleep 15
}

run_bench() {
    local bench_config="$1"
    local port="$2"
    shift 2
    # All remaining args are passed through to bench_multi_workflow.py
    local bench_log="$LOG_DIR/bench_${bench_config}_${port}.log"

    echo "[$(date)] Running $bench_config DAG benchmark on port $port..."
    echo "[$(date)] Benchmark args: $@"
    cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow

    "$PYTHON_BIN" -m benchmark.multi_workflow.bench_multi_workflow \
        --config "$bench_config" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --host 127.0.0.1 \
        --port "$port" \
        --agents-seed 42 \
        --seed 42 \
        --output-dir "$RESULT_DIR" \
        "$@" \
        2>&1 | tee "$bench_log"
}

# =============================================================================
# Run one pressure level: run ALL 4 configs sequentially
# =============================================================================

run_ablation_pressure_level() {
    local wf_label="$1"
    local num_wf="$2"
    local tier0="$3"
    local tier1="$4"
    local tier2="$5"

    echo ""
    echo "=============================================================="
    echo "ABLATION: $wf_label ($num_wf workflows)"
    echo "Config: tier0=$tier0, tier1=$tier1, tier2=$tier2"
    echo "=============================================================="

    for config_entry in "${ABLATION_CONFIGS[@]}"; do
        IFS=':' read -r bench_cfg evict hicache prefetch port <<< "$config_entry"

        echo ""
        echo "--- Config: $bench_cfg (evict=$evict, hicache=$hicache, prefetch=$prefetch) ---"

        # Start server
        start_server "$bench_cfg" "$evict" "$hicache" "$prefetch" "$port"
        wait_ready "$port" 600 || exit 1  # Model loading can take 5-10 minutes
        sleep 5

        # Run benchmark (pass ALL tier args as individual parameters)
        run_bench "$bench_cfg" "$port" \
            --num-workflows "$num_wf" \
            --tier0-len "$tier0" \
            --tier1-len "$tier1" \
            --tier2-len "$tier2" \
            --suffix-len 32 \
            --output-len 32 \
            --num-rounds 5 \
            --warmup-rounds 1

        # Stop server and wait for GPU to cool
        stop_server "$port"
        sleep 15
        sync

        # Find result file
        local latest_result
        latest_result=$(ls -t "$RESULT_DIR"/mwf_${bench_cfg}_*_${num_wf}wf*.json 2>/dev/null | head -1 || true)
        if [[ -n "$latest_result" ]]; then
            echo "[$(date)] Result: $latest_result"
        else
            echo "[$(date)] WARNING: No result file found for $bench_cfg"
        fi
    done

    echo ""
    echo "=============================================================="
    echo "ABLATION RESULTS: $wf_label"
    echo "=============================================================="

    # Parse and display results
    "$PYTHON_BIN" -c "
import json
import glob
import sys
import os

num_wf = $num_wf

configs = {
    'lru_wb_only':      {'label': 'LRU Baseline',           'priority': False, 'prefetch': False},
    'priority_wb_only': {'label': 'Priority (no prefetch)', 'priority': True, 'prefetch': False},
    'lru_wb_pf':        {'label': 'LRU + Prefetch',         'priority': False, 'prefetch': True},
    'kvflow':           {'label': 'KVFlow (full)',           'priority': True, 'prefetch': True},
}

results = {}
for cfg, info in configs.items():
    pattern = f'$RESULT_DIR/mwf_{cfg}_*_{num_wf}wf*.json'
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
            agg = data['aggregate']
            results[cfg] = {
                'label': info['label'],
                'ttft': agg.get('stable_ttft_avg_ms', 0),
                'e2e': agg.get('stable_e2e_avg_ms', 0),
                'hit': agg.get('est_ttft_hit_rate', 0) * 100,
                'warmup_ttft': agg.get('warmup_ttft_avg_ms', 0),
            }

if not results:
    print('No results found!')
    sys.exit(1)

baseline = results.get('lru_wb_only', {})
baseline_ttft = baseline.get('ttft', 1)

print()
print(f'{\"Config\":<28} {\"Stable TTFT\":>12} {\"Speedup\":>8} {\"vs Baseline\":>12} {\"Hit Rate\":>10} {\"Warmup\":>10}')
print('-' * 85)

for cfg in ['lru_wb_only', 'priority_wb_only', 'lru_wb_pf', 'kvflow']:
    if cfg not in results:
        continue
    r = results[cfg]
    speedup = baseline_ttft / r['ttft'] if r['ttft'] > 0 else 0
    diff_pct = (baseline_ttft - r['ttft']) / baseline_ttft * 100 if baseline_ttft > 0 else 0
    diff_str = f'+{diff_pct:.1f}%' if diff_pct >= 0 else f'{diff_pct:.1f}%'
    print(f\"{r['label']:<28} {r['ttft']:>10.1f}ms {speedup:>7.2f}x {diff_str:>12} {r['hit']:>9.1f}% {r['warmup_ttft']:>9.1f}ms\")

print()
print('Component Contribution Analysis:')
if 'lru_wb_only' in results and 'priority_wb_only' in results:
    p_ttft = results['priority_wb_only']['ttft']
    l_ttft = results['lru_wb_only']['ttft']
    print(f'  Priority alone (LRU→Priority):  {l_ttft/p_ttft:.2f}x = {(l_ttft-p_ttft)/l_ttft*100:+.1f}%')

if 'lru_wb_only' in results and 'lru_wb_pf' in results:
    pf_ttft = results['lru_wb_pf']['ttft']
    l_ttft = results['lru_wb_only']['ttft']
    print(f'  Prefetch alone (LRU→LRU+PF):    {l_ttft/pf_ttft:.2f}x = {(l_ttft-pf_ttft)/l_ttft*100:+.1f}%')

if 'lru_wb_pf' in results and 'kvflow' in results:
    kv_ttft = results['kvflow']['ttft']
    pf_ttft = results['lru_wb_pf']['ttft']
    print(f'  Priority over LRU+PF (PF→KVFlow): {pf_ttft/kv_ttft:.2f}x = {(pf_ttft-kv_ttft)/pf_ttft*100:+.1f}%')

if 'lru_wb_only' in results and 'kvflow' in results:
    kv_ttft = results['kvflow']['ttft']
    l_ttft = results['lru_wb_only']['ttft']
    print(f'  Combined (LRU→KVFlow):          {l_ttft/kv_ttft:.2f}x = {(l_ttft-kv_ttft)/l_ttft*100:+.1f}%')
"
}

# =============================================================================
# Main: Run experiments based on mode
# =============================================================================

MODE="${1:-all}"

case "$MODE" in
    quick)
        echo "Quick smoke test: 1wf, 2 rounds"
        run_ablation_pressure_level "quick" 1 512 512 256
        ;;
    low)
        echo "Low pressure: 4 workflows, 5 rounds"
        run_ablation_pressure_level "low_pressure" 4 2048 1024 1024
        ;;
    high)
        echo "High pressure: 16 workflows, 5 rounds"
        run_ablation_pressure_level "high_pressure" 16 2048 1024 1024
        ;;
    all)
        run_ablation_pressure_level "low_pressure" 4 2048 1024 1024
        run_ablation_pressure_level "high_pressure" 16 2048 1024 1024
        ;;
    *)
        echo "Usage: $0 [quick|low|high|all]"
        exit 1
        ;;
esac

echo ""
echo "[$(date)] ========================================"
echo "[$(date)] DAG ABLATION COMPLETE"
echo "[$(date)] ========================================"
ls -lh "$RESULT_DIR"/mwf_*_4wf*.json "$RESULT_DIR"/mwf_*_16wf*.json 2>/dev/null | tail -20
