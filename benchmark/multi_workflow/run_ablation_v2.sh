#!/bin/bash
# =============================================================================
# KVFlow Ablation V2: Tiered Priority vs Original Priority
#
# This script specifically tests the new TieredPriorityStrategy to verify
# that it fixes the Priority x Prefetch negative interaction problem.
#
# Expected results:
# - Priority alone: +5.9% ~ +14.4% improvement
# - Priority + Prefetch: -13.5% ~ -24.6% (NEGATIVE due to interference)
# - Tiered + Prefetch: > +8% (should fix the interaction issue)
#
# Usage:
#   sbatch run_ablation_v2.sh           # Default: 8 workflows
#   sbatch run_ablation_v2.sh 16       # High pressure: 16 workflows
# =============================================================================

#SBATCH --job-name=kvflow-ablation-v2
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=hkbugpusrv07,hkbugpusrv08,hkbugpusrv15  # CUDA incompatibility
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@example.com

set -uo pipefail

BENCHMARK_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow"
NWF="${1:-8}"

# Output to home directory directly
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-ablation-v2"
RESULT_DIR="$LOG_DIR/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

OUT="$LOG_DIR/slurm-${SLURM_JOB_ID:-$$}.out"
exec >"$OUT" 2>&1

echo "=========================================="
echo "KVFlow Ablation V2: Tiered Priority"
echo "Workflows: $NWF"
echo "=========================================="

# Clear flashinfer JIT cache
rm -rf /home/comp/25480812/.cache/flashinfer 2>/dev/null || true

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 || true

# Modules
module load gcc/11.2.0 2>/dev/null || true
module load conda 2>/dev/null || true

# Source conda.sh
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
MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"

# Environment setup
export CC="${CC:-/usr/local/gcc/gcc-11.2.0/bin/gcc}"
export CXX="${CXX:-/usr/local/gcc/gcc-11.2.0/bin/g++}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
for cuda_ver in cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1 cuda-11.8; do
    if [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/$cuda_ver"
        break
    fi
done
export PATH="$CUDA_HOME/bin:$PATH"

export TORCHINDUCTOR_DISABLE_FLEX_ATTENTION=1
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}:/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python"

PYTHON="$CONDA_ENV_PATH/bin/python"

# Verify CUDA
echo "Checking CUDA..."
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>&1; then
    echo "ERROR: CUDA not available"
    exit 1
fi
echo "CUDA OK"

# Ensure correct PyTorch version (sglang requires torch>=2.8 with CUDA 12+)
echo "Installing correct PyTorch version..."
"$PYTHON" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall 2>&1 | tail -5

# Reinstall sglang from source
echo "Installing sglang..."
cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/python
"$PYTHON" -m pip install -e . --no-deps --quiet 2>&1 || true

# Build sgl_kernel from source (required for sglang-kvflow)
echo "Building sgl-kernel from source..."
cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/sgl-kernel
export CC=/usr/local/gcc/gcc-11.2.0/bin/gcc
export CXX=/usr/local/gcc/gcc-11.2.0/bin/g++
if "$PYTHON" -c "import sgl_kernel" 2>&1; then
    echo "sgl_kernel already installed"
else
    echo "Building sgl_kernel (this may take 10-30 minutes)..."
    "$PYTHON" -m pip install -e . --no-build-isolation 2>&1 | tail -20
    if "$PYTHON" -c "import sgl_kernel" 2>&1; then
        echo "sgl_kernel build: SUCCESS"
    else
        echo "WARNING: sgl_kernel build failed, but continuing anyway..."
    fi
fi

# Verify tiered strategy is available (skip full import due to sgl_kernel dependency)
echo "Verifying tiered eviction policy..."
"$PYTHON" -c "
import sys
sys.path.insert(0, '/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python')
# Check if file exists
import os
evict_policy_path = 'sglang/srt/mem_cache/evict_policy.py'
full_path = os.path.join('/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python', evict_policy_path)
with open(full_path, 'r') as f:
    content = f.read()
    if 'TieredPriorityStrategy' in content:
        print('TieredPriorityStrategy: OK (found in evict_policy.py)')
    else:
        print('ERROR: TieredPriorityStrategy not found')
        sys.exit(1)
" 2>&1 || {
    echo "ERROR: TieredPriorityStrategy not found"
    exit 1
}

# Run ablation
echo "Running ablation V2 with $NWF workflows..."
cd "$BENCHMARK_DIR"

# =============================================================================
# Test configurations:
# 1. lru_nocache: Baseline (no HiCache) - eviction=lru, hicache=off
# 2. priority_nopf: Priority without Prefetch - eviction=priority, hicache=on, prefetch=off
# 3. kvflow: Priority + Prefetch - eviction=priority, hicache=on, prefetch=on (NEGATIVE)
# 4. tiered: New Tiered + Prefetch - eviction=tiered, hicache=on, prefetch=on (FIXED)
# =============================================================================

PORT=30500

# Configuration 1: lru_nocache
cfg="lru_nocache"
echo "=========================================="
echo "Config: $cfg (port=$PORT)"
echo "=========================================="

LOG_FILE="$LOG_DIR/server_${cfg}.log"
"$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-total-tokens 90000 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 8192 \
    --radix-eviction-policy lru \
    --hicache-io-backend direct \
    --enable-cache-report \
    --disable-cuda-graph \
    --disable-pie \
    --skip-server-warmup \
    --log-level info \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server PID=$SERVER_PID"

for i in $(seq 1 60); do
    curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health_generate" > /dev/null 2>&1 && break
    [ $((i % 12)) -eq 0 ] && echo "  ...waiting..."
    sleep 5
done

BENCH_LOG="$LOG_DIR/bench_${cfg}.log"
"$PYTHON" -B bench_multi_workflow.py \
    --config "$cfg" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --agents-seed 42 \
    --seed 42 \
    --output-dir "$RESULT_DIR" \
    --num-workflows "$NWF" \
    --agents-per-workflow 5 \
    --tier0-len 512 \
    --tier1-len 1024 \
    --tier2-len 512 \
    --suffix-len 64 \
    --output-len 64 \
    --num-rounds 5 \
    --warmup-rounds 1 \
    >> "$BENCH_LOG" 2>&1

kill $SERVER_PID 2>/dev/null || true
sleep 10
pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 30

# Configuration 2: priority_nopf (Priority without Prefetch)
cfg="priority_nopf"
PORT=$((PORT + 10))
echo "=========================================="
echo "Config: $cfg (port=$PORT)"
echo "=========================================="

LOG_FILE="$LOG_DIR/server_${cfg}.log"
"$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-total-tokens 90000 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 8192 \
    --radix-eviction-policy priority \
    --enable-hierarchical-cache \
    --hicache-ratio 2.5 \
    --hicache-write-policy write_back \
    --hicache-io-backend direct \
    --enable-cache-report \
    --disable-cuda-graph \
    --disable-pie \
    --skip-server-warmup \
    --log-level info \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server PID=$SERVER_PID"

for i in $(seq 1 60); do
    curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health_generate" > /dev/null 2>&1 && break
    [ $((i % 12)) -eq 0 ] && echo "  ...waiting..."
    sleep 5
done

BASELINE_JSON=""
for candidate in "$(ls -t "$RESULT_DIR"/mwf_lru_nocache_*${NWF}wf*.json 2>/dev/null | head -1)" \
                 "$(ls -t "$RESULT_DIR"/mwf_lru_nocache_*_${NWF}wf*.json 2>/dev/null | head -1)"; do
    if [[ -n "$candidate" ]] && [[ -f "$candidate" ]]; then
        BASELINE_JSON="--baseline-json $candidate"
        break
    fi
done

BENCH_LOG="$LOG_DIR/bench_${cfg}.log"
"$PYTHON" -B bench_multi_workflow.py \
    --config "$cfg" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --agents-seed 42 \
    --seed 42 \
    --output-dir "$RESULT_DIR" \
    --num-workflows "$NWF" \
    --agents-per-workflow 5 \
    --tier0-len 512 \
    --tier1-len 1024 \
    --tier2-len 512 \
    --suffix-len 64 \
    --output-len 64 \
    --num-rounds 5 \
    --warmup-rounds 1 \
    $BASELINE_JSON \
    >> "$BENCH_LOG" 2>&1

kill $SERVER_PID 2>/dev/null || true
sleep 10
pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 30

# Configuration 3: kvflow (Priority + Prefetch - NEGATIVE interaction)
cfg="kvflow"
PORT=$((PORT + 10))
echo "=========================================="
echo "Config: $cfg (port=$PORT)"
echo "=========================================="

LOG_FILE="$LOG_DIR/server_${cfg}.log"
"$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
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
    --skip-server-warmup \
    --log-level info \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server PID=$SERVER_PID"

for i in $(seq 1 60); do
    curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health_generate" > /dev/null 2>&1 && break
    [ $((i % 12)) -eq 0 ] && echo "  ...waiting..."
    sleep 5
done

BASELINE_JSON=""
for candidate in "$(ls -t "$RESULT_DIR"/mwf_lru_nocache_*${NWF}wf*.json 2>/dev/null | head -1)" \
                 "$(ls -t "$RESULT_DIR"/mwf_lru_nocache_*_${NWF}wf*.json 2>/dev/null | head -1)"; do
    if [[ -n "$candidate" ]] && [[ -f "$candidate" ]]; then
        BASELINE_JSON="--baseline-json $candidate"
        break
    fi
done

BENCH_LOG="$LOG_DIR/bench_${cfg}.log"
"$PYTHON" -B bench_multi_workflow.py \
    --config "$cfg" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --agents-seed 42 \
    --seed 42 \
    --output-dir "$RESULT_DIR" \
    --num-workflows "$NWF" \
    --agents-per-workflow 5 \
    --tier0-len 512 \
    --tier1-len 1024 \
    --tier2-len 512 \
    --suffix-len 64 \
    --output-len 64 \
    --num-rounds 5 \
    --warmup-rounds 1 \
    $BASELINE_JSON \
    >> "$BENCH_LOG" 2>&1

kill $SERVER_PID 2>/dev/null || true
sleep 10
pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 30

# Configuration 4: tiered (NEW: Tiered + Prefetch - should FIX the issue)
cfg="tiered"
PORT=$((PORT + 10))
echo "=========================================="
echo "Config: $cfg (port=$PORT)"
echo "=========================================="

LOG_FILE="$LOG_DIR/server_${cfg}.log"
"$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-total-tokens 90000 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 8192 \
    --radix-eviction-policy tiered \
    --enable-hierarchical-cache \
    --hicache-ratio 2.5 \
    --hicache-write-policy write_back \
    --enable-hicache-prefetch \
    --hicache-io-backend direct \
    --enable-cache-report \
    --disable-cuda-graph \
    --disable-pie \
    --skip-server-warmup \
    --log-level info \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server PID=$SERVER_PID"

for i in $(seq 1 60); do
    curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health_generate" > /dev/null 2>&1 && break
    [ $((i % 12)) -eq 0 ] && echo "  ...waiting..."
    sleep 5
done

BASELINE_JSON=""
for candidate in "$(ls -t "$RESULT_DIR"/mwf_lru_nocache_*${NWF}wf*.json 2>/dev/null | head -1)" \
                 "$(ls -t "$RESULT_DIR"/mwf_lru_nocache_*_${NWF}wf*.json 2>/dev/null | head -1)"; do
    if [[ -n "$candidate" ]] && [[ -f "$candidate" ]]; then
        BASELINE_JSON="--baseline-json $candidate"
        break
    fi
done

BENCH_LOG="$LOG_DIR/bench_${cfg}.log"
"$PYTHON" -B bench_multi_workflow.py \
    --config "$cfg" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --agents-seed 42 \
    --seed 42 \
    --output-dir "$RESULT_DIR" \
    --num-workflows "$NWF" \
    --agents-per-workflow 5 \
    --tier0-len 512 \
    --tier1-len 1024 \
    --tier2-len 512 \
    --suffix-len 64 \
    --output-len 64 \
    --num-rounds 5 \
    --warmup-rounds 1 \
    $BASELINE_JSON \
    >> "$BENCH_LOG" 2>&1

kill $SERVER_PID 2>/dev/null || true
sleep 10
pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 30

# Copy output to home (already done since OUT is in LOG_DIR)
echo "=========================================="
echo "Ablation V2 Complete"
echo "=========================================="
echo "Results: $RESULT_DIR"
ls -lh "$RESULT_DIR"/mwf_*.json 2>/dev/null | awk '{print "  "$NF}'

echo "Done at $(date)"
