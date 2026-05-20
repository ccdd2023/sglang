#!/bin/bash
# Test SGLang server startup - Minimal version
#SBATCH --job-name=sglang-test
#SBATCH --time=00:15:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# All output to local /tmp first
OUT="/tmp/slurm-${SLURM_JOB_ID}.out"
exec >"$OUT" 2>&1

echo "=== SGLang Server Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Setup
CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python:$PYTHONPATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
PYTHON="$CONDA_ENV_PATH/bin/python"

echo ""
echo "=== Check 1: Conda env ==="
if [[ -d "$CONDA_ENV_PATH" ]]; then
    echo "OK: Conda env exists"
else
    echo "FAIL: Conda env not found"
    exit 1
fi

echo ""
echo "=== Check 2: Python and SGLang ==="
if $PYTHON -c "import sglang; print(f'SGLang OK: {sglang.__version__}')" 2>&1; then
    echo "OK: SGLang import successful"
else
    echo "FAIL: SGLang import failed"
    exit 1
fi

echo ""
echo "=== Check 3: Model ==="
if [[ -d "$MODEL_PATH" ]]; then
    echo "OK: Model exists"
else
    echo "FAIL: Model not found"
    exit 1
fi

echo ""
echo "=== Starting SGLang server ==="
LOGFILE="/tmp/server_${SLURM_JOB_ID}.log"

$PYTHON -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port 30500 \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 30000 \
    --disable-cuda-graph \
    --disable-pie \
    --skip-server-warmup \
    --log-level info \
    > "$LOGFILE" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Log file: $LOGFILE"

# Wait for server
echo ""
echo "=== Waiting for server (max 300s) ==="
for i in $(seq 1 60); do
    if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:30500/health_generate" > /dev/null 2>&1; then
        echo "SUCCESS: Server ready after $((i*5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "FAIL: Server died"
        echo "=== Last 30 lines of log ==="
        tail -30 "$LOGFILE" 2>/dev/null || echo "No log"
        exit 1
    fi
    sleep 5
done

# Check if ready
if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:30500/health_generate" > /dev/null 2>&1; then
    echo "=== Server is healthy ==="
else
    echo "FAIL: Server did not become ready"
    tail -50 "$LOGFILE" 2>/dev/null || echo "No log"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Cleanup
echo ""
echo "=== Cleanup ==="
kill $SERVER_PID 2>/dev/null || true
lsof -ti :30500 2>/dev/null | xargs -r kill -9 2>/dev/null || true

echo ""
echo "=== Test Complete ==="
echo "End time: $(date)"

# Copy logs to home
cp "$OUT" "/home/comp/25480812/CodeMAS_Project/logs/kvflow-smoke-test/slurm-${SLURM_JOB_ID}.out" 2>/dev/null || true
cp "$LOGFILE" "/home/comp/25480812/CodeMAS_Project/logs/kvflow-smoke-test/server_${SLURM_JOB_ID}.log" 2>/dev/null || true
