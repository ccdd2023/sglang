#!/bin/bash
# Simple smoke test
#SBATCH --job-name=kvflow-smoke
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=hkbugpusrv08,hkbugpusrv15
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32

set -uo pipefail

LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-smoke-test"
mkdir -p "$LOG_DIR"

# Redirect to both /tmp and file
exec > >(tee "$LOG_DIR/slurm-${SLURM_JOB_ID}.out") 2>&1

echo "=========================================="
echo "KVFlow Smoke Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# Environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
for cuda_ver in cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1; do
    if [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/$cuda_ver"
        break
    fi
done
export PATH="$CUDA_HOME/bin:$PATH"
CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}:/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python"
PYTHON="$CONDA_ENV_PATH/bin/python"

echo ""
echo "=== Environment ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA: $($CUDA_HOME/bin/nvcc --version | grep release)"

echo ""
echo "=== Test 1: Python ==="
$PYTHON --version

echo ""
echo "=== Test 2: SGLang ==="
$PYTHON -c "import sglang; print(f'SGLang: {sglang.__version__}')"

echo ""
echo "=== Test 3: PyTorch CUDA ==="
$PYTHON -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "=== Test 4: SGLang Server ==="
MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
PORT=30500

# Kill existing
lsof -ti :$PORT 2>/dev/null | xargs -r kill -9 || true
sleep 3

echo "Starting server..."
$PYTHON -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 30000 \
    --disable-cuda-graph \
    --disable-pie \
    --skip-server-warmup \
    --log-level info > "$LOG_DIR/server_${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server (max 300s)
for i in $(seq 1 60); do
    if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health_generate" > /dev/null 2>&1; then
        echo "Server ready after $((i*5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died!"
        tail -20 "$LOG_DIR/server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
done

# Test inference
echo ""
echo "=== Test 5: Inference ==="
RESULT=$(curl -sf --noproxy '*' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}')

if echo "$RESULT" | grep -q "content"; then
    echo "Inference OK"
else
    echo "Inference response: $RESULT"
fi

# Cleanup
kill $SERVER_PID 2>/dev/null || true
lsof -ti :$PORT 2>/dev/null | xargs -r kill -9 || true

echo ""
echo "=========================================="
echo "Smoke Test Complete!"
echo "=========================================="
echo "End: $(date)"
