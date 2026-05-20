#!/bin/bash
#SBATCH --job-name=kvflow-test
#SBATCH --output=/tmp/kvflow-test/slurm-test.out
#SBATCH --error=/tmp/kvflow-test/slurm-test.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

# Create temp directory on local SSD
mkdir -p /tmp/kvflow-test
cd /tmp/kvflow-test

echo "===== Test Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Temp dir: $(pwd)"

# Check GPU
echo ""
echo "===== GPU Check ====="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

# Copy conda environment to local storage
echo ""
echo "===== Setting up local Python environment ====="
LOCAL_ENV="/tmp/kvflow-test/env"
if [[ ! -d "$LOCAL_ENV" ]]; then
    echo "Copying conda environment..."
    cp -rL /home/comp/25480812/.conda/envs/sglang-kvflow "$LOCAL_ENV" 2>&1 | tail -3
fi
export PATH="$LOCAL_ENV/bin:$PATH"
PYTHON_BIN="$LOCAL_ENV/bin/python"

echo "Python: $($PYTHON_BIN --version)"
echo "PyTorch: $($PYTHON_BIN -c "import torch; print(torch.__version__)")"

# Set PYTHONPATH to local sglang
export PYTHONPATH="/tmp/kvflow-test/sglang-kvflow/python"
if [[ ! -d "$LOCAL_ENV/sglang-kvflow" ]]; then
    echo "Copying sglang-kvflow..."
    cp -rL /home/comp/25480812/CodeMAS_Project/sglang-kvflow /tmp/kvflow-test/
fi

# Copy model to local
MODEL_DIR="/tmp/kvflow-test/Qwen3-1.7B"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Copying model..."
    cp -rL /home/comp/25480812/models/hub/models--Qwen--Qwen3-1.7B "$MODEL_DIR"
fi

echo ""
echo "===== Launching Server ====="
$PYTHON_BIN -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --port 30001 \
    --tp-size 1 \
    --mem-fraction-static 0.7 \
    --max-total-tokens 30000 \
    --disable-cuda-graph \
    --skip-server-warmup \
    --log-level warning \
    > /tmp/kvflow-test/server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo ""
echo "===== Waiting for Server (120s) ====="
for i in {1..120}; do
    if curl -sf --noproxy '*' --max-time 1 "http://127.0.0.1:30001/health_generate" > /dev/null 2>&1; then
        echo "Server ready after ${i}s!"
        exit 0
    fi
    if [ $((i % 20)) -eq 0 ]; then
        echo "  Still waiting... ${i}s"
    fi
    sleep 1
done

echo "Server failed to start. Checking log:"
tail -50 /tmp/kvflow-test/server.log
exit 1
