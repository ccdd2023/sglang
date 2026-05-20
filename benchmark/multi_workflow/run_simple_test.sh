#!/bin/bash
#SBATCH --job-name=kvflow-test
#SBATCH --output=/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/slurm-test.out
#SBATCH --error=/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/slurm-test.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

echo "===== Test Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"

# Check GPU
echo ""
echo "===== GPU Check ====="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

# Set environment
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python"
PYTHON_BIN="/home/comp/25480812/.conda/envs/sglang-kvflow/bin/python"
export TORCHINDUCTOR_DISABLE_FLEX_ATTENTION=1

echo ""
echo "===== Python Check ====="
echo "Python: $($PYTHON_BIN --version)"
echo "PyTorch: $($PYTHON_BIN -c "import torch; print(torch.__version__)")"
echo "CUDA: $($PYTHON_BIN -c "import torch; print(torch.version.cuda)")"

echo ""
echo "===== sglang Check ====="
$PYTHON_BIN -c "import sglang; print('sglang version:', sglang.__version__)" || echo "sglang import failed"

echo ""
echo "===== Launching Server ====="
$PYTHON_BIN -m sglang.launch_server \
    --model-path /home/comp/25480812/models/hub/models--Qwen--Qwen3-1.7B \
    --port 30001 \
    --tp-size 1 \
    --mem-fraction-static 0.7 \
    --max-total-tokens 30000 \
    --disable-cuda-graph \
    --skip-server-warmup \
    --log-level warning \
    > /home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/server-test.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo ""
echo "===== Waiting for Server (60s) ====="
for i in {1..60}; do
    if curl -sf --noproxy '*' --max-time 1 "http://127.0.0.1:30001/health_generate" > /dev/null 2>&1; then
        echo "Server ready after ${i}s!"
        exit 0
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... ${i}s"
    fi
    sleep 1
done

echo "Server failed to start. Checking log:"
tail -30 /home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/server-test.log
exit 1
