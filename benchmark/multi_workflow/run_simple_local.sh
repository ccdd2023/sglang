#!/bin/bash
#SBATCH --job-name=kvflow-simple
#SBATCH --output=/tmp/kvflow-simple/slurm.out
#SBATCH --error=/tmp/kvflow-simple/slurm.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

WORK_DIR="/tmp/kvflow-simple"
mkdir -p "$WORK_DIR"

# Write to log file, not stdout
exec > "$WORK_DIR/main.log" 2>&1

echo "[START] Job started at $(date)"
echo "[START] Node: $SLURM_JOB_NODELIST"

# Copy conda env
echo "[COPY] Copying conda env..."
cp -rL /home/comp/25480812/.conda/envs/sglang-kvflow "$WORK_DIR/env" 2>&1
echo "[COPY] Done"

# Copy sglang
echo "[COPY] Copying sglang..."
cp -rL /home/comp/25480812/CodeMAS_Project/sglang-kvflow "$WORK_DIR/sglang" 2>&1
echo "[COPY] Done"

# Copy model
echo "[COPY] Copying model..."
cp -rL /home/comp/25480812/models/hub/models--Qwen--Qwen3-1.7B "$WORK_DIR/model" 2>&1
echo "[COPY] Done"

# Set environment
export PATH="$WORK_DIR/env/bin:$PATH"
export PYTHONPATH="$WORK_DIR/sglang/python"

echo "[TEST] Testing Python..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo "[LAUNCH] Starting server..."
python -m sglang.launch_server \
    --model-path "$WORK_DIR/model" \
    --port 30001 \
    --tp-size 1 \
    --mem-fraction-static 0.7 \
    --max-total-tokens 30000 \
    --disable-cuda-graph \
    --skip-server-warmup \
    --log-level info \
    > "$WORK_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "[LAUNCH] Server PID: $SERVER_PID"

echo "[WAIT] Waiting for server (120s)..."
for i in {1..120}; do
    if curl -sf --noproxy '*' --max-time 1 "http://127.0.0.1:30001/health_generate" > /dev/null 2>&1; then
        echo "[SUCCESS] Server ready after ${i}s!"
        exit 0
    fi
    if [ $((i % 20)) -eq 0 ]; then
        echo "[WAIT] Still waiting... ${i}s"
    fi
    sleep 1
done

echo "[FAIL] Server failed to start"
tail -30 "$WORK_DIR/server.log"
exit 1
