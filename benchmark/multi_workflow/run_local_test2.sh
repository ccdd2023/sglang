#!/bin/bash
#SBATCH --job-name=kvflow-test
#SBATCH --output=/tmp/kvflow-test/slurm.out
#SBATCH --error=/tmp/kvflow-test/slurm.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

# Use local storage
WORK_DIR="/tmp/kvflow-test"
mkdir -p "$WORK_DIR"

# Log to file (avoid print to shared FS)
log() { echo "[$(date '+%H:%M:%S')] $*" >> "$WORK_DIR/status.log"; }

log "===== Job Started ====="
log "Job ID: $SLURM_JOB_ID"
log "Node: $SLURM_JOB_NODELIST"

# Check GPU
log "===== GPU Check ====="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader >> "$WORK_DIR/status.log" 2>&1

# Copy conda env to local (avoid shared FS I/O)
log "===== Copying conda env to local ====="
LOCAL_ENV="$WORK_DIR/env"
if [[ ! -d "$LOCAL_ENV" ]]; then
    cp -rL /home/comp/25480812/.conda/envs/sglang-kvflow "$LOCAL_ENV" 2>&1 >> "$WORK_DIR/copy.log"
fi

# Copy sglang-kvflow to local
log "===== Copying sglang-kvflow to local ====="
LOCAL_SGLANG="$WORK_DIR/sglang"
if [[ ! -d "$LOCAL_SGLANG" ]]; then
    cp -rL /home/comp/25480812/CodeMAS_Project/sglang-kvflow "$LOCAL_SGLANG" 2>&1 >> "$WORK_DIR/copy.log"
fi

# Copy model to local
log "===== Copying model to local ====="
LOCAL_MODEL="$WORK_DIR/model"
if [[ ! -d "$LOCAL_MODEL" ]]; then
    cp -rL /home/comp/25480812/models/hub/models--Qwen--Qwen3-1.7B "$LOCAL_MODEL" 2>&1 >> "$WORK_DIR/copy.log"
fi

log "===== Setup complete ====="

# Set up environment
export PATH="$LOCAL_ENV/bin:$PATH"
export PYTHONPATH="$LOCAL_SGLANG/python"

# Verify Python works
log "Python version: $(python --version 2>&1)"
log "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
log "CUDA version: $(python -c 'import torch; print(torch.version.cuda)' 2>&1)"

# Launch server
log "===== Launching Server ====="
python -m sglang.launch_server \
    --model-path "$LOCAL_MODEL" \
    --port 30001 \
    --tp-size 1 \
    --mem-fraction-static 0.7 \
    --max-total-tokens 30000 \
    --disable-cuda-graph \
    --skip-server-warmup \
    --log-level info \
    > "$WORK_DIR/server.log" 2>&1 &

SERVER_PID=$!
log "Server PID: $SERVER_PID"

# Wait for server
log "===== Waiting for Server (120s) ====="
for i in {1..120}; do
    if curl -sf --noproxy '*' --max-time 1 "http://127.0.0.1:30001/health_generate" > /dev/null 2>&1; then
        log "Server ready after ${i}s!"
        log "===== SUCCESS ====="
        exit 0
    fi
    if [ $((i % 20)) -eq 0 ]; then
        log "  Still waiting... ${i}s"
    fi
    sleep 1
done

log "Server failed to start"
log "===== FAIL ====="
log "Last 50 lines of server log:"
tail -50 "$WORK_DIR/server.log" >> "$WORK_DIR/status.log"
exit 1
