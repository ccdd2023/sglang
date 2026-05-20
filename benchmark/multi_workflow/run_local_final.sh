#!/bin/bash
# KVFlow Local Storage Test
# All output written to files, not stdout (avoid shared FS I/O)

WORK_DIR="/tmp/kvflow-$$"
mkdir -p "$WORK_DIR"

# Redirect ALL output to log file
exec > "$WORK_DIR/main.log" 2>&1

echo "[$(date)] ===== Job Started ====="

# Copy conda env
echo "[$(date)] Copying conda env..."
cp -rL /home/comp/25480812/.conda/envs/sglang-kvflow "$WORK_DIR/env" 2>&1
echo "[$(date)] Conda env copied"

# Copy sglang
echo "[$(date)] Copying sglang..."
cp -rL /home/comp/25480812/CodeMAS_Project/sglang-kvflow "$WORK_DIR/sglang" 2>&1
echo "[$(date)] sglang copied"

# Copy model
echo "[$(date)] Copying model..."
cp -rL /home/comp/25480812/models/hub/models--Qwen--Qwen3-1.7B "$WORK_DIR/model" 2>&1
echo "[$(date)] Model copied"

# Set environment
export PATH="$WORK_DIR/env/bin:$PATH"
export PYTHONPATH="$WORK_DIR/sglang/python"

echo "[$(date)] Testing Python..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo "[$(date)] Launching sglang server..."
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
echo "[$(date)] Server PID: $SERVER_PID"

echo "[$(date)] Waiting for server..."
for i in {1..120}; do
    if curl -sf --noproxy '*' --max-time 1 "http://127.0.0.1:30001/health_generate" > /dev/null 2>&1; then
        echo "[$(date)] SUCCESS: Server ready after ${i}s!"
        exit 0
    fi
    if [ $((i % 20)) -eq 0 ]; then
        echo "[$(date)] Still waiting... ${i}s"
    fi
    sleep 1
done

echo "[$(date)] FAIL: Server failed"
tail -50 "$WORK_DIR/server.log"
exit 1
