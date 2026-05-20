#!/bin/bash
# =============================================================================
# Local Test Script - SGLang KVFlow Server
# Tests the sglang-kvflow server with local RTX 4090 setup
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="/home/gfy/.conda/envs/sglang-kvflow"
PYTHON_BIN="$CONDA_ENV/bin/python"
MODEL_PATH="${MODEL_PATH:-/home/gfy/models/Qwen2.5-3B-Instruct}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-/home/gfy/CodeMAS_Project/logs}"
EVICTION="${EVICTION:-priority}"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "SGLang KVFlow Local Test"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Eviction Policy: $EVICTION"
echo "=============================================="

# Kill any existing server on the port
echo "[INFO] Checking port $PORT..."
if pid=$(lsof -ti :"$PORT" 2>/dev/null); then
    echo "[INFO] Killing existing process on port $PORT (PID=$pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 5
fi

# Determine eviction policy and HiCache settings
case "$EVICTION" in
    lru)
        RADIX_POLICY="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 1.5 --hicache-write-policy write_through"
        ;;
    priority)
        RADIX_POLICY="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 1.5 --hicache-write-policy write_through"
        ;;
    priority_wb)
        RADIX_POLICY="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_back"
        ;;
    lru_wb)
        RADIX_POLICY="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_back"
        ;;
    *)
        echo "[ERROR] Unknown eviction mode: $EVICTION"
        echo "Valid modes: lru, priority, priority_wb, lru_wb"
        exit 1
        ;;
esac

SERVER_LOG="$LOG_DIR/server_${EVICTION}_${PORT}.log"

echo "[INFO] Starting SGLang server..."
echo "[INFO] Log file: $SERVER_LOG"
echo "[INFO] Radix policy: $RADIX_POLICY"
echo "[INFO] HiCache flags: $HICACHE_FLAGS"

export PYTHONPATH="/home/gfy/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-total-tokens 32768 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 8192 \
    --radix-eviction-policy "$RADIX_POLICY" \
    $HICACHE_FLAGS \
    --enable-cache-report \
    --disable-cuda-graph \
    --log-level info \
    2>&1 | tee "$SERVER_LOG" &

SERVER_PID=$!
echo "[INFO] Server PID: $SERVER_PID"
echo "$SERVER_PID" > "$LOG_DIR/.server_${EVICTION}_${PORT}.pid"

# Wait for server to be ready
echo "[INFO] Waiting for server to be ready..."
MAX_WAIT=180
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health_generate" > /dev/null 2>&1; then
        echo "[SUCCESS] Server is ready after $((i*2))s!"
        echo ""
        echo "=============================================="
        echo "Server is running!"
        echo "  PID: $SERVER_PID"
        echo "  URL: http://127.0.0.1:$PORT"
        echo "  Eviction: $RADIX_POLICY"
        echo "=============================================="
        echo ""
        echo "To test the server, run:"
        echo "  curl -X POST http://127.0.0.1:$PORT/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 100}'"
        echo ""
        echo "To stop the server:"
        echo "  kill $SERVER_PID"
        exit 0
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "[INFO] Still waiting... ($((i*2))s / ${MAX_WAIT}s)"
    fi
    sleep 2
done

echo "[ERROR] Server failed to start within ${MAX_WAIT}s"
echo "[ERROR] Check log: $SERVER_LOG"
tail -50 "$SERVER_LOG" 2>/dev/null || true
exit 1
