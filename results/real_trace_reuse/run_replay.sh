#!/usr/bin/env bash
# Start sglang-kvflow server, wait for ready, run the SWE-bench trace
# replay, then shut down the server.
set -euo pipefail

PROJECT_ROOT="/home/gfy/CodeMAS_Project/sglang-kvflow"
LOG="$PROJECT_ROOT/results/real_trace_reuse/data/replay_sglang.log"
PYTHON_BIN="/home/gfy/.conda/envs/sglang-kvflow/bin/python"
MODEL="/home/gfy/models/Qwen2.5-3B-Instruct"
PORT=31083

echo "[run_replay] starting server, log: $LOG"
SGLANG_LOSSY_FUZZY_MATCH=1 SGLANG_LOSSY_SKIP_TOKEN_CHECK=1 \
  $PYTHON_BIN -m sglang.launch_server \
  --model-path "$MODEL" \
  --port $PORT \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --max-total-tokens 32768 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --radix-eviction-policy priority \
  --enable-hierarchical-cache \
  --hicache-ratio 1.5 \
  --disable-cuda-graph \
  --log-level info \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "[run_replay] server PID: $SERVER_PID"

cleanup() {
  echo "[run_replay] killing server PID $SERVER_PID"
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "[run_replay] waiting for server (up to 240s)..."
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/v1/models | grep -q 200; then
    echo "[run_replay] server up"
    break
  fi
  sleep 4
done

if ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/v1/models | grep -q 200; then
  echo "[run_replay] FATAL: server didn't start in 240s"
  exit 1
fi

# Run the replay
$PYTHON_BIN "$PROJECT_ROOT/results/real_trace_reuse/replay_server.py" \
  --traces "$PROJECT_ROOT/results/real_trace_reuse/data/swe_bench_traces.jsonl" \
  --out "$PROJECT_ROOT/results/real_trace_reuse/data/replay_log.jsonl" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --request-timeout 60 \
  2>&1

# Aggregate
$PYTHON_BIN "$PROJECT_ROOT/results/real_trace_reuse/aggregate.py" \
  --log "$PROJECT_ROOT/results/real_trace_reuse/data/replay_log.jsonl" \
  2>&1
