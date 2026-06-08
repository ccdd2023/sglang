#!/usr/bin/env bash
# Smoke-test wrapper for the 500-case harness.  Same shape as run_500.sh
# but uses a smaller --max-cases.  Default: 3 cases into
# qwen2_5_7b_500_smoke3/ on port 30011.
#
# Always feeds the 500-case manifest/instances (--max-cases truncates the
# manifest at run time) so we never depend on a smaller manifest file.
#
# Usage:  bash run_smoke.sh [max-cases] [out-dir-name] [port]
# Example: bash run_smoke.sh 3 qwen2_5_7b_500_smoke3 30011
#          bash run_smoke.sh 30 qwen2_5_7b_500_smoke30 30012

set -u

PROJECT=/home/gfy/CodeMAS_Project/sglang-kvflow
PYTHON=/home/gfy/.conda/envs/sglang-kvflow/bin/python

MAX_CASES="${1:-3}"
OUT_NAME="${2:-qwen2_5_7b_500_smoke3}"
PORT="${3:-30011}"
shift 3 2>/dev/null || true

OUT_DIR="$PROJECT/results/coding_kvflow_prefetch/$OUT_NAME"
DATASET="$PROJECT/results/repo_level_datasets/swe_verified_500_instances.json"
MANIFEST="$PROJECT/results/repo_level_datasets/manifest_500.json"

cd "$PROJECT" || { echo "cd failed"; exit 1; }

( while true; do
    date +%s >> "$OUT_DIR/heartbeat.log"
    sleep 30
  done ) &
HEARTBEAT_PID=$!
echo "heartbeat pid: $HEARTBEAT_PID"

nohup "$PYTHON" -m benchmark.multi_workflow.bench_coding_kvflow_prefetch \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --max-cases "$MAX_CASES" \
    --files-per-case 2 \
    --disable-hierarchical-cache \
    --out-dir "$OUT_DIR" \
    --port "$PORT" \
    --server-timeout 300 \
    --eval-timeout 1200 \
    --max-total-tokens 65536 \
    --mem-fraction-static 0.78 \
    "$@" \
    > "$OUT_DIR/nohup.out" 2>&1 < /dev/null &
BENCH_PID=$!
echo "$BENCH_PID" > "$OUT_DIR/wrapper.pid"
disown $BENCH_PID 2>/dev/null || true
disown $HEARTBEAT_PID 2>/dev/null || true

echo "benchmark pid: $BENCH_PID"
echo "tail -f $OUT_DIR/nohup.out"
echo "tail -f $OUT_DIR/heartbeat.log"
echo "kill $BENCH_PID  # to stop"
