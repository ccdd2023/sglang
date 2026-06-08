#!/usr/bin/env bash
# 200-case AgentTemplateKV speedup run wrapper.  Mid-size alternative to the
# 500-case run; estimated wall-clock ~7-8 h on RTX 4090 24 GB.
#
# PRE-FIX RUN: pre-fix engine hung at case 5 (astropy-13453).  Output dir
#   qwen2_5_7b_500_200_prefix/ has 4-case partial data + crash artifacts.
# POST-FIX RUN: case-5 hang fix in radix_cache.py (5 hunks) should let this
#   complete all 200 cases.  Output dir qwen2_5_7b_500_200_postfix/.
#
# IMPORTANT: do NOT send any other HTTP request to port $PORT while this
# run is in progress -- a concurrent probe (e.g. curl) can crash the
# sglang::scheduler subprocess and abort the run.

set -u

PROJECT=/home/gfy/CodeMAS_Project/sglang-kvflow
PYTHON=/home/gfy/.conda/envs/sglang-kvflow/bin/python
OUT_DIR="$PROJECT/results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix"
DATASET="$PROJECT/results/repo_level_datasets/swe_verified_200_instances.json"
MANIFEST="$PROJECT/results/repo_level_datasets/manifest_200.json"
PORT="${1:-30014}"

cd "$PROJECT" || { echo "cd failed"; exit 1; }

( while true; do
    date +%s >> "$OUT_DIR/heartbeat.log"
    sleep 60
  done ) &
HEARTBEAT_PID=$!
echo "heartbeat pid: $HEARTBEAT_PID"

nohup "$PYTHON" -m benchmark.multi_workflow.bench_coding_kvflow_prefetch \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --max-cases 200 \
    --files-per-case 2 \
    --disable-hierarchical-cache \
    --out-dir "$OUT_DIR" \
    --port "$PORT" \
    --server-timeout 300 \
    --eval-timeout 1200 \
    --max-total-tokens 65536 \
    --mem-fraction-static 0.78 \
    > "$OUT_DIR/nohup.out" 2>&1 < /dev/null &
BENCH_PID=$!
echo "$BENCH_PID" > "$OUT_DIR/wrapper.pid"
disown $BENCH_PID 2>/dev/null || true
disown $HEARTBEAT_PID 2>/dev/null || true

echo "benchmark pid: $BENCH_PID"
echo "tail -f $OUT_DIR/nohup.out"
echo "tail -f $OUT_DIR/heartbeat.log"
echo "kill $BENCH_PID  # to stop"
echo "WARNING: do NOT send any other HTTP request to port $PORT during the run"
