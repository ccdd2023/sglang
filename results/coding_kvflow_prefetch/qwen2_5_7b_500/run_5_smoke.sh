#!/usr/bin/env bash
# Minimal repro for the case-5 scheduler hang.  Slices the 500-case manifest
# at the chosen start-index.  Default start-index=0 (cases 1-5 of 500, with
# astropy__astropy-13453 at position 4 — the previously-failing case).
#
# Pre-fix: hangs at case 1 (astropy-13453) when start-index=0 (after 4 prior
# cases built up the protected-anchor state).
# Post-fix: completes all 5 cases.
#
# Usage:  bash run_5_smoke.sh [port] [out-dir-name] [start-index]
# Example: bash run_5_smoke.sh 30013 qwen2_5_7b_500_smoke5_pre 0
#          bash run_5_smoke.sh 30014 qwen2_5_7b_500_smoke5_post 0
#          bash run_5_smoke.sh 30015 qwen2_5_7b_500_smoke5 4   # (skips cases 1-3)
#
# IMPORTANT: do NOT send any other HTTP request to port $PORT while this
# run is in progress -- a concurrent probe (e.g. curl) can crash the
# sglang::scheduler subprocess and abort the run.

set -u

PROJECT=/home/gfy/CodeMAS_Project/sglang-kvflow
PYTHON=/home/gfy/.conda/envs/sglang-kvflow/bin/python

PORT="${1:-30013}"
OUT_NAME="${2:-qwen2_5_7b_500_smoke5_pre}"
START_INDEX="${3:-0}"
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
    --max-cases 5 \
    --start-index "$START_INDEX" \
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
echo "start-index: $START_INDEX (cases 1-5 of 500 = astropy-12907..13453)"
echo "tail -f $OUT_DIR/nohup.out"
echo "tail -f $OUT_DIR/heartbeat.log"
echo "kill $BENCH_PID  # to stop"
echo "WARNING: do NOT send any other HTTP request to port $PORT during the run"
