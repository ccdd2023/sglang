#!/usr/bin/env bash
# 500-case AgentTemplateKV speedup run wrapper.
#
# The SGLang server load + token-shape compile/capture is slow on RTX 4090
# (3-5 min for Qwen2.5-7B at 16k-bucket, see sglang_server.log timestamps
# from the 100-case run on 2026-06-03).  The case loop then runs ~3.8 s/case.
#
# This wrapper detaches the Python process with setsid+nohup+disown so the
# Claude foreground turn can exit without killing it.  A heartbeat file
# (heartbeat.log) is updated every 60 s while the case loop is in progress.
#
# Usage:  bash run_500.sh [extra args to bench_coding_kvflow_prefetch.py]
#
# Outputs (all under --out-dir):
#   - sglang_server.log       SGLang stdout/stderr
#   - nohup.out               Python stdout/stderr (incl. [case] <id> done)
#   - wrapper.pid             PID of the benchmark process
#   - heartbeat.log           "alive" with epoch time every 60 s
#   - summary.json, prefetch_summary.json, prefetch_table.csv, PREFETCH_REPORT.md
#     are written by the Python script at the END of the run (all-or-nothing).
#
# Resumability: the Python script writes results only at the end.  If the
# process is killed mid-run, restart this wrapper from scratch.  The 500-case
# run is expected to take ~9 h; smoke-3 takes ~5 min, smoke-30 takes ~30 min.

set -u

PROJECT=/home/gfy/CodeMAS_Project/sglang-kvflow
PYTHON=/home/gfy/.conda/envs/sglang-kvflow/bin/python
OUT_DIR="$PROJECT/results/coding_kvflow_prefetch/qwen2_5_7b_500"
DATASET="$PROJECT/results/repo_level_datasets/swe_verified_500_instances.json"
MANIFEST="$PROJECT/results/repo_level_datasets/manifest_500.json"

cd "$PROJECT" || { echo "cd failed"; exit 1; }

# Heartbeat in the background.  Updates heartbeat.log with the current epoch
# time every 60 s.  Dies naturally when the parent exits.
( while true; do
    date +%s >> "$OUT_DIR/heartbeat.log"
    sleep 60
  done ) &
HEARTBEAT_PID=$!
echo "heartbeat pid: $HEARTBEAT_PID"

# Detach the benchmark process.  setsid + nohup + disown so the Claude
# foreground turn can return without killing the run.
nohup "$PYTHON" -m benchmark.multi_workflow.bench_coding_kvflow_prefetch \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --max-cases 500 \
    --files-per-case 2 \
    --disable-hierarchical-cache \
    --out-dir "$OUT_DIR" \
    --port 30010 \
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
echo "tail -f $OUT_DIR/nohup.out  # to watch live"
echo "tail -f $OUT_DIR/heartbeat.log  # to verify process is alive"
echo "kill $BENCH_PID  # to stop the run"
