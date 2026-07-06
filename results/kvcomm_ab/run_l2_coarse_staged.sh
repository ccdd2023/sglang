#!/bin/bash
# FAIR-MEASUREMENT / KVCOMM regime — L2 whole-slot, STAGED gap-prefill (non-lossy gap).
# Same as run_l2_coarse.sh (whole-slot coarse chunking) but WITHOUT
# SGLANG_CACHEBLEND_DIRECT (which gap-ZEROES the header → destroys the task
# instruction → F1~0.1). Instead uses the staged CacheBlend path: prefill the
# leading gap with REAL KV (scheduler recompute-gap round), then copy the
# contiguous chunk run. The gap KV is exact; only the copied chunk KV is
# cross-context (the accuracy question this run answers).
# MiniLM L3 + offset-gate OFF. Position-shift, no vary-code.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_CHUNKED_PLACEHOLDER_KNN=1
export SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1
export SGLANG_CHUNK_COARSE=1
export SGLANG_CACHEBLEND_CHUNK=1
export SGLANG_CACHEBLEND_BATCH=1
export SGLANG_CACHEBLEND_COMPACT=1
export SGLANG_CACHEBLEND_OFFMAP=1
export SGLANG_CACHEBLEND_MAX_CACHED_RATIO=0.95
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/l2_coarse_staged \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-tasks 4 --agent-count 5 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 2
