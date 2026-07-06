#!/bin/bash
# FAIR-MEASUREMENT / KVCOMM regime — L2 WHOLE-SLOT GENERAL BASELINE.
# Same byte-exact + RoPE-rotation + C2-direct gap-zeroing mechanism as L4+C2,
# but with WHOLE-SLOT granularity (SGLANG_CHUNK_COARSE=1 → 1 "module" chunk per
# slot) instead of per-AST-anchor chunking. This is the general KVCOMM baseline
# (no AST): the ONLY difference vs run_l4_c2.sh is coarse vs per-anchor
# chunking, so the L4-vs-L2 delta isolates the AST-chunking contribution.
# MiniLM L3 + offset-gate OFF (wrong regime). Position-shift, no vary-code.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_CHUNKED_PLACEHOLDER_KNN=1
export SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1
export SGLANG_CHUNK_COARSE=1
export SGLANG_CACHEBLEND_CHUNK=1
export SGLANG_CACHEBLEND_DIRECT=1
export SGLANG_CACHEBLEND_BATCH=1
export SGLANG_CACHEBLEND_COMPACT=1
export SGLANG_CACHEBLEND_OFFMAP=1
export SGLANG_CACHEBLEND_MAX_CACHED_RATIO=0.95
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/l2_coarse \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-tasks 4 --agent-count 5 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 2
