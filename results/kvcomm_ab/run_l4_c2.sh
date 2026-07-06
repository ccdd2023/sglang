#!/bin/bash
# FAIR-MEASUREMENT / KVCOMM regime (Step 2) — L4 AST-chunk + C2 CODE-AWARE.
# Byte-exact AST-chunk selection + RoPE rotation + gap-prefill (KVCOMM + AST).
# MiniLM L3 + offset-gate OFF (wrong regime). L2 stays ON (general path composes).
# Position-shift, no vary-code.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_CHUNKED_PLACEHOLDER_KNN=1
export SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1
export SGLANG_CACHEBLEND_CHUNK=1
export SGLANG_CACHEBLEND_DIRECT=1
export SGLANG_CACHEBLEND_BATCH=1
export SGLANG_CHUNK_TOPLEVEL=1
export SGLANG_CACHEBLEND_COMPACT=1
export SGLANG_CACHEBLEND_OFFMAP=1
export SGLANG_CACHEBLEND_MAX_CACHED_RATIO=0.95
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/l4_c2 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-tasks 4 --agent-count 5 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 2
