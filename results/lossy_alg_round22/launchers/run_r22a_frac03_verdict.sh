#!/bin/bash
# R22a — R19 + FRAC=0.30 (skip 30% largest chunks) under verdict task-mode
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_CHUNKED_PLACEHOLDER_KNN=1
export SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1
export SGLANG_CHUNK_TOPLEVEL=1
export SGLANG_CHUNK_COARSE=0
export SGLANG_CACHEBLEND_CHUNK=1
export SGLANG_CACHEBLEND_BATCH=1
export SGLANG_CACHEBLEND_COMPACT=0
export SGLANG_CACHEBLEND_MULTI_SLOT=1
export SGLANG_CACHEBLEND_MULTI_SLOT_MAX_GAP=256
export SGLANG_CACHEBLEND_OFFMAP=1
export SGLANG_CACHEBLEND_MAX_CACHED_RATIO=0.95
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case_v4
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=1
export SGLANG_PRECOMPUTE_SELECTIVE_REFRESH_FRAC=0.30    # R22a: was 0.25
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/lossy_alg_round22/r22a_frac03 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --mem-fraction-static 0.85 --max-total-tokens 65536 \
  --max-tasks 5 --agent-count 5 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 6 \
  --precompute-kv-dir results/codebase_kv/pandas_5case_v4 \
  --precompute-host-size-gb 2 \
  --precompute-canonical-prefix \
  --include-source-with-precompute \
  --disable-hierarchical-cache \
  --task-mode verdict
