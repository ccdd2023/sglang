#!/bin/bash
# R40 Phase 1: Same R38b config as yesterday's baseline_ours_r38b_5x5_verdict
# but with R40 TTFT breakdown instrumentation. Goal: confirm 8 new CSV columns
# populate and sum ~= ttft_ms.
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
export SGLANG_PRECOMPUTE_SELECTIVE_REFRESH_FRAC=0.25
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
unset SGLANG_AST_REUSE_TYPES
unset SGLANG_PY_IMPORTS_PRELUDE
unset SGLANG_USE_SIG_GATED_RECOMPUTE
unset SGLANG_CHUNK_TYPE_AWARE_FRAC
export SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY=0.60
export SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_LATE=0.15
export SGLANG_CHUNK_HEAD_RECOMPUTE_EARLY_N=2
unset SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/baseline_ours_r38b_5x5_verdict_r40_phase1 \
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