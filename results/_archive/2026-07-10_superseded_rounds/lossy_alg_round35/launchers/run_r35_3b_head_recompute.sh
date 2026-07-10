#!/bin/bash
# R35 — TREATMENT: 3B-Instruct × 3 agents + head_recompute_30 + coarse chunks.
# Aggregate R26 (3B × 3 = 2.014× speedup) + R32 (head_recompute_30% = 41.7%
# failure-type agreement baseline). Hypothesis: both bars met simultaneously —
# speed wins from small-model 3B × 3 agents, accuracy wins from selective
# head recompute.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_CHUNKED_PLACEHOLDER_KNN=1
export SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1
export SGLANG_CHUNK_TOPLEVEL=1
# R26 baseline uses COARSE chunks for speed; combined with R32 head_recompute
# which is per-chunk-position logic. Both should be compatible.
export SGLANG_CHUNK_COARSE=1
export SGLANG_CACHEBLEND_CHUNK=1
export SGLANG_CACHEBLEND_BATCH=1
export SGLANG_CACHEBLEND_COMPACT=0
export SGLANG_CACHEBLEND_MULTI_SLOT=1
export SGLANG_CACHEBLEND_MULTI_SLOT_MAX_GAP=256
export SGLANG_CACHEBLEND_OFFMAP=1
export SGLANG_CACHEBLEND_MAX_CACHED_RATIO=0.95
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
# R26 used a separate precompute pool for the 3B model (different KV
# layout: 3 layers, 2 heads vs 7B's 28 layers, 4 heads).
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case_v6_verdict_3b
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=1
export SGLANG_PRECOMPUTE_SELECTIVE_REFRESH_FRAC=0.25
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
unset SGLANG_AST_REUSE_TYPES
unset SGLANG_PY_IMPORTS_PRELUDE
unset SGLANG_USE_SIG_GATED_RECOMPUTE
# R35 TREATMENT: R32 head recompute Pareto value
export SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC=0.30
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/lossy_alg_round35/r35_3b_head_recompute_verdict \
  --model /home/gfy/models/Qwen2.5-3B-Instruct \
  --mem-fraction-static 0.85 --max-total-tokens 65536 \
  --max-tasks 5 --agent-count 3 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 6 \
  --precompute-kv-dir results/codebase_kv/pandas_5case_v6_verdict_3b \
  --precompute-host-size-gb 2 \
  --precompute-canonical-prefix \
  --include-source-with-precompute \
  --disable-hierarchical-cache \
  --task-mode verdict