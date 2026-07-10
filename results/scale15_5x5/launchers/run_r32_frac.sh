#!/bin/bash
# Scale-15 ablation: R32 uniform-FRAC sweep launcher (parameterized).
# Usage: run_r32_frac.sh <frac> <suffix>
#   e.g. run_r32_frac.sh 0.26 f026   -> out-dir results/scale15_5x5/r32_f026
#        run_r32_frac.sh 0.15 f015
# frac*=0.261 matches node-kind interface B (equal-budget vertical slice).
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
FRAC="${1:?usage: $0 <frac> <suffix>}"
SUF="${2:?usage: $0 <frac> <suffix>}"
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
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_15case_v1
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=1
export SGLANG_PRECOMPUTE_SELECTIVE_REFRESH_FRAC=0.25
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
unset SGLANG_AST_REUSE_TYPES SGLANG_PY_IMPORTS_PRELUDE SGLANG_USE_SIG_GATED_RECOMPUTE SGLANG_CHUNK_TYPE_AWARE_FRAC
unset SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_LATE SGLANG_CHUNK_HEAD_RECOMPUTE_EARLY_N
unset SGLANG_CHUNK_HEAD_RECOMPUTE_NODE_KIND SGLANG_CHUNK_NODE_KIND_BOUNDARY
export SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC="$FRAC"
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000_diverse15/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir "results/scale15_5x5/r32_${SUF}" \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --mem-fraction-static 0.72 --max-total-tokens 16384 \
  --max-tasks 15 --agent-count 5 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 1 \
  --precompute-kv-dir results/codebase_kv/pandas_15case_v1 \
  --precompute-host-size-gb 2 \
  --precompute-canonical-prefix \
  --include-source-with-precompute \
  --disable-hierarchical-cache \
  --task-mode verdict
