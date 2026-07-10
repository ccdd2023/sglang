#!/bin/bash
# Smoke test: node-kind interface-recompute on 2 cases x 2 agents.
# Verifies the path fires at runtime (node_kind_k_count > 0), chunks are
# copied (codeaware_reused_tokens > 0), and no crashes - before the full
# 8-config ablation. Throwaway.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0 SGLANG_L3_AST_GATE=0 SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_CHUNKED_PLACEHOLDER_KNN=1 SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1
export SGLANG_CHUNK_TOPLEVEL=1 SGLANG_CHUNK_COARSE=0
export SGLANG_CACHEBLEND_CHUNK=1 SGLANG_CACHEBLEND_BATCH=1 SGLANG_CACHEBLEND_COMPACT=0
export SGLANG_CACHEBLEND_MULTI_SLOT=1 SGLANG_CACHEBLEND_MULTI_SLOT_MAX_GAP=256
export SGLANG_CACHEBLEND_OFFMAP=1 SGLANG_CACHEBLEND_MAX_CACHED_RATIO=0.95
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_15case_v1
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2 SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=1 SGLANG_PRECOMPUTE_SELECTIVE_REFRESH_FRAC=0.25
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
unset SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_LATE
unset SGLANG_CHUNK_HEAD_RECOMPUTE_EARLY_N SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC
export SGLANG_CHUNK_HEAD_RECOMPUTE_NODE_KIND=1
export SGLANG_CHUNK_NODE_KIND_BOUNDARY=interface
export SGLANG_NK_DEBUG=1
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000_diverse15/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/smoke_nodekind \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --mem-fraction-static 0.72 --max-total-tokens 16384 \
  --max-tasks 2 --agent-count 2 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 1 \
  --precompute-kv-dir results/codebase_kv/pandas_15case_v1 \
  --precompute-host-size-gb 2 \
  --precompute-canonical-prefix \
  --include-source-with-precompute \
  --disable-hierarchical-cache \
  --task-mode verdict
