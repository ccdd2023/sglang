#!/bin/bash
# Exp4 (H3): SYNC + per-layer no-op event loop AFTER the base sync. Tests
# whether the per-layer wait pattern (not load_stream) is what LAYERED's
# F1=0.508 derives from.
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
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=0  # SYNC base
# Exp4 toggle: per-layer event.record/wait loop.
export SGLANG_KVFLOW_PERLAYERWAIT=1

exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/7b_precompute_ab_sync_perlayerwait \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --mem-fraction-static 0.85 --max-total-tokens 32768 \
  --max-tasks 5 --agent-count 5 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --precompute-kv-dir results/codebase_kv/pandas_5case \
  --precompute-host-size-gb 2 \
  --precompute-canonical-prefix \
  --include-source-with-precompute \
  --chunk-size 6