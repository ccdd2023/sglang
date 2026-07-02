#!/bin/bash
# Exp2 (H2): LAYERED with byte-cmp dump enabled. Sister to the SYNC bytecmp
# launcher; dumps LAYERED's per-layer RoPE output (tagged by layer_id) for
# comparison against the SYNC dumps.
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
export SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=1  # LAYERED
# Exp2 toggle: dump K-tensor stats after each LAYERED per-layer RoPE.
export SGLANG_KVFLOW_BYTECMP_DUMP=1
export SGLANG_KVFLOW_BYTECMP_LIMIT=4

exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/7b_precompute_ab_layered_bytecmp \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --mem-fraction-static 0.85 --max-total-tokens 32768 \
  --max-tasks 1 --agent-count 1 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --precompute-kv-dir results/codebase_kv/pandas_5case \
  --precompute-host-size-gb 2 \
  --precompute-canonical-prefix \
  --include-source-with-precompute \
  --chunk-size 6