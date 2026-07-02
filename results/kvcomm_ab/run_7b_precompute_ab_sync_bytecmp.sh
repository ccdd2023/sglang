#!/bin/bash
# Exp2 (H2): SYNC with byte-cmp dump enabled. Dumps first 4 occurrences of
# kvcache.k_buffer[0][dst_slice] after each SYNC RoPE call site to
# results/kvcomm_ab/bytecmp/sync_*.pt. The LAYERED path is exercised by
# run_7b_precompute_ab.sh (LAYERED env). After both runs, compare the dumped
# K-tensor statistics (sum/mean/std/first16) — if SYNC and LAYERED dumps
# match, H2 is dead (RoPE math is identical); the F1 gap must be from
# ordering, not numerics.
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
export SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=0  # SYNC
# Exp2 toggle: dump K-tensor stats after each SYNC RoPE site.
export SGLANG_KVFLOW_BYTECMP_DUMP=1
export SGLANG_KVFLOW_BYTECMP_LIMIT=4

exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/7b_precompute_ab_sync_bytecmp \
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