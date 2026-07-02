#!/bin/bash
# 7B-Coder A/B experimental: offline-precomputed codebase KV.
# Server loads results/codebase_kv/pandas_5case into a CPU host pool at start;
# read path transfers chunks CPU->GPU on reuse (location="host"). Canonical
# prefix prepended to every system message. Agent 1 INCLUDED (pool warm at
# start → agent 1 is a reuse beneficiary).
# Speedup must come ONLY from more reuse. Honest accuracy bound: only the
# canonical preamble is lossless; file content at shifted positions stays lossy.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
# L4 AST chunk read/write path (the path that consults placeholder_chunk_pool,
# which the precompute loader populates with host-resident entries).
export SGLANG_CHUNKED_PLACEHOLDER_KNN=1
export SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1
export SGLANG_CHUNK_TOPLEVEL=1
export SGLANG_CHUNK_COARSE=0
export SGLANG_CACHEBLEND_CHUNK=1
export SGLANG_CACHEBLEND_BATCH=1
# precompute env (propagated by the driver too, but set here for clarity):
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
# Async CPU->GPU transfer: per-layer copies on a dedicated stream + event wait
# (vs sync mode's full cuda.synchronize). Overlaps the transfer with GPU work.
export SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=1

exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/7b_precompute_ab \
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
