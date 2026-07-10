#!/bin/bash
# Lossless 3-case baseline (post-fix) for per-stage TTFT comparison vs R32_f015.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_15case_v1
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
unset SGLANG_AST_REUSE_TYPES SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_LATE SGLANG_CHUNK_HEAD_RECOMPUTE_EARLY_N SGLANG_CHUNK_TYPE_AWARE_FRAC
unset SGLANG_CHUNK_HEAD_RECOMPUTE_NODE_KIND SGLANG_CHUNK_NODE_KIND_BOUNDARY
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000_diverse15/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/p4_verify/lossless_3case \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --mem-fraction-static 0.72 --max-total-tokens 16384 \
  --max-tasks 3 --agent-count 5 \
  --mode placeholder_slot_lossless --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 1 \
  --precompute-kv-dir results/codebase_kv/pandas_15case_v1 \
  --precompute-host-size-gb 2 \
  --precompute-canonical-prefix \
  --include-source-with-precompute \
  --task-mode verdict