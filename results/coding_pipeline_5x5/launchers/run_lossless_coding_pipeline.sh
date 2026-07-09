#!/bin/bash
# R40 (2026-07-09): Lossless reference for coding_pipeline task (5×5).
# Same harness as results/lossy_alg_round32/launchers/run_lossless_verdict.sh but
# with --task-mode coding_pipeline (5 agents: coder/tester/reviewer/refactorer/integrator).
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case_v4
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
unset SGLANG_AST_REUSE_TYPES
unset SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC
unset SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_EARLY
unset SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC_LATE
unset SGLANG_CHUNK_HEAD_RECOMPUTE_EARLY_N
unset SGLANG_CHUNK_TYPE_AWARE_FRAC
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/coding_pipeline_5x5/lossless \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --mem-fraction-static 0.85 --max-total-tokens 65536 \
  --max-tasks 5 --agent-count 5 \
  --mode placeholder_slot_lossless --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 6 \
  --agent-max-tokens 768 \
  --task-mode coding_pipeline