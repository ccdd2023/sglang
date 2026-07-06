#!/bin/bash
# R26 — 3B model × 3 agents × verdict task_mode (lossless control)
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
export SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case_v6_verdict_3b
export SGLANG_PRECOMPUTE_HOST_SIZE_GB=2
export SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1
export SGLANG_PRECOMPUTE_PROMPT_ALIGN=1
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/lossy_alg_round26/r26_3b_3agent_lossless \
  --model /home/gfy/models/Qwen2.5-3B-Instruct \
  --mem-fraction-static 0.85 --max-total-tokens 65536 \
  --max-tasks 5 --agent-count 3 \
  --mode placeholder_slot_lossless --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 6 \
  --task-mode verdict