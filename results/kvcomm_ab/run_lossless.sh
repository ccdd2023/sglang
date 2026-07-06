#!/bin/bash
# FAIR-MEASUREMENT / KVCOMM regime (Step 2) — LOSSLESS accuracy reference.
# SAME slot-decomposed prompt as placeholder_knn_reuse (build_slot_messages),
# but reuse_mode=lossless → NO reuse path fires → full prefill of the identical
# prompt. This is the valid F1 ground truth (lossless_full_prefill uses a
# different prompt). MiniLM L3 / offset-gate / L4 / C2 all OFF.
# Position-shift, no vary-code.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/lossless \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-tasks 4 --agent-count 5 \
  --mode placeholder_slot_lossless --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 2
