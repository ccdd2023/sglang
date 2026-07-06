#!/bin/bash
# FAIR-MEASUREMENT / KVCOMM regime (Step 2) — L2 whole-slot GENERAL BASELINE.
# Byte-exact hash selection + RoPE rotation (KVCOMM), NO AST chunking.
# MiniLM L3 + offset-gate OFF (wrong regime). L4/C2 OFF. Position-shift, no vary-code.
set -e
cd /home/gfy/CodeMAS_Project/sglang-kvflow
export SGLANG_PLACEHOLDER_KNN_MATCH=0
export SGLANG_L3_AST_GATE=0
export SGLANG_L3_AST_GATE_OFFSET=0
unset SGLANG_CHUNKED_PLACEHOLDER_KNN SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH 2>/dev/null || true
unset SGLANG_CACHEBLEND_CHUNK SGLANG_CACHEBLEND_DIRECT SGLANG_CACHEBLEND_BATCH 2>/dev/null || true
exec /home/gfy/.conda/envs/sglang-kvflow/bin/python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
  --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
  --repo-root results/giant_codebase/pandas_src \
  --out-dir results/kvcomm_ab/l2_baseline \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-tasks 4 --agent-count 5 \
  --mode placeholder_knn_reuse --segment-count 5 \
  --position-shift --no-vary-code \
  --chunk-size 2
