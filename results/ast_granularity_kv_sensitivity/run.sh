#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/gfy/.conda/envs/sglang-kvflow/bin/python}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
DEVICE=${DEVICE:-cuda}
MAX_SAMPLES=${MAX_SAMPLES:-2}
FILES_PER_SAMPLE=${FILES_PER_SAMPLE:-1}
PER_GRANULARITY=${PER_GRANULARITY:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}

cd /home/gfy/CodeMAS_Project/sglang-kvflow

"${PYTHON}" results/ast_granularity_kv_sensitivity/granularity_sampler.py \
  --max-samples "${MAX_SAMPLES}" \
  --files-per-sample "${FILES_PER_SAMPLE}" \
  --per-granularity "${PER_GRANULARITY}"

"${PYTHON}" results/ast_granularity_kv_sensitivity/granularity_analyzer.py \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --max-seq-len "${MAX_SEQ_LEN}"

"${PYTHON}" results/ast_granularity_kv_sensitivity/report_generator.py
