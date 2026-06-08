#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gfy/CodeMAS_Project/sglang-kvflow
PY=${PY:-/home/gfy/.conda/envs/sglang-kvflow/bin/python}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
MAX_SEGMENTS=${MAX_SEGMENTS:-12}
MAX_VARIATIONS=${MAX_VARIATIONS:--1}
DEVICE=${DEVICE:-cuda}

cd "$ROOT"

"$PY" results/coding_structure_kv_sensitivity/structure_sampler.py \
  --max-segments "$MAX_SEGMENTS"

"$PY" results/coding_structure_kv_sensitivity/kv_sensitivity_analyzer.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --max-variations "$MAX_VARIATIONS"

"$PY" results/coding_structure_kv_sensitivity/report_generator.py
