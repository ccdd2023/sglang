#!/usr/bin/env bash
# Run the same_code_context_variation analyzer + table builder for
# each model in the cross-model study. Each model produces its own
# data/context_distance_<slug>.json and data/predicted_distance_table_<slug>.json.
#
# The Qwen2.5-Coder-7B-Instruct data was already produced by the
# original same_code_context_variation/ run (commit 7735cc3d1), so
# here we copy it under the slugged name for the comparison report.
#
# Usage:
#   bash results/lookup_table_transferability/run_all.sh

set -euo pipefail

PROJECT_ROOT="/home/gfy/CodeMAS_Project/sglang-kvflow"
ANALYZER="$PROJECT_ROOT/results/same_code_context_variation/kv_distance_analyzer.py"
TABLE_BUILDER="$PROJECT_ROOT/results/same_code_context_variation/distance_table_builder.py"
OUT_DIR="$PROJECT_ROOT/results/lookup_table_transferability"
SEGS="$PROJECT_ROOT/results/same_code_context_variation/data/segments.json"
VARS="$PROJECT_ROOT/results/same_code_context_variation/data/variations.json"

# Slug helper: lowercase, alnum + dash
slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's|/|--|g; s|[^a-z0-9.]|-|g; s|-\+|-|g; s|^-||; s|-$||'
}

run_model() {
  local model="$1"
  local s; s=$(slug "$model")
  local analyzer_out="$OUT_DIR/data/context_distance_${s}.json"
  if [[ -f "$analyzer_out" ]]; then
    echo "[run_all] skip $model ($analyzer_out already exists)"
  else
    echo "[run_all] $model -> $analyzer_out"
    /home/gfy/.conda/envs/sglang-kvflow/bin/python "$ANALYZER" \
      --model "$model" \
      --segments "$SEGS" \
      --variations "$VARS" \
      --max-variations -1 \
      --max-seq-len 512 \
      --out "$analyzer_out" \
      > "$OUT_DIR/data/run_${s}.log" 2>&1
  fi
  local table_out="$OUT_DIR/data/predicted_distance_table_${s}.json"
  if [[ ! -f "$table_out" ]]; then
    /home/gfy/.conda/envs/sglang-kvflow/bin/python "$TABLE_BUILDER" \
      --in "$analyzer_out" \
      --out "$table_out" \
      >> "$OUT_DIR/data/run_${s}.log" 2>&1
  fi
}

# Slug the existing Qwen2.5-Coder-7B data first (no re-run needed)
mkdir -p "$OUT_DIR/data"
S7B=$(slug "Qwen/Qwen2.5-Coder-7B-Instruct")
if [[ ! -f "$OUT_DIR/data/context_distance_${S7B}.json" ]]; then
  cp "$PROJECT_ROOT/results/same_code_context_variation/data/context_distance_7b.json" \
     "$OUT_DIR/data/context_distance_${S7B}.json"
  cp "$PROJECT_ROOT/results/same_code_context_variation/data/predicted_distance_table.json" \
     "$OUT_DIR/data/predicted_distance_table_${S7B}.json"
  echo "[run_all] copied reference Qwen2.5-Coder-7B-Instruct data"
fi

# 3 new models to run (already in HF cache, no network needed)
run_model "Qwen/Qwen2.5-Coder-3B-Instruct"
run_model "Qwen/Qwen2.5-7B-Instruct"
run_model "Qwen/Qwen3-8B"
# 5th model (Mistral-7B-Instruct-v0.3) was attempted on 2026-06-08 but
# stalled at 3.6/14 GB due to HF unauthenticated rate limits. The 4/4
# Qwen result stands; see results/non_qwen_attempted/REPORT.md.

echo "[run_all] all models done"
ls -la "$OUT_DIR/data/"
