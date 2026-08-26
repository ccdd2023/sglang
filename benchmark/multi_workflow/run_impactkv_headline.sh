#!/usr/bin/env bash
# Run the 7B SWE-bench file-island campaign on this machine (no Slurm).
# Writes a NEW artifact directory. Never overwrites frozen 137185 RESULT.json.
set -euo pipefail

_mw="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_mw/impactkv_local_env.sh"
cd "$IMPACTKV_PROJECT"

if [[ ! -d "$IMPACTKV_ARTIFACTS/impactkv_swebench_7b_file_modules_prefixkey_20260824" ]]; then
  "$IMPACTKV_EVAL_PYTHON" "$_mw/fetch_impactkv_artifacts.py" --dest "$IMPACTKV_ARTIFACTS"
fi

plan="$IMPACTKV_ARTIFACTS/impactkv_swebench_7b_file_modules_prefixkey_20260824/PLAN.json"
if [[ ! -f "$plan" ]]; then
  echo "missing frozen PLAN: $plan" >&2
  exit 1
fi
if [[ ! -f "$IMPACTKV_MODEL/config.json" ]]; then
  echo "IMPACTKV_MODEL must be a local snapshot with config.json, got: $IMPACTKV_MODEL" >&2
  echo "huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir \"\$HOME/models/Qwen2.5-Coder-7B-Instruct\"" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; this campaign needs a GPU (~24GB for 7B bf16)" >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
run="${IMPACTKV_RUN_DIR:-$IMPACTKV_ARTIFACTS/runs/headline_7b_${stamp}}"
mkdir -p "$run"
cp -a "$plan" "$run/PLAN.json"
port="${IMPACTKV_PORT:-30000}"
max_groups="${IMPACTKV_MAX_GROUPS:-}"

echo "IMPACTKV_PROJECT=$IMPACTKV_PROJECT"
echo "IMPACTKV_MODEL=$IMPACTKV_MODEL"
echo "run dir $run"
echo "port $port"
echo "This is NOT job 137185. Compare RESULT.json to the frozen file; do not replace it."

extra=()
if [[ -n "$max_groups" ]]; then
  extra+=(--max-groups "$max_groups")
fi

PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_swebench_prerotated_file_modules.py \
  --artifact "$run" --port "$port" --model "$IMPACTKV_MODEL" \
  "${extra[@]}"
