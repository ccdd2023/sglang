#!/usr/bin/env bash
# Online-admit 7B campaign. New directory only. Never writes prefixkey_20260824.
set -euo pipefail

_mw="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_mw/impactkv_local_env.sh"
cd "$IMPACTKV_PROJECT"

frozen="$IMPACTKV_ARTIFACTS/impactkv_swebench_7b_file_modules_prefixkey_20260824/PLAN.json"
if [[ ! -f "$frozen" ]]; then
  echo "missing frozen PLAN: $frozen" >&2
  exit 1
fi
if [[ ! -f "$IMPACTKV_MODEL/config.json" ]]; then
  echo "IMPACTKV_MODEL must be a local snapshot with config.json, got: $IMPACTKV_MODEL" >&2
  exit 1
fi
weights="$(find "$IMPACTKV_MODEL" -name '*.safetensors' -o -name '*.bin' | head -1 || true)"
if [[ -z "$weights" ]]; then
  echo "IMPACTKV_MODEL looks tokenizer-only: $IMPACTKV_MODEL" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
run="${IMPACTKV_RUN_DIR:-$IMPACTKV_ARTIFACTS/runs/online_admit_7b_${stamp}}"
mkdir -p "$run"
if [[ ! -f "$run/PLAN.json" ]]; then
  PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
    "$IMPACTKV_EVAL_PYTHON" \
    benchmark/multi_workflow/prepare_online_admit_plan.py \
    --official-plan "$frozen" \
    --output-dir "$run"
fi
port="${IMPACTKV_PORT:-30000}"
max_groups="${IMPACTKV_MAX_GROUPS:-}"

echo "online admit run $run"
echo "This is NOT job 137185. Do not copy RESULT.json over prefixkey_20260824."

extra=()
if [[ -n "$max_groups" ]]; then
  extra+=(--max-groups "$max_groups")
fi

export IMPACTKV_ONLINE_ADMIT=1
template="${SGLANG_KVCOMM_CLASS_TEMPLATE:-$_mw/templates/coding_agent.json}"
if [[ -f "$template" ]]; then
  export SGLANG_KVCOMM_CLASS_TEMPLATE="$template"
  echo "class template $SGLANG_KVCOMM_CLASS_TEMPLATE"
fi
PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_swebench_prerotated_file_modules.py \
  --artifact "$run" --port "$port" --model "$IMPACTKV_MODEL" \
  "${extra[@]}"
