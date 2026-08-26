#!/usr/bin/env bash
# Lossy copy → prefix-priority prefetch → template prefetch of the lossy pool.
# Fair host-resident baseline. New directory only. Never writes frozen RESULT.
set -euo pipefail

_mw="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_mw/impactkv_local_env.sh"
export IMPACTKV_CHAT_TEMPLATE="${IMPACTKV_CHAT_TEMPLATE:-$_mw/qwen2_5_coder_tool_chat_template.jinja}"
cd "$IMPACTKV_PROJECT"

frozen="${IMPACTKV_PREFETCH_PLAN:-$IMPACTKV_ARTIFACTS/impactkv_swebench_template_prefetch_7b_dualisland_20260822/PLAN.json}"
if [[ ! -f "$frozen" ]]; then
  echo "missing dual-island PLAN: $frozen" >&2
  exit 1
fi
if [[ ! -f "$IMPACTKV_MODEL/config.json" ]]; then
  echo "IMPACTKV_MODEL must be a local snapshot with config.json, got: $IMPACTKV_MODEL" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
run="${IMPACTKV_RUN_DIR:-$IMPACTKV_ARTIFACTS/runs/prefetch_ablation_7b_${stamp}}"
mkdir -p "$run"
if [[ ! -f "$run/PLAN.json" ]]; then
  cp -a "$frozen" "$run/PLAN.json"
fi
port="${IMPACTKV_PORT:-30000}"
max_groups="${IMPACTKV_MAX_GROUPS:-}"
modes="${IMPACTKV_PREFETCH_MODES:-dense,lossy_host,prefix_prefetch,template_prefetch}"

echo "prefetch ablation run $run"
echo "This is NOT job 137185 and not the frozen 7-group dual-island RESULT."

extra=()
if [[ -n "$max_groups" ]]; then
  extra+=(--max-groups "$max_groups")
fi
template="${SGLANG_KVCOMM_CLASS_TEMPLATE:-$_mw/templates/coding_agent.json}"
if [[ -f "$template" ]]; then
  export SGLANG_KVCOMM_CLASS_TEMPLATE="$template"
  echo "class template $SGLANG_KVCOMM_CLASS_TEMPLATE"
fi
PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_swebench_template_prefetch.py \
  --artifact "$run" --port "$port" --model "$IMPACTKV_MODEL" \
  --modes "$modes" \
  "${extra[@]}"
