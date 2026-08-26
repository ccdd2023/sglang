#!/usr/bin/env bash
# Off-cluster ImpactKV environment. No CodeMAS, no HKBU proxy.
# Source from the repo root or from a campaign script.
set -euo pipefail

_this="${BASH_SOURCE[0]:-$0}"
_mw="$(cd "$(dirname "$_this")" && pwd)"
export IMPACTKV_PROJECT="${IMPACTKV_PROJECT:-$(cd "$_mw/../.." && pwd)}"
export IMPACTKV_ARTIFACTS="${IMPACTKV_ARTIFACTS:-$IMPACTKV_PROJECT/impactkv-artifacts}"
export IMPACTKV_MODEL="${IMPACTKV_MODEL:-$HOME/models/Qwen2.5-Coder-7B-Instruct}"
export IMPACTKV_CHAT_TEMPLATE="${IMPACTKV_CHAT_TEMPLATE:-$IMPACTKV_PROJECT/benchmark/multi_workflow/qwen3_coder_tool_chat_template.jinja}"
export IMPACTKV_MEM_FRACTION_STATIC="${IMPACTKV_MEM_FRACTION_STATIC:-0.82}"
export IMPACTKV_SERVER_READY_TIMEOUT="${IMPACTKV_SERVER_READY_TIMEOUT:-900}"
export PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${IMPACTKV_EVAL_PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    export IMPACTKV_EVAL_PYTHON="$(command -v python)"
  else
    export IMPACTKV_EVAL_PYTHON="$(command -v python3)"
  fi
fi
export IMPACTKV_MINI_PYTHON="${IMPACTKV_MINI_PYTHON:-$IMPACTKV_EVAL_PYTHON}"

mkdir -p "$IMPACTKV_ARTIFACTS" "$IMPACTKV_ARTIFACTS/runs"

if [[ ! -f "$IMPACTKV_PROJECT/python/sglang/srt/mem_cache/kvcomm_exact.py" ]]; then
  echo "IMPACTKV_PROJECT does not look like this sglang clone: $IMPACTKV_PROJECT" >&2
  echo "sbatch from the repo root, or export IMPACTKV_PROJECT=/path/to/sglang-kvflow" >&2
  exit 1
fi
