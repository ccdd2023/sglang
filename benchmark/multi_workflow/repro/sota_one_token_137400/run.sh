#!/usr/bin/env bash
# Replay job 137400: same-engine file-island vs KVCOMM-style vs CacheBlend-style.
# Historical full-island geometry. Not the served center-8 bundle.
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IMPACTKV_PROJECT="${IMPACTKV_PROJECT:-$(cd "$_here/../../../.." && pwd)}"
# shellcheck disable=SC1091
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/impactkv_local_env.sh"
cd "$IMPACTKV_PROJECT"

artifact="${1:-$IMPACTKV_ARTIFACTS/runs/sota_one_token_137400}"
mkdir -p "$artifact"

# Any explicit quality knob disables the served bundle. All four stay 0 so
# the copy is the historical full island, matching job 137400.
export SGLANG_KVCOMM_SHIFT_DENSE_LIMIT=0
export SGLANG_KVCOMM_CENTER_COPY=0
export SGLANG_KVCOMM_RIGHT_SEAM_ANCHOR=0
export SGLANG_KVCOMM_RIGHT_SEAM_K=0
unset SGLANG_KV_PREFETCH || true
unset SGLANG_KV_PREFETCH_MIDDLE || true
unset SGLANG_KV_PREFETCH_ON_DECODE || true
unset SGLANG_KV_PRODUCER_EXTRACT || true
unset IMPACTKV_PRODUCER_EXTRACT || true
unset SGLANG_KV_EXTENDED_MODULES || true
unset SGLANG_KVCOMM_L1 || true
unset SGLANG_KVCOMM_WIDE_COPY_MAX_SHIFT || true
unset SGLANG_KVCOMM_SHIFT_DENSE_WHOLE_PROMPT || true
unset IMPACTKV_PREFER_HOST || true
unset IMPACTKV_ONLINE_ADMIT || true
unset IMPACTKV_OPERATION_BINS || true
unset IMPACTKV_PROMOTE_AFTER_BIND || true

plan="$_here/PLAN.coding.json"
expected="2cf6f1fd7324d18bd859ec9e312f4b240bd8e05a754c18d4236dc51f9d9f1dce"
got="$(sha256sum "$plan" | awk '{print $1}')"
if [[ "$got" != "$expected" ]]; then
  echo "PLAN.coding.json sha256 mismatch: $got" >&2
  exit 2
fi
cp -a "$plan" "$artifact/PLAN.coding.json"

PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/prepare_7b_sota_copier_plans.py \
  --coding-plan "$artifact/PLAN.coding.json" \
  --output-dir "$artifact"

port="${IMPACTKV_PORT:-38080}"
PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_swebench_7b_sota_copiers.py \
  --artifact "$artifact" \
  --port "$port" \
  --model "$IMPACTKV_MODEL"

echo "wrote $artifact/RESULT.json"
echo "compare with $_here/RESULT.json"
