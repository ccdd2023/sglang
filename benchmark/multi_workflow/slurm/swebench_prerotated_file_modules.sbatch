#!/usr/bin/env bash
#SBATCH --job-name=impactkv-swefile
#SBATCH --partition=long
# gpu11-13 are debug login/debug nodes; never schedule compute there.
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

_this="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IMPACTKV_PROJECT="${IMPACTKV_PROJECT:-$(cd "$_this/../../.." && pwd)}"
export IMPACTKV_MODEL="${IMPACTKV_MODEL:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit}"
# shellcheck disable=SC1091
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/impactkv_local_env.sh"
cd "$IMPACTKV_PROJECT"

# Frozen paper RESULT is 96092; this job must use a new dir.
artifact="${IMPACTKV_30B_FM_ARTIFACT:-$IMPACTKV_ARTIFACTS/runs/30b_${SLURM_JOB_ID:-manual}}"
mkdir -p "$artifact"
if [[ ! -f "$artifact/PLAN.json" ]]; then
  cp -a "$IMPACTKV_ARTIFACTS/impactkv_swebench_prerotated_file_modules_20260818/PLAN.json" \
    "$artifact/PLAN.json"
fi
port="$((46000 + ${SLURM_JOB_ID:-0} % 1000))"
if [[ "$port" -lt 30000 ]]; then port=30000; fi

PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_swebench_prerotated_file_modules.py \
  --artifact "$artifact" --port "$port" --model "$IMPACTKV_MODEL"
