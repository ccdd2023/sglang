#!/usr/bin/env bash
#SBATCH --job-name=impactkv-7b-sota
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,15,17,23-24]
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

pick_free_gpu() {
  local spec="$1"
  local best="" best_used=999999 g used
  IFS=',' read -ra _gpus <<< "$spec"
  for g in "${_gpus[@]}"; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt "$best_used" ]; then
      best_used=$used
      best=$g
    fi
  done
  if [ -z "$best" ]; then
    echo "no allocated GPU from spec=$spec" >&2
    nvidia-smi || true
    exit 79
  fi
  export CUDA_VISIBLE_DEVICES="$best"
  echo "CUDA_VISIBLE_DEVICES=$best used=${best_used}MiB SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
  if [ "$best_used" -gt 1024 ]; then
    echo "least-used allocated GPU still holds ${best_used} MiB; refuse" >&2
    nvidia-smi || true
    exit 79
  fi
}
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
  pick_free_gpu "$SLURM_JOB_GPUS"
elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
  pick_free_gpu "$SLURM_STEP_GPUS"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

_this="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IMPACTKV_PROJECT="${IMPACTKV_PROJECT:-$(cd "$_this/../../.." && pwd)}"
# shellcheck disable=SC1091
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/impactkv_local_env.sh"
cd "$IMPACTKV_PROJECT"

# Same-engine 7B SWE-bench copier bake-off. Not 96092. Not 137185 overwrite.
artifact="${IMPACTKV_7B_SOTA_ARTIFACT:-$IMPACTKV_ARTIFACTS/runs/copiers_${SLURM_JOB_ID:-manual}}"
mkdir -p "$artifact"
src="$IMPACTKV_ARTIFACTS/impactkv_swebench_7b_sota_copiers_20260824"
for plan in PLAN.coding.json PLAN.kvcomm.json PLAN.cacheblend.json; do
  if [[ ! -f "$artifact/$plan" ]]; then
    cp -a "$src/$plan" "$artifact/$plan"
  fi
done
port="$((48200 + ${SLURM_JOB_ID:-0} % 1000))"
if [[ "$port" -lt 30000 ]]; then port=30000; fi

PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_swebench_7b_sota_copiers.py \
  --artifact "$artifact" --port "$port" --model "$IMPACTKV_MODEL"
