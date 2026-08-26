#!/usr/bin/env bash
#SBATCH --job-name=impactkv-7b-pre
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

# 235-group prefix-on increment. Same PLAN as 137185. Prefetch off.
# Do not write into frozen 137185 / 96092 artifacts.
artifact="${IMPACTKV_7B_PREFIX_ON_ARTIFACT:-$IMPACTKV_ARTIFACTS/runs/prefix_on_${SLURM_JOB_ID:-manual}}"
mkdir -p "$artifact"
if [[ ! -f "$artifact/PLAN.json" ]]; then
  cp -a "$IMPACTKV_ARTIFACTS/impactkv_swebench_7b_file_modules_prefixkey_20260824/PLAN.json" \
    "$artifact/PLAN.json"
fi
port="$((48300 + ${SLURM_JOB_ID:-0} % 1000))"
if [[ "$port" -lt 30000 ]]; then port=30000; fi
modes_file="${IMPACTKV_PREFIX_ON_MODES_FILE:-$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/prefix_on_modes.txt}"
if [[ -f "$modes_file" ]]; then
  modes="$(tr -d ' \t\r\n' < "$modes_file")"
else
  modes="dense,prefix_only,lossy_only,dual"
fi
if [[ "$modes" == *combined* ]]; then
  echo "prefix-on campaign refuses combined/prefetch: $modes" >&2
  exit 1
fi

PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_swebench_7b_prefix_on.py \
  --artifact "$artifact" --port "$port" --model "$IMPACTKV_MODEL" \
  --modes "$modes"
