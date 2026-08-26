#!/usr/bin/env bash
#SBATCH --job-name=impactkv-gen-lcs
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,15,23-24]
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

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
export IMPACTKV_MODEL="${IMPACTKV_MODEL:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit}"
export IMPACTKV_CHAT_TEMPLATE="$IMPACTKV_PROJECT/benchmark/multi_workflow/qwen3_coder_tool_chat_template.jinja"
export IMPACTKV_SERVER_READY_TIMEOUT=900
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

# Same-token general LCS copier. Do not write into 96092.
artifact="${IMPACTKV_GENERAL_ARTIFACT:-$HOME/CodeMAS_Project/kvflow-artifacts/impactkv_swebench_general_lcs_20260822}"
port="$((47000 + SLURM_JOB_ID % 1000))"

PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_MINI_PYTHON" \
  benchmark/multi_workflow/run_swebench_prerotated_file_modules.py \
  --artifact "$artifact" --port "$port" --model "$IMPACTKV_MODEL"
