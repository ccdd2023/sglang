#!/usr/bin/env bash
#SBATCH --job-name=impactkv-tpl-pf
#SBATCH --partition=long
# gpu11-13 are debug login/debug nodes; never schedule compute there.
#SBATCH --exclude=gpu[10-13,15,17,23-24]
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Two jobs on one dual-GPU node otherwise both open CUDA:0 and OOM
# (124522/124523 on gpu14). Exclusive nodes expose SLURM_JOB_GPUS=0,1;
# binding the first index hit leftover CUDA:0 (124584). Pick the allocated
# GPU with the least used memory; refuse if that GPU still holds >1GiB.
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
elif [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICE_ORDINAL"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

export IMPACTKV_PROJECT="${IMPACTKV_PROJECT:-$HOME/CodeMAS_Project/worktrees/sglang-common-agent}"
export IMPACTKV_MODEL="${IMPACTKV_MODEL:-$HOME/models/Qwen2.5-Coder-7B-Instruct}"
export IMPACTKV_CHAT_TEMPLATE="${IMPACTKV_CHAT_TEMPLATE:-$IMPACTKV_PROJECT/benchmark/multi_workflow/qwen2_5_coder_tool_chat_template.jinja}"
export IMPACTKV_SERVER_READY_TIMEOUT=900
export IMPACTKV_MEM_FRACTION_STATIC="${IMPACTKV_MEM_FRACTION_STATIC:-0.82}"
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
# Exact-prompt replay uses the SGLang eval env, not mini-SWE-agent.
export IMPACTKV_EVAL_PYTHON="${IMPACTKV_EVAL_PYTHON:-$HOME/miniconda3/envs/sglang-kvflow/bin/python}"
export IMPACTKV_MINI_PYTHON="${IMPACTKV_EVAL_PYTHON}"
cd "$IMPACTKV_PROJECT"

# Never write frozen 137185 / 96092 / dualisland_20260822 RESULT.
artifact="${IMPACTKV_PREFETCH_ARTIFACT:-$HOME/CodeMAS_Project/kvflow-artifacts/runs/prefetch_ablation_7b_${SLURM_JOB_ID}}"
port="$((46000 + SLURM_JOB_ID % 1000))"
template="$IMPACTKV_PROJECT/benchmark/multi_workflow/templates/coding_agent.json"
if [[ -f "$template" ]]; then
  export SGLANG_KVCOMM_CLASS_TEMPLATE="$template"
fi

# Never put IMPACTKV_PREFETCH_MODES=a,b on sbatch --export (Slurm comma-splits;
# job 114807 lost combined that way). Prefer a modes file, then env, then all four.
modes_file="${IMPACTKV_PREFETCH_MODES_FILE:-$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/prefetch_modes.txt}"
if [[ -f "$modes_file" ]]; then
  modes="$(tr -d ' \t\r\n' < "$modes_file")"
elif [[ -n "${IMPACTKV_PREFETCH_MODES:-}" ]]; then
  modes="$IMPACTKV_PREFETCH_MODES"
else
  modes="dense,prefix_only,lossy_only,dual,combined"
fi
if [[ -z "$modes" ]]; then
  echo "empty IMPACTKV prefetch modes" >&2
  exit 1
fi

PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT" \
  "$IMPACTKV_MINI_PYTHON" \
  benchmark/multi_workflow/run_swebench_template_prefetch.py \
  --artifact "$artifact" --port "$port" --model "$IMPACTKV_MODEL" \
  --modes "$modes"
