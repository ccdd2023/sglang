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

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
export IMPACTKV_MODEL="$HOME/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
export IMPACTKV_CHAT_TEMPLATE="$IMPACTKV_PROJECT/benchmark/multi_workflow/qwen3_coder_tool_chat_template.jinja"
export IMPACTKV_SERVER_READY_TIMEOUT=900
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

artifact="$HOME/CodeMAS_Project/kvflow-artifacts/impactkv_swebench_prerotated_file_modules_20260818"
port="$((46000 + SLURM_JOB_ID % 1000))"

PYTHONNOUSERSITE=1 PYTHONPATH="$IMPACTKV_PROJECT" "$IMPACTKV_MINI_PYTHON" \
  benchmark/multi_workflow/run_swebench_prerotated_file_modules.py \
  --artifact "$artifact" --port "$port" --model "$IMPACTKV_MODEL"
