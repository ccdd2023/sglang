#!/usr/bin/env bash
#SBATCH --job-name=impactkv-sglang-exact
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --nodelist=gpu17
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
export IMPACTKV_MODEL="$HOME/models/Qwen2.5-Coder-7B-Instruct"
export IMPACTKV_TOKENIZER_JSON="$IMPACTKV_MODEL/tokenizer.json"
export IMPACTKV_CHAT_TEMPLATE="$IMPACTKV_PROJECT/benchmark/multi_workflow/qwen2_5_coder_tool_chat_template.jinja"
export IMPACTKV_SERVER_READY_TIMEOUT=900
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

label="${IMPACTKV_COMMON_REPLAY_LABEL:?set canary4 or fresh24}"
campaign="${IMPACTKV_COMMON_CAMPAIGN:-$IMPACTKV_ARTIFACTS/impactkv_common_agent_baselines_fresh24_20260812}"
arm="${IMPACTKV_COMMON_ARM:-coding_dependency_graph_cold_lcb}"
port="${IMPACTKV_COMMON_PORT:-$((35000 + SLURM_JOB_ID % 1000))}"

PYTHONPATH="$IMPACTKV_PROJECT" timeout --foreground --signal=TERM \
  --kill-after=60s 7h "$IMPACTKV_MINI_PYTHON" \
  benchmark/multi_workflow/run_common_sglang_exact_prompt_replay.py \
  --campaign "$campaign" --arm "$arm" \
  --label "$label" --port "$port" run-all
