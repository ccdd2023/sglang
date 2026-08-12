#!/usr/bin/env bash
#SBATCH --job-name=impactkv-sglang-agent
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --nodelist=gpu18
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
export IMPACTKV_MODEL="$HOME/models/Qwen2.5-Coder-7B-Instruct"
export IMPACTKV_TOKENIZER_JSON="$IMPACTKV_MODEL/tokenizer.json"
export IMPACTKV_AGENT_CONFIG="$IMPACTKV_PROJECT/benchmark/multi_workflow/swebench_common_qwen25_agent.yaml"
export IMPACTKV_CHAT_TEMPLATE="$IMPACTKV_PROJECT/benchmark/multi_workflow/qwen2_5_coder_tool_chat_template.jinja"
export IMPACTKV_AGENT_STEP_LIMIT=32
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

arm="${IMPACTKV_COMMON_ARM:?set IMPACTKV_COMMON_ARM=dense or coding_dependency_graph_cold_lcb}"
cohort="${IMPACTKV_COMMON_SCOPE:-canary}"
instance="${IMPACTKV_COMMON_INSTANCE:-}"
port="${IMPACTKV_COMMON_PORT:-$((34000 + SLURM_JOB_ID % 1000))}"
campaign="$IMPACTKV_ARTIFACTS/impactkv_common_agent_baselines_fresh24_20260812"

case "$cohort" in
  canary)
    export IMPACTKV_DATASET_ROOT="$campaign/canary_dataset"
    export IMPACTKV_EVAL_SNAPSHOT="$campaign/CANARY4.json"
    export IMPACTKV_EVAL_REGISTRATION="$campaign/BRIDGE_CANARY4_REGISTRATION.json"
    ;;
  formal)
    export IMPACTKV_DATASET_ROOT="$campaign/formal_dataset"
    export IMPACTKV_EVAL_SNAPSHOT="$campaign/FROZEN_FRESH24.json"
    export IMPACTKV_EVAL_REGISTRATION="$campaign/BRIDGE_FRESH24_REGISTRATION.json"
    ;;
  *)
    echo "unknown common-agent scope: $cohort" >&2
    exit 64
    ;;
esac

if [[ -n "$instance" ]]; then
  output="$campaign/runs/sglang_${cohort}_${instance}"
  scope_args=(--scope canary --instance-filter "$instance")
else
  output="$campaign/runs/sglang_${cohort}"
  scope_args=(--scope full)
fi

"$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py \
  --output "$output" \
  run-arm \
  --arm "$arm" \
  "${scope_args[@]}" \
  --port "$port" \
  --agent-container-backend enroot \
  --evaluation-container-backend enroot \
  --official
