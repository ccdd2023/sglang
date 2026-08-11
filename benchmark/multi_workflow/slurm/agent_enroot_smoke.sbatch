#!/usr/bin/env bash
#SBATCH --job-name=impactkv-agent-smoke
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13]
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
source "$HOME/CodeMAS_Project/sglang/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

output="$IMPACTKV_ARTIFACTS/gpuhome_migration_agent_smoke_$SLURM_JOB_ID"
common=(
  --output "$output"
  run-arm
  --scope canary
  --instance-filter 'astropy__astropy-7336'
  --agent-container-backend enroot
  --evaluation-container-backend enroot
  --official
)
"$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py \
  "${common[@]}" --arm dense --port 32000
"$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py \
  "${common[@]}" --arm coding_dependency_graph_cold_lcb --port 32001
