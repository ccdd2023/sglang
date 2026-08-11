#!/usr/bin/env bash
#SBATCH --job-name=impactkv-agent-smoke
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --nodelist=gpu19
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
instance="${IMPACTKV_AGENT_INSTANCE:-astropy__astropy-7336}"
arm_list="${IMPACTKV_AGENT_ARMS:-dense,coding_dependency_graph_cold_lcb}"
base_port="${IMPACTKV_AGENT_BASE_PORT:-32000}"
common=(
  --output "$output"
  run-arm
  --scope canary
  --instance-filter "$instance"
  --agent-container-backend enroot
  --evaluation-container-backend enroot
  --official
)
IFS=',' read -r -a arms <<< "$arm_list"
for index in "${!arms[@]}"; do
  arm="${arms[$index]}"
  [[ -n "$arm" ]] || continue
  "$IMPACTKV_EVAL_PYTHON" \
    benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py \
    "${common[@]}" --arm "$arm" --port "$((base_port + index))"
done
