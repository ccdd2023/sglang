#!/usr/bin/env bash
#SBATCH --job-name=impactkv-common-images
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

campaign="$IMPACTKV_ARTIFACTS/impactkv_common_agent_baselines_fresh24_20260812"
scope="${IMPACTKV_COMMON_SCOPE:-canary}"
case "$scope" in
  canary)
    dataset="$campaign/CANARY4.json"
    ;;
  formal)
    dataset="$campaign/FROZEN_FRESH24.json"
    ;;
  *)
    echo "unknown common-agent scope: $scope" >&2
    exit 64
    ;;
esac

"$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/prepare_enroot_images.py \
  --dataset "$dataset"

