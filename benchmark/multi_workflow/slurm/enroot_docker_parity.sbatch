#!/usr/bin/env bash
#SBATCH --job-name=impactkv-enroot-parity
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
source "$HOME/CodeMAS_Project/sglang/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

inputs="$IMPACTKV_HOME/migration-inputs/enroot-parity"
output="$IMPACTKV_ARTIFACTS/gpuhome_enroot_parity_$SLURM_JOB_ID"
instances="${IMPACTKV_PARITY_INSTANCES:-astropy__astropy-7336,django__django-16899,pytest-dev__pytest-7432,scikit-learn__scikit-learn-13779,sphinx-doc__sphinx-9230}"

"$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/enroot_swebench_evaluator.py \
  --dataset "$IMPACTKV_EVAL_SNAPSHOT" \
  --predictions "$inputs/predictions.jsonl" \
  --output "$output" \
  --run-id "gpuhome-enroot-parity-$SLURM_JOB_ID" \
  --instances "$instances" \
  --timeout 3600

"$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/compare_enroot_docker_reports.py \
  --docker "$inputs/docker_official_result.json" \
  --enroot "$output/impactkv__gpuhome-enroot-parity-$SLURM_JOB_ID.json" \
  --instances "$instances" \
  --output "$output/PARITY_RESULT.json"
