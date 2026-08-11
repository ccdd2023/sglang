#!/usr/bin/env bash
#SBATCH --job-name=impactkv-import-canary
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
source "$HOME/CodeMAS_Project/sglang/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

quota -s || true
"$IMPACTKV_EVAL_PYTHON" \
  benchmark/multi_workflow/prepare_enroot_images.py \
  --dataset "$IMPACTKV_EVAL_SNAPSHOT" \
  --instances "astropy__astropy-7336,django__django-16899,pytest-dev__pytest-7432,scikit-learn__scikit-learn-13779,sphinx-doc__sphinx-9230"
quota -s || true
