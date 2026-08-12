#!/usr/bin/env bash
#SBATCH --job-name=impactkv-native-envs
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
uv="$HOME/.local/bin/uv"
export UV_CACHE_DIR="$HOME/.cache/impactkv/uv"
export PYTHONNOUSERSITE=1
logs="$HOME/impactkv-runtime/logs/common-baselines"
mkdir -p "$logs"

cacheblend="$HOME/.venvs/cacheblend-native"
kvcomm="$HOME/.venvs/kvcomm-native"
"$uv" venv --python 3.10 "$cacheblend"
"$uv" pip install --python "$cacheblend/bin/python" \
  --requirement \
  "$HOME/CodeMAS_Project/worktrees/cacheblend-common-agent/vllm_blend/requirements-cuda.txt" \
  transformers==4.40.2 tokenizers==0.19.1 numpy==1.26.4 setuptools==69.5.1

"$uv" venv --python 3.10 "$kvcomm"
"$uv" pip install --python "$kvcomm/bin/python" \
  torch==2.1.0 torchvision==0.16.0 transformers==4.50.2 \
  accelerate==1.10.1 class-registry==2.1.2 loguru==0.7.2 \
  python-dotenv==1.0.0 tenacity==8.2.3 async-timeout==4.0.3 \
  openai==1.6.0 pytest==8.0.2 numpy==1.26.4 setuptools==69.5.1

"$uv" pip freeze --python "$cacheblend/bin/python" \
  > "$logs/CACHEBLEND_NATIVE_FREEZE.txt"
"$uv" pip freeze --python "$kvcomm/bin/python" \
  > "$logs/KVCOMM_NATIVE_FREEZE.txt"

PYTHONPATH="$HOME/CodeMAS_Project/worktrees/kvcomm-common-agent" \
  "$kvcomm/bin/python" -c \
  'import torch, transformers; import KVCOMM; print(torch.__version__, transformers.__version__)'
