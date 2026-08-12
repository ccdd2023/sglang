#!/usr/bin/env bash
#SBATCH --job-name=impactkv-cb-build
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
repo="$HOME/CodeMAS_Project/worktrees/cacheblend-common-agent"
python="$HOME/.venvs/cacheblend-native/bin/python"

export PYTHONNOUSERSITE=1
# The compute nodes only expose a complete CUDA 12 toolkit at 12.9.  PyTorch
# 2.2.1 is cu121; the major version matches, and its extension checker permits
# a newer CUDA 12 minor while warning about the minor-version difference.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$HOME/.venvs/cacheblend-native/bin:$CUDA_HOME/bin:/usr/bin:/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS="${MAX_JOBS:-16}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export CMAKE_BUILD_TYPE=Release
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDA_TOOLKIT_INCLUDE=$CUDA_HOME/include -DCUDA_CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so"
export PYTHONPATH="$repo/vllm_blend:$repo"

cd "$repo/vllm_blend"
rm -rf build/temp.linux-x86_64-cpython-310
"$python" setup.py build_ext --inplace
"$python" -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)'
sha256sum \
  vllm/_C.cpython-310-x86_64-linux-gnu.so \
  vllm/_moe_C.cpython-310-x86_64-linux-gnu.so \
  > "$HOME/impactkv-runtime/logs/common-baselines/CACHEBLEND_EXTENSION_SHA256SUMS"
