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
# Compute nodes expose CUDA 12.9 headers/libraries below targets/x86_64-linux,
# while the legacy FindCUDA bundled with torch 2.2 expects include and lib64 at
# the toolkit root.  Build a home-scoped symlink view; no node-local /tmp or
# system path is written.
system_cuda=/usr/local/cuda-12.9
header_snapshot="$HOME/impactkv-runtime/toolchains/cuda-12.9-login-headers"
library_snapshot="$HOME/impactkv-runtime/toolchains/cuda-12.9-login-libs"
nvtx_root="$HOME/.venvs/cacheblend-native/lib/python3.10/site-packages/nvidia/nvtx"
export CUDA_HOME="$HOME/impactkv-runtime/toolchains/cuda-12.9-legacy-layout"
test -f "$header_snapshot/cuda_runtime.h"
test -f "$library_snapshot/libcudart_static.a"
test -f "$nvtx_root/include/nvToolsExt.h"
mkdir -p "$CUDA_HOME"
rm -rf "$CUDA_HOME/bin" "$CUDA_HOME/include" "$CUDA_HOME/lib64" \
  "$CUDA_HOME/targets" "$CUDA_HOME/nvvm"
mkdir -p "$CUDA_HOME/bin"
for entry in "$system_cuda"/bin/*; do
  ln -sfn "$entry" "$CUDA_HOME/bin/$(basename "$entry")"
done
ln -sfn "$system_cuda/nvvm" "$CUDA_HOME/nvvm"
ln -sfn "$header_snapshot" "$CUDA_HOME/include"
for entry in "$nvtx_root"/include/nvToolsExt*.h; do
  ln -sfn "$entry" "$CUDA_HOME/include/$(basename "$entry")"
done
mkdir -p "$CUDA_HOME/lib64" "$CUDA_HOME/targets/x86_64-linux"
for entry in "$system_cuda"/targets/x86_64-linux/lib/*; do
  ln -sfn "$entry" "$CUDA_HOME/lib64/$(basename "$entry")"
done
for entry in "$library_snapshot"/*; do
  ln -sfn "$entry" "$CUDA_HOME/lib64/$(basename "$entry")"
done
# Compute nodes omit the development-name symlink although the shared object
# is present. FindCUDA searches for the unversioned name.
ln -sfn "$system_cuda/targets/x86_64-linux/lib/libcudart.so.12" \
  "$CUDA_HOME/lib64/libcudart.so"
ln -sfn "$nvtx_root/lib/libnvToolsExt.so.1" \
  "$CUDA_HOME/lib64/libnvToolsExt.so"
ln -sfn "$header_snapshot" "$CUDA_HOME/targets/x86_64-linux/include"
ln -sfn "$CUDA_HOME/lib64" "$CUDA_HOME/targets/x86_64-linux/lib"
export PATH="$CUDA_HOME/bin:/usr/bin:/bin:$HOME/.venvs/cacheblend-native/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS="${MAX_JOBS:-16}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export CMAKE_BUILD_TYPE=Release
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME"
export PYTHONPATH="$repo/vllm_blend:$repo"

cd "$repo/vllm_blend"
rm -rf build/temp.linux-x86_64-cpython-310
"$python" setup.py build_ext --inplace
"$python" -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)'
sha256sum \
  vllm/_C.cpython-310-x86_64-linux-gnu.so \
  vllm/_moe_C.cpython-310-x86_64-linux-gnu.so \
  > "$HOME/impactkv-runtime/logs/common-baselines/CACHEBLEND_EXTENSION_SHA256SUMS"
