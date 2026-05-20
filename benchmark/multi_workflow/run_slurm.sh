#!/bin/bash
# =============================================================================
# SLURM Job Submission Script for KVFlow Benchmark
#
# Usage:
#   sbatch run_slurm.sh              # Submit job with default settings
#   sbatch run_slurm.sh exp1-large   # Paper-style experiment
#
# Key: All echo/print output goes to /tmp first, copied to home at the end.
# =============================================================================

#SBATCH --job-name=kvflow-benchmark
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=hkbugpusrv07,hkbugpusrv08,hkbugpusrv15  # CUDA incompatibility / NFS issues
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@example.com

set -uo pipefail  # Use -u with care, set defaults for variables below

# All output → /tmp (local tmpfs, no NFS)
OUT="/tmp/slurm-${SLURM_JOB_ID:-$$}.out"
exec >"$OUT" 2>&1

BENCHMARK_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow"
RESULT_DIR="$LOG_DIR/results"
EXP_TYPE="${1:-exp1-large}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Clear flashinfer JIT cache — cached ops were built with wrong CUDA_HOME path.
# Without this, flashinfer uses stale cache and fails with "nvcc: No such file".
rm -rf /home/comp/25480812/.cache/flashinfer 2>/dev/null || true

# GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 || true

# Modules
module load gcc/11.2.0 2>/dev/null || true
module load conda 2>/dev/null || true

# Source conda.sh
for conda_sh in \
    "/usr/local/miniconda/py312_24.7.1-0/etc/profile.d/conda.sh" \
    "/home/comp/25480812/.conda/etc/profile.d/conda.sh" \
    "$HOME/.conda/etc/profile.d/conda.sh"
do
    if [[ -f "$conda_sh" ]]; then
        source "$conda_sh"
        break
    fi
done

CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
if [[ ! -d "$CONDA_ENV_PATH" ]]; then
    exit 1
fi

# DO NOT conda activate -- it modifies sys.path via conda shell hooks (conda.pth, __conda_setup)
# which redirects Python imports to NFS site-packages causing I/O errors on compute nodes.
# Instead, set environment variables directly and run with explicit PYTHONPATH.

# GCC 11.2.0 (loaded via module, also in .bashrc for interactive use)
export CC="${CC:-/usr/local/gcc/gcc-11.2.0/bin/gcc}"
export CXX="${CXX:-/usr/local/gcc/gcc-11.2.0/bin/g++}"

# CUDA: Set default first, then find best match (avoid unbound variable)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
for cuda_ver in cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1 cuda-11.8; do
    if [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/$cuda_ver"
        break
    fi
done
export PATH="$CUDA_HOME/bin:$PATH"

export TORCHINDUCTOR_DISABLE_FLEX_ATTENTION=1
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}:/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python"

# Python binary: use conda env's Python directly (no activation = no sys.path pollution)
export PYTHON_BIN="$CONDA_ENV_PATH/bin/python"

# Verify CUDA availability before proceeding
echo "Checking CUDA availability..."
if ! "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available - check driver and PyTorch version'" 2>&1; then
    echo "ERROR: CUDA not available. GPU may be incompatible or driver issue."
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "CUDA_HOME: $CUDA_HOME"
    "$PYTHON_BIN" -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
    exit 1
fi
echo "CUDA available"

# sglang check and reinstall (in case of code changes)
"$PYTHON_BIN" -c "import sglang; print(f'sglang {sglang.__version__}')" 2>&1 || true
"$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1 || {
    cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/sgl-kernel
    "$PYTHON_BIN" -m pip install scikit-build-core cmake ninja --upgrade --quiet 2>&1 || true
    "$PYTHON_BIN" -m pip install -e . --no-build-isolation 2>&1 || true
    cd "$BENCHMARK_DIR"
}
# Reinstall sglang from source to pick up latest code changes
cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/python
"$PYTHON_BIN" -m pip install -e . --no-deps --quiet 2>&1 || true
cd "$BENCHMARK_DIR"
"$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1 || true

# Run pipeline
cd "$BENCHMARK_DIR"
bash "$BENCHMARK_DIR/run_pipeline.sh" "$EXP_TYPE"

# Copy output to home at the very end
cp "$OUT" "$LOG_DIR/slurm-${SLURM_JOB_ID:-$$}.out" 2>/dev/null || true
ls "$RESULT_DIR"/mwf_*.json 2>/dev/null | head -10 || true
