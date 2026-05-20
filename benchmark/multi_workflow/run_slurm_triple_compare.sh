#!/bin/bash
# =============================================================================
# SLURM Job Submission Script for vLLM vs SGLang vs KVFlow Triple Comparison
#
# Usage:
#   sbatch run_slurm_triple_compare.sh           # Run all 4 configs
#   sbatch run_slurm_triple_compare.sh vllm     # Run only vLLM
#   sbatch run_slurm_triple_compare.sh kvflow   # Run only KVFlow
#
# 4 Configurations:
#   vllm           - vLLM baseline with LRU, CPU offload
#   sglang         - SGLang with LRU, CPU offload (no HiCache)
#   sglang_hicache - SGLang with LRU + HiCache
#   kvflow         - SGLang with Priority + HiCache (full KVFlow)
#
# Key: All output goes to /tmp first, copied to home at the end.
# =============================================================================

#SBATCH --job-name=kvflow-triple-compare
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=hkbugpusrv08,hkbugpusrv15
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@example.com

set -euo pipefail

# All output → /tmp (local tmpfs, no NFS)
OUT="/tmp/slurm-${SLURM_JOB_ID:-$$}.out"
exec >"$OUT" 2>&1

BENCHMARK_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow"
RESULT_DIR="$LOG_DIR/results"
EXP_TYPE="${1:-all}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Clear flashinfer JIT cache
rm -rf /home/comp/25480812/.cache/flashinfer 2>/dev/null || true

# GPU info
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

SGLANG_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
VLLM_ENV_PATH="/home/comp/25480812/.conda/envs/vllm"

if [[ ! -d "$SGLANG_ENV_PATH" ]]; then
    echo "ERROR: sglang-kvflow environment not found at $SGLANG_ENV_PATH"
    exit 1
fi

# GCC 11.2.0
export CC="/usr/local/gcc/gcc-11.2.0/bin/gcc"
export CXX="/usr/local/gcc/gcc-11.2.0/bin/g++"

# CUDA
for cuda_ver in cuda-13 cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1; do
    if [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/$cuda_ver"
        export PATH="$CUDA_HOME/bin:$PATH"
        break
    fi
done

export TORCHINDUCTOR_DISABLE_FLEX_ATTENTION=1
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# vLLM environment setup (may need runtime installation)
setup_vllm_env() {
    if [[ -d "$VLLM_ENV_PATH/bin/python" ]]; then
        export VLLM_PYTHON="$VLLM_ENV_PATH/bin/python"
        export VLLM_PIP="$VLLM_ENV_PATH/bin/pip"
        export VLLM_PATH="$VLLM_ENV_PATH"
    else
        # Create vLLM environment if it doesn't exist
        echo "Creating vLLM environment..."
        conda create --clone "$SGLANG_ENV_PATH" -p "$VLLM_ENV_PATH" -y 2>&1 || true
        if [[ -d "$VLLM_ENV_PATH/bin/python" ]]; then
            export VLLM_PYTHON="$VLLM_ENV_PATH/bin/python"
            export VLLM_PIP="$VLLM_ENV_PATH/bin/pip"
            export VLLM_PATH="$VLLM_ENV_PATH"
        else
            echo "WARNING: vLLM environment creation failed, will use sglang-kvflow for vLLM server"
            export VLLM_PYTHON="$SGLANG_ENV_PATH/bin/python"
            export VLLM_PIP="$SGLANG_ENV_PATH/bin/pip"
            export VLLM_PATH="$SGLANG_ENV_PATH"
        fi
    fi
}

# SGLang environment setup
setup_sglang_env() {
    export PATH="$SGLANG_ENV_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$SGLANG_ENV_PATH/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHON_BIN="$SGLANG_ENV_PATH/bin/python"
}

# Detect NVIDIA driver version and install compatible PyTorch
fix_pytorch_version() {
    echo "[$(date '+%H:%M:%S')] Detecting NVIDIA driver version..."
    
    # Get driver version from nvidia-smi
    local driver_ver
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    echo "[$(date '+%H:%M:%S')] Driver version: $driver_ver"
    
    # Determine compatible CUDA version
    # Driver 525+ supports CUDA 12.x, 535+ supports CUDA 13.x
    local cuda_ver="cu121"
    if [[ -n "$driver_ver" ]]; then
        if [[ "$driver_ver" -ge 535 ]]; then
            cuda_ver="cu130"
        elif [[ "$driver_ver" -ge 525 ]]; then
            cuda_ver="cu121"
        else
            cuda_ver="cu118"
        fi
    fi
    echo "[$(date '+%H:%M:%S')] Will use PyTorch with $cuda_ver"
    
    # Check current PyTorch CUDA version
    local current_cuda=$("$PYTHON_BIN" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo "[$(date '+%H:%M:%S')] Current PyTorch CUDA: $current_cuda"
    
    # If version mismatch, reinstall PyTorch
    if [[ "$current_cuda" != "$cuda_ver" ]] && [[ "$current_cuda" != "unknown" ]]; then
        echo "[$(date '+%H:%M:%S')] Reinstalling PyTorch for $cuda_ver..."
        "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$cuda_ver" --upgrade --quiet 2>&1 | tail -5
    fi
}

# Rebuild sgl_kernel to fix undefined symbol errors
rebuild_sgl_kernel() {
    echo "[$(date '+%H:%M:%S')] Rebuilding sgl_kernel for current PyTorch/CUDA environment..."
    
    cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/sgl-kernel
    
    # Uninstall and reinstall with fresh build
    "$PYTHON_BIN" -m pip uninstall sglang-kernel -y 2>&1 || true
    "$PYTHON_BIN" -m pip install scikit-build-core cmake ninja --upgrade --quiet 2>&1 || true
    
    # Build with explicit PyTorch paths
    export TORCH_CUDA_ARCH_LIST="8.0"  # A100 = SM80
    "$PYTHON_BIN" -m pip install -e . --no-build-isolation 2>&1 | tail -20
    
    # Verify
    if "$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1; then
        echo "[$(date '+%H:%M:%S')] sgl_kernel rebuild successful"
    else
        echo "[$(date '+%H:%M:%S')] sgl_kernel rebuild FAILED - trying alternative method"
        # Fallback: install pre-built wheel
        "$PYTHON_BIN" -m pip install sglang-kernel --upgrade --index-url https://wheels.sglang.ai/site/simple 2>&1 | tail -10
    fi
    
    cd "$BENCHMARK_DIR"
}

# Verify sglang installation and rebuild if needed
setup_sglang_env
fix_pytorch_version  # Fix PyTorch/CUDA compatibility first

if ! "$PYTHON_BIN" -c "import sglang; print(f'sglang {sglang.__version__}')" 2>&1; then
    echo "ERROR: sglang import failed"
fi
if ! "$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1; then
    rebuild_sgl_kernel
fi
# Reinstall sglang from source
cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/python
"$PYTHON_BIN" -m pip install -e . --no-deps --quiet 2>&1 || true
cd "$BENCHMARK_DIR"

# Run the triple comparison pipeline
cd "$BENCHMARK_DIR"
bash "$BENCHMARK_DIR/run_pipeline_triple_compare.sh" "$EXP_TYPE"

# Copy output to home at the very end
cp "$OUT" "$LOG_DIR/slurm-triple-${SLURM_JOB_ID:-$$}.out" 2>/dev/null || true
ls "$RESULT_DIR"/mwf_*_triple_*.json 2>/dev/null | head -10 || true
