#!/bin/bash
# =============================================================================
# SLURM Job Submission Script for KVFlow Benchmark - FIXED VERSION
#
# Key fix: Use local miniconda environment instead of shared home directory
# which has I/O errors on compute nodes.
#
# Usage:
#   sbatch run_slurm_fixed.sh              # Submit job with default settings
#   sbatch run_slurm_fixed.sh quick        # Quick smoke test
#
# =============================================================================

#SBATCH --job-name=kvflow-fixed
#SBATCH --output=/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/slurm-%j.out
#SBATCH --error=/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@example.com

set -euo pipefail

BENCHMARK_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow"
RESULT_DIR="$LOG_DIR/results"

echo "=============================================="
echo "SLURM Job Started (Fixed Version)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs allocated: $SLURM_JOB_GPUS"
echo "=============================================="

# Verify GPU is accessible
echo ""
echo "[Setup] Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 | head -4 || echo "WARNING: nvidia-smi failed"
echo ""

# Setup environment - use local miniconda to avoid home directory I/O errors
MINICONDA_BASE="/usr/local/miniconda/py312_24.7.1-0"
LOCAL_CONDA_ENV="/tmp/sglang-kvflow-env"  # Persistent local env across jobs

echo "[Setup] Using local miniconda: $MINICONDA_BASE"

# Set CUDA environment
for cuda_ver in cuda-13 cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1; do
    if [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/$cuda_ver"
        export PATH="$CUDA_HOME/bin:$PATH"
        echo "[Setup] CUDA_HOME=$CUDA_HOME"
        break
    fi
done

export TORCHINDUCTOR_DISABLE_FLEX_ATTENTION=1
export CMAKE_POLICY_VERSION_MINIMUM=3.5

cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow

# Check if we already have a working installation in the persistent local env
if [[ -d "$LOCAL_CONDA_ENV" ]]; then
    PYTHON_BIN="$LOCAL_CONDA_ENV/bin/python"
    echo "[Setup] Checking existing local env at $LOCAL_CONDA_ENV..."
    
    if "$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1 | grep -q "OK"; then
        echo "[Setup] Found working sgl_kernel installation - skipping rebuild!"
    else
        echo "[Setup] Existing env broken, rebuilding..."
        rm -rf "$LOCAL_CONDA_ENV"
        unset PYTHON_BIN
    fi
fi

# Create local conda environment if needed
if [[ -z "${PYTHON_BIN:-}" ]]; then
    echo "[Setup] Creating local conda environment..."
    "$MINICONDA_BASE/bin/conda" create -y -p "$LOCAL_CONDA_ENV" python=3.12 2>&1 | tail -5
    PYTHON_BIN="$LOCAL_CONDA_ENV/bin/python"
fi

# Verify Python
echo "[Setup] Python version:"
"$PYTHON_BIN" --version

echo "[Setup] Checking PyTorch..."
PYTORCH_VER=$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>&1)
CUDA_VER=$("$PYTHON_BIN" -c "import torch; print(torch.version.cuda)" 2>&1)
echo "  PyTorch: $PYTORCH_VER"
echo "  CUDA: $CUDA_VER"

# Install dependencies only if not already properly installed
echo "[Setup] Checking sgl_kernel..."
if ! "$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('OK')" 2>&1 | grep -q "OK"; then
    echo "[Setup] sgl_kernel not found or broken, installing..."
    
    # Install build tools
    "$PYTHON_BIN" -m pip install scikit-build-core cmake ninja --upgrade --quiet 2>&1 | tail -3
    
    # Uninstall any existing sgl/sglang
    "$PYTHON_BIN" -m pip uninstall sgl sglang sgl-kernel -y 2>&1 | tail -3 || true
    
    # Build and install sgl-kernel
    echo "[Setup] Building sgl-kernel (this may take a few minutes)..."
    if [[ -d "sgl-kernel" ]]; then
        cd sgl-kernel
        "$PYTHON_BIN" -m pip install -e . --no-build-isolation 2>&1 | tail -10
        cd ..
    fi
    
    # Install sglang from source
    echo "[Setup] Installing sglang from source..."
    "$PYTHON_BIN" -m pip install -e ./python --no-deps 2>&1 | tail -5
else
    echo "[Setup] sgl_kernel already working - using cached installation!"
fi

# Final verification
echo "[Setup] Final verification..."
if "$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1 | grep -q "OK"; then
    echo "[Setup] ✓ sgl_kernel verified"
else
    echo "[Setup] ✗ sgl_kernel verification failed"
fi

if "$PYTHON_BIN" -c "import sglang; print(f'sglang {sglang.__version__}')" 2>&1 | grep -q "sglang"; then
    echo "[Setup] ✓ sglang verified"
else
    echo "[Setup] ✗ sglang verification failed"
fi

# Set PYTHONPATH to use local sglang-kvflow code
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"

echo "[Setup] Environment ready:"
echo "  PYTHON_BIN: $PYTHON_BIN"
echo "  PYTHONPATH: ${PYTHONPATH:-<not set>}"
echo ""

# Create directories
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Run the benchmark pipeline
EXP_TYPE="${1:-quick}"

echo "[$(date '+%H:%M:%S')] Starting benchmark: $EXP_TYPE"
echo "=============================================="

cd "$BENCHMARK_DIR"
PYTHON_BIN="$PYTHON_BIN" bash "$BENCHMARK_DIR/run_pipeline.sh" "$EXP_TYPE"

echo ""
echo "[$(date '+%H:%M:%S')] Benchmark complete!"
echo "=============================================="
echo "Results saved to: $RESULT_DIR"
ls -lh "$RESULT_DIR"/mwf_*.json 2>/dev/null || echo "No result files found"
echo "=============================================="
echo "SLURM Job Finished"
echo "=============================================="
