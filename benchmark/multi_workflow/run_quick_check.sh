#!/bin/bash
# Quick smoke test with fixed CUDA compatibility
#SBATCH --job-name=kvflow-smoke
#SBATCH --time=00:20:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=hkbugpusrv07,hkbugpusrv08,hkbugpusrv15
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32

set -uo pipefail

OUT="/tmp/slurm-${SLURM_JOB_ID}.out"
exec >"$OUT" 2>&1

LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-smoke-test"
mkdir -p "$LOG_DIR"

echo "=== KVFlow Smoke Test ==="
echo "Job: $SLURM_JOB_ID, Node: $(hostname)"

# Environment setup
CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
for cuda_ver in cuda-12.8 cuda-12.6 cuda-12.4 cuda-12.1; do
    if [[ -d "/usr/local/$cuda_ver" ]] && [[ -f "/usr/local/$cuda_ver/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/$cuda_ver"
        break
    fi
done
export PATH="$CUDA_HOME/bin:$PATH"

export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}:/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python"
PYTHON="$CONDA_ENV_PATH/bin/python"

echo "CUDA: $CUDA_HOME"

# GPU info
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# CUDA check
echo "Checking CUDA..."
if ! $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>&1; then
    echo "ERROR: CUDA not available"
    $PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
    exit 1
fi
echo "CUDA OK"

# SGLang check
echo "Checking SGLang..."
$PYTHON -c "import sglang; print(f'SGLang: {sglang.__version__}')"

# Model check
echo "Checking model..."
if [[ -d "$MODEL_PATH" ]]; then
    echo "Model OK"
else
    echo "ERROR: Model not found"
    exit 1
fi

echo "=== All checks passed ==="

# Cleanup and exit
cp "$OUT" "$LOG_DIR/slurm-${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "Done at $(date)"
