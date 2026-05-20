#!/bin/bash
# =============================================================================
# Environment Setup for KVFlow Benchmark
#
# This script sets up the conda environment needed to run the KVFlow benchmark.
# Run this ONCE before submitting jobs, or include it in your Slurm script.
#
# Usage:
#   bash setup_kvflow_env.sh              # Interactive setup
#   conda env create -f environment.yml   # Or use conda directly
#
# =============================================================================

set -euo pipefail

SGLANG_ROOT="/home/gfy/CodeMAS_Project/sglang-kvflow"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
ENV_NAME="${ENV_NAME:-sglang-kvflow}"

echo "=============================================="
echo "KVFlow Environment Setup"
echo "=============================================="
echo "SGLang root: $SGLANG_ROOT"
echo "Python version: $PYTHON_VERSION"
echo "Environment name: $ENV_NAME"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "You can download it from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Using existing environment."
        echo "To activate: conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create environment
echo "Creating conda environment..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate and install
echo "Activating environment..."
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flashinfer dependencies
echo "Installing FlashInfer..."
pip install flashinfer

# Install SGLang from source
echo "Installing SGLang from source..."
cd "$SGLANG_ROOT/python"
pip install -e .

# Install additional dependencies for benchmark
echo "Installing benchmark dependencies..."
pip install aiohttp

# Go back to original directory
cd "$SGLANG_ROOT"

echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the installation:"
echo "  python -c 'import sglang; print(sglang.__version__)'"
echo "=============================================="
