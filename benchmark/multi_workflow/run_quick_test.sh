#!/bin/bash
#SBATCH --job-name=kvflow-quick
#SBATCH --output=/tmp/kvflow-quick/slurm.out
#SBATCH --error=/tmp/kvflow-quick/slurm.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

WORK_DIR="/tmp/kvflow-quick"
mkdir -p "$WORK_DIR"

exec > "$WORK_DIR/main.log" 2>&1

echo "[START] Job started"

# Just test if Python works on local
export PATH="/home/comp/25480812/.conda/envs/sglang-kvflow/bin:$PATH"
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python"

echo "[TEST] Testing Python..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo "[TEST] Done"
