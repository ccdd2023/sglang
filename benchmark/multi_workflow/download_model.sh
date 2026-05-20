#!/bin/bash
#SBATCH --job-name=download-qwen3-8b
#SBATCH --output=/home/comp/25480812/logs/model-download-%j.out
#SBATCH --error=/home/comp/25480812/logs/model-download-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

CONDA_ENV="/home/comp/25480812/.conda/envs/sglang-kvflow"
PYTHON_BIN="$CONDA_ENV/bin/python"
MODEL_DIR="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"

mkdir -p /home/comp/25480812/logs

echo "Downloading Qwen3-8B model..."
echo "Target: $MODEL_DIR"

$PYTHON_BIN -c "
from huggingface_hub import snapshot_download
import os

os.makedirs('$MODEL_DIR', exist_ok=True)
snapshot_download(
    repo_id='Qwen/Qwen3-8B',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('Download complete!')
"

echo "Model files:"
ls -lh "$MODEL_DIR/" | head -20
