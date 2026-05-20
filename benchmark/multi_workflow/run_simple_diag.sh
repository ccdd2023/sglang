#!/bin/bash
# Simple diagnostic script
#SBATCH --job-name=kvflow-diag
#SBATCH --time=00:20:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16

echo "=== Simple Diagnostic ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Date: $(date)"

echo ""
echo "=== Environment ==="
echo "HOME=$HOME"
echo "USER=$USER"
echo "CONDA_ENV_PATH=$CONDA_ENV_PATH"

echo ""
echo "=== File System ==="
ls -la /home/comp/25480812/CodeMAS_Project/logs/kvflow-smoke-test/ 2>&1 | head -5

echo ""
echo "=== Conda ==="
CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
if [[ -d "$CONDA_ENV_PATH" ]]; then
    echo "[OK] Conda env exists"
else
    echo "[FAIL] Conda env not found"
fi

echo ""
echo "=== Python ==="
PYTHON="$CONDA_ENV_PATH/bin/python"
if [[ -f "$PYTHON" ]]; then
    echo "[OK] Python exists: $PYTHON"
    $PYTHON --version
else
    echo "[FAIL] Python not found"
fi

echo ""
echo "=== SGLang ==="
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python:$PYTHONPATH"
$PYTHON -c "import sglang; print(f'SGLang: {sglang.__version__}')" 2>&1

echo ""
echo "=== Done ==="
