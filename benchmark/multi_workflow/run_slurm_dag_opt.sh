#!/bin/bash
###############################################################################
# SLURM Job Submission Script for DAG Priority v4 Verification
#
# This script runs the DAG Priority v4 experiment to verify:
# - Removed shared_boost (50x per workflow ref caused wrong eviction order)
# - role_type_boost * 100 (Tier-0/1 domination)
# - Critical path distance * 100 (critical path protection)
# - LRU+PF comparison baseline
#
# Priority v4 formula: priority = node.priority + role_type*100 + crit_distance*100
#
# Usage:
#   sbatch --parsable run_slurm_dag_opt.sh           # Submit with defaults (low + high)
#   sbatch --parsable run_slurm_dag_opt.sh low      # Low pressure only
#   sbatch --parsable run_slurm_dag_opt.sh high     # High pressure only
#
###############################################################################

#SBATCH --job-name=dag-priority-v4
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=hkbugpusrv08,hkbugpusrv09,hkbugpusrv15,hkbugpusrv16,hkbugpudgx01
#SBATCH --gres=gpu:a100:2          # 2x A100-80GB for tp-size=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@example.com

set -uo pipefail  # 不要用 -e，因为 pipeline 中的 tee 可能返回非0

# SLURM will write to NFS directly
NFS_OUT="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/slurm-dag-v4-${SLURM_JOB_ID:-$$}.out"
mkdir -p "$(dirname "$NFS_OUT")"
exec >"$NFS_OUT" 2>&1

BENCHMARK_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow"
RESULT_DIR="$LOG_DIR/results"
EXP_TYPE="${1:-all}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

echo "========================================"
echo "DAG Optimized KVFlow Verification"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Exp Type: ${EXP_TYPE}"
echo "========================================"

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

CONDA_ENV_PATH="/home/comp/25480812/.conda/envs/sglang-kvflow"
if [[ ! -d "$CONDA_ENV_PATH" ]]; then
    echo "ERROR: Conda env not found: $CONDA_ENV_PATH"
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
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python${PYTHONPATH:+:$PYTHONPATH}"

# Python binary
export PYTHON_BIN="$CONDA_ENV_PATH/bin/python"

# Verify sglang installation
"$PYTHON_BIN" -c "import sglang; print(f'sglang {sglang.__version__}')" 2>&1 || true
"$PYTHON_BIN" -c "from sgl_kernel import common_ops; print('sgl_kernel OK')" 2>&1 || {
    echo "WARNING: sgl_kernel not found, skipping reinstallation"
}

# Reinstall sglang from source to pick up latest code changes
echo "Reinstalling sglang-kvflow from source..."
cd /home/comp/25480812/CodeMAS_Project/sglang-kvflow/python
"$PYTHON_BIN" -m pip install -e . --no-deps --quiet 2>&1 || echo "WARNING: pip install had issues, continuing"

# After pip install, disable HTTP proxy to avoid ProxyError in server warmup
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""

# Verify run_dag_optimized.sh exists
if [[ ! -f "$BENCHMARK_DIR/run_dag_optimized.sh" ]]; then
    echo "ERROR: run_dag_optimized.sh not found at $BENCHMARK_DIR/run_dag_optimized.sh"
    exit 1
fi

# Check if run_dag_optimized.sh is readable
if ! head -1 "$BENCHMARK_DIR/run_dag_optimized.sh" > /dev/null 2>&1; then
    echo "ERROR: Cannot read run_dag_optimized.sh (possible NFS I/O error)"
    exit 1
fi

# Verify the installed code has our changes
echo "Verifying Priority v4 support..."
"$PYTHON_BIN" -c "
from sglang.srt.mem_cache.evict_policy import PriorityStrategy
from sglang.srt.mem_cache.radix_cache import TreeNode

# Check PriorityStrategy v4 has the correct fields
import inspect
src = inspect.getsource(PriorityStrategy.get_priority)
assert 'lock_ref' in src, 'lock_ref check not found in PriorityStrategy.get_priority'
assert 'critical_path_distance' in src, 'critical_path_distance not found in PriorityStrategy.get_priority'
# v4: role_type_boost*100 + crit_boost, no shared_boost
assert 'role_type_boost * 100' in src or 'role_type_boost*100' in src.replace(' ', ''), \
    'role_type_boost*100 not found in PriorityStrategy.get_priority'

# Check TreeNode has critical_path_distance
node = TreeNode()
assert hasattr(node, 'critical_path_distance'), 'critical_path_distance not found in TreeNode'

# Check protocol.py has critical_path_distance
import os
protocol_path = os.path.join(
    '/home/comp/25480812/CodeMAS_Project/sglang-kvflow/python/sglang/srt/entrypoints/openai',
    'protocol.py'
)
with open(protocol_path) as f:
    protocol_src = f.read()
assert 'critical_path_distance' in protocol_src, 'critical_path_distance not found in protocol.py'

print('All Priority v4 verification checks passed!')
" || {
    echo "ERROR: Priority v4 verification failed. Please ensure code changes are installed."
    exit 1
}

echo "========================================"
echo "Running DAG Optimized experiment..."
echo "========================================"

# Run the DAG optimized experiment
cd "$BENCHMARK_DIR"
bash "$BENCHMARK_DIR/run_dag_optimized.sh" "$EXP_TYPE"

echo "========================================"
echo "Experiment complete!"
echo "========================================"

echo "Logs: $NFS_OUT"
