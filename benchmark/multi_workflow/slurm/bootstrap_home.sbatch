#!/usr/bin/env bash
#SBATCH --job-name=impactkv-bootstrap
#SBATCH --partition=debug
#SBATCH --nodelist=gpu11
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
source "$HOME/CodeMAS_Project/sglang/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
stage="$IMPACTKV_HOME/migration-staging/gpuhome-20260811"
base_python="$HOME/miniconda3/envs/sglang-kvflow/bin/python"

if [ ! -x "$IMPACTKV_MINI_PYTHON" ]; then
  "$base_python" -m venv "$IMPACTKV_MINI_VENV"
fi
"$IMPACTKV_MINI_PYTHON" -m pip install \
  --requirement "$stage/mini-swe-agent-pip-freeze.txt"

"$HOME/miniconda3/envs/sglang-kvflow/bin/hf" download \
  cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit \
  --revision 2831070b7b8c7aa6b7012333c6c4a2bd257f6cdf \
  --local-dir "$IMPACTKV_MODEL"

STAGE="$stage" "$base_python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

stage = Path(os.environ["STAGE"])
manifest = json.loads((stage / "MIGRATION_MANIFEST.json").read_text())
model = Path(os.environ["IMPACTKV_MODEL"])
for name, expected in manifest["model"]["files"].items():
    path = model / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
        raise RuntimeError(f"model metadata mismatch: {path}")
print("MODEL_METADATA_OK")
PY

"$IMPACTKV_MINI_PYTHON" - <<'PY'
import litellm
import minisweagent
print("MINISWEAGENT", minisweagent.__version__)
print("LITELLM", litellm.__version__ if hasattr(litellm, "__version__") else "installed")
PY
quota -s || true
