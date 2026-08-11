#!/usr/bin/env bash
#SBATCH --job-name=impactkv-bootstrap
#SBATCH --partition=debug
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

"$IMPACTKV_MINI_VENV/bin/hf" download \
  cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit \
  --revision 4bd30395b72ea6045edd04806c4fea448d4467b3 \
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
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
        raise RuntimeError(f"model metadata mismatch: {path}")
print("MODEL_METADATA_OK")
PY

"$IMPACTKV_MINI_PYTHON" - <<'PY'
import importlib.metadata
import litellm
import minisweagent
print("MINISWEAGENT", importlib.metadata.version("mini-swe-agent"))
print("LITELLM", importlib.metadata.version("litellm"))
PY
quota -s || true
