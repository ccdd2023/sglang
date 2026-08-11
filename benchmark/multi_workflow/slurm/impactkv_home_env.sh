#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  slurm_node="${SLURMD_NODENAME:-$(hostname -s)}"
  case "$slurm_node" in
    gpu10|gpu11|gpu12|gpu13|gpu23|gpu24)
      echo "ImpactKV refuses unsupported Slurm node: $slurm_node" >&2
      exit 78
      ;;
  esac

  cluster_proxy="${IMPACTKV_CLUSTER_PROXY:-http://proxy.comp.hkbu.edu.hk:8080}"
  export HTTP_PROXY="$cluster_proxy"
  export HTTPS_PROXY="$cluster_proxy"
  export http_proxy="$cluster_proxy"
  export https_proxy="$cluster_proxy"
  unset ALL_PROXY all_proxy
  export NO_PROXY="localhost,127.0.0.1,::1,.local"
  export no_proxy="$NO_PROXY"
fi

export IMPACTKV_HOME="${IMPACTKV_HOME:-$HOME/CodeMAS_Project}"
export IMPACTKV_PROJECT="${IMPACTKV_PROJECT:-$IMPACTKV_HOME/sglang}"
export IMPACTKV_ARTIFACTS="${IMPACTKV_ARTIFACTS:-$IMPACTKV_HOME/kvflow-artifacts}"
export IMPACTKV_REPORTS="${IMPACTKV_REPORTS:-$IMPACTKV_HOME/kvflow-reports}"
export IMPACTKV_RUNTIME_ROOT="${IMPACTKV_RUNTIME_ROOT:-$HOME/impactkv-runtime}"
export IMPACTKV_MODEL="${IMPACTKV_MODEL:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit}"
export IMPACTKV_MINI_VENV="${IMPACTKV_MINI_VENV:-$HOME/.venvs/mini-swe-agent-v2.3.0}"
export IMPACTKV_MINI_PYTHON="${IMPACTKV_MINI_PYTHON:-$IMPACTKV_MINI_VENV/bin/python}"
export IMPACTKV_MINI="${IMPACTKV_MINI:-$IMPACTKV_MINI_VENV/bin/mini-extra}"
export IMPACTKV_EVAL_PYTHON="${IMPACTKV_EVAL_PYTHON:-$HOME/miniconda3/envs/sglang-kvflow/bin/python}"
export PATH="$(dirname "$IMPACTKV_EVAL_PYTHON"):$PATH"
export IMPACTKV_DATASET_ROOT="${IMPACTKV_DATASET_ROOT:-$IMPACTKV_ARTIFACTS/swebench_verified_bridge_v1_20260724/minisweagent_dataset}"
export IMPACTKV_EVAL_SNAPSHOT="${IMPACTKV_EVAL_SNAPSHOT:-$IMPACTKV_ARTIFACTS/swebench_verified_bridge_v1_20260724/frozen_subset.json}"

job_key="${SLURM_JOB_ID:-interactive}"
export ENROOT_CONFIG_PATH="$IMPACTKV_RUNTIME_ROOT/enroot/config"
export ENROOT_CACHE_PATH="$IMPACTKV_RUNTIME_ROOT/enroot/cache"
export ENROOT_DATA_PATH="$IMPACTKV_RUNTIME_ROOT/enroot/data"
export ENROOT_TEMP_PATH="$IMPACTKV_RUNTIME_ROOT/enroot/tmp"
export ENROOT_RUNTIME_PATH="$IMPACTKV_RUNTIME_ROOT/enroot/run/$job_key"
export IMPACTKV_ENROOT_RUNTIME_BASE="$IMPACTKV_RUNTIME_ROOT/enroot/run/$job_key"
export IMPACTKV_ENROOT_IMAGE_DIR="$IMPACTKV_RUNTIME_ROOT/enroot/images"
export IMPACTKV_ENROOT_IMAGE_INDEX="$IMPACTKV_ENROOT_IMAGE_DIR/IMAGE_INDEX.json"
export TMPDIR="$IMPACTKV_RUNTIME_ROOT/tmp/$job_key"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_RUNTIME_DIR="$IMPACTKV_RUNTIME_ROOT/xdg/$job_key"
export HF_HOME="${HF_HOME:-$HOME/.cache/impactkv/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/impactkv/pip}"
export PYTHONPATH="$IMPACTKV_PROJECT/python:$IMPACTKV_PROJECT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p \
  "$IMPACTKV_ARTIFACTS" \
  "$IMPACTKV_REPORTS" \
  "$IMPACTKV_RUNTIME_ROOT/logs" \
  "$ENROOT_CONFIG_PATH" \
  "$ENROOT_CACHE_PATH" \
  "$ENROOT_DATA_PATH" \
  "$ENROOT_TEMP_PATH" \
  "$ENROOT_RUNTIME_PATH" \
  "$IMPACTKV_ENROOT_IMAGE_DIR" \
  "$TMPDIR" \
  "$XDG_RUNTIME_DIR" \
  "$HF_HOME" \
  "$PIP_CACHE_DIR"
