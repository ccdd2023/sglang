#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gfy/CodeMAS_Project
REPO="${ROOT}/sglang-kvflow-worktrees/coding-aware"
CACHEBLEND="${ROOT}/kvflow-reproductions/worktrees/cacheblend-qwen2"
PY=/home/gfy/.conda/envs/cacheblend-repro-20260719/bin/python
MODEL=/home/gfy/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5
SOURCE="${ROOT}/kvflow-reproductions/qcfuse-official/data/coding_200_qwen3_8b_5k"
ARTIFACT="${ROOT}/kvflow-artifacts/impactkv_lossy_kv_coding_matrix_20260728/cacheblend_native"
ADAPTER="${REPO}/benchmark/multi_workflow/cacheblend_coding_matrix.py"
RUNNER="${CACHEBLEND}/example/repro_common.py"

SIZE="${SIZE:-200}"
DATASETS="${DATASETS:-lcc repobench-p}"
ARMS="${ARMS:-dense reuse}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${CACHEBLEND}/vllm_blend"
export KVFLOW_ENGINE_COMMIT
KVFLOW_ENGINE_COMMIT="$(git -C "${CACHEBLEND}" rev-parse HEAD)"

# Prevent a user-site Torch upgrade from being mixed with CacheBlend's
# vLLM 0.4.1 extension.  This import does not create a model or allocate VRAM.
"${PY}" -c 'import torch, vllm; assert vllm.__version__ == "0.4.1"'

for dataset in ${DATASETS}; do
  output_dir="${ARTIFACT}/${dataset}"
  workload="${output_dir}/WORKLOAD.json"
  mkdir -p "${output_dir}"
  "${PY}" "${ADAPTER}" prepare \
    --source "${SOURCE}/${dataset}.jsonl" \
    --dataset "${dataset}" \
    --limit "${SIZE}" \
    --output "${workload}"

  for arm in ${ARMS}; do
    metrics="${output_dir}/${arm}.jsonl"
    complete="${output_dir}/${arm}.complete"
    if [[ -f "${complete}" ]]; then
      continue
    fi
    if [[ -e "${metrics}" ]]; then
      echo "Refusing to append to incomplete ledger: ${metrics}" >&2
      echo "Move that ledger aside before explicitly retrying this arm." >&2
      exit 2
    fi
    mode="${arm}"
    args=(
      --workload "${workload}"
      --metrics "${metrics}"
      --model "${MODEL}"
      --mode "${mode}"
      --phase accuracy
      --split formal
      --limit 0
      --run-id "cacheblend-${dataset}-${arm}-20260728"
      --gpu-memory-utilization 0.85
    )
    if [[ "${arm}" == "reuse" ]]; then
      args+=(--recompute-ratio "${RECOMPUTE_RATIO}")
    fi
    "${PY}" "${RUNNER}" "${args[@]}"
    touch "${complete}"
  done

  if [[ -f "${output_dir}/dense.complete" && -f "${output_dir}/reuse.complete" ]]; then
    "${PY}" "${ADAPTER}" summarize \
      --workload "${workload}" \
      --dense "${output_dir}/dense.jsonl" \
      --reuse "${output_dir}/reuse.jsonl" \
      --recompute-ratio "${RECOMPUTE_RATIO}" \
      --output "${output_dir}/RESULT.json"
  fi
done
