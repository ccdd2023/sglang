#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/gfy/.conda/envs/sglang-kvflow/bin/python}"
PORT_BASE="${PORT_BASE:-31000}"

run_cpu_experiments() {
  "$PY" benchmark/multi_workflow/bench_template_codebase_segments.py
  "$PY" benchmark/multi_workflow/build_nearmatch_safety_expansion.py \
    --max-negative-pairs 500 \
    --out-dir results/kvcomm_ablation_package
}

require_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "GPU runtime is not available in this shell; E1/E2/E5 require SGLang serving." >&2
    echo "Run this script in a GPU-visible session where nvidia-smi succeeds." >&2
    exit 2
  fi
}

run_e1_smoke() {
  "$PY" benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
    --model /home/gfy/models/Qwen2.5-32B-Instruct-GPTQ-Int4 \
    --manifest results/repo_level_datasets/manifest_30.json \
    --dataset results/swebench_local_envs/expanded_30_discriminative_instances.json \
    --max-cases 3 \
    --output-schema json-edit \
    --server-timeout 600 \
    --port "$((PORT_BASE + 1))" \
    --out-dir results/swe_generated_patch_kvcomm/qwen2_5_32b_gptq_json_30_smoke
}

run_e1_full() {
  "$PY" benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
    --model /home/gfy/models/Qwen2.5-32B-Instruct-GPTQ-Int4 \
    --manifest results/repo_level_datasets/manifest_30.json \
    --dataset results/swebench_local_envs/expanded_30_discriminative_instances.json \
    --max-cases 28 \
    --output-schema json-edit \
    --server-timeout 600 \
    --port "$((PORT_BASE + 2))" \
    --emit-ttft \
    --out-dir results/swe_generated_patch_kvcomm/qwen2_5_32b_gptq_json_30_ttft_20260615
}

run_e1_smallctx_smoke() {
  "$PY" benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
    --model /home/gfy/models/Qwen2.5-32B-Instruct-GPTQ-Int4 \
    --manifest results/repo_level_datasets/manifest_30.json \
    --dataset results/swebench_local_envs/expanded_30_discriminative_instances.json \
    --max-cases 3 \
    --files-per-case 1 \
    --max-file-chars 3000 \
    --max-tokens 512 \
    --output-schema json-edit \
    --server-timeout 600 \
    --port "$((PORT_BASE + 6))" \
    --out-dir results/swe_generated_patch_kvcomm/qwen2_5_32b_gptq_json_30_smallctx_smoke
}

run_e1_smallctx_full() {
  "$PY" benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
    --model /home/gfy/models/Qwen2.5-32B-Instruct-GPTQ-Int4 \
    --manifest results/repo_level_datasets/manifest_30.json \
    --dataset results/swebench_local_envs/expanded_30_discriminative_instances.json \
    --max-cases 28 \
    --files-per-case 1 \
    --max-file-chars 3000 \
    --max-tokens 512 \
    --output-schema json-edit \
    --server-timeout 600 \
    --port "$((PORT_BASE + 7))" \
    --out-dir results/swe_generated_patch_kvcomm/qwen2_5_32b_gptq_json_30_smallctx
}

run_e2_smoke() {
  "$PY" benchmark/multi_workflow/bench_coding_kvflow_prefetch.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --dataset results/repo_level_datasets/swe_verified_100_instances.json \
    --manifest results/repo_level_datasets/manifest_100.json \
    --max-cases 5 \
    --files-per-case 3 \
    --disable-hierarchical-cache \
    --port "$((PORT_BASE + 3))" \
    --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_smoke
}

run_e2_full() {
  "$PY" benchmark/multi_workflow/bench_coding_kvflow_prefetch.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --dataset results/repo_level_datasets/swe_verified_100_instances.json \
    --manifest results/repo_level_datasets/manifest_100.json \
    --max-cases 100 \
    --files-per-case 3 \
    --disable-hierarchical-cache \
    --port "$((PORT_BASE + 4))" \
    --emit-ttft \
    --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_ttft_20260615
}

run_e5_smoke() {
  "$PY" benchmark/multi_workflow/bench_coding_kvflow_prefetch.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --dataset results/repo_level_datasets/swe_verified_30_instances.json \
    --manifest results/repo_level_datasets/manifest_30.json \
    --max-cases 3 \
    --files-per-case 3 \
    --hicache-storage-backend file \
    --port "$((PORT_BASE + 5))" \
    --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_file_backend_smoke
}

run_e6_smoke() {
  "$PY" benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --manifest results/repo_level_datasets/manifest_500.json \
    --max-cases 3 \
    --length-buckets 8000,16000 \
    --max-token-settings 1 \
    --disable-hierarchical-cache \
    --skip-e7 --skip-e8 \
    --port "$((PORT_BASE + 8))" \
    --out-dir results/kvcomm_ttft_stress/qwen2_5_7b_smoke
}

run_e6() {
  "$PY" benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --manifest results/repo_level_datasets/manifest_500.json \
    --max-cases 50 \
    --length-buckets 8000,16000 \
    --disable-hierarchical-cache \
    --skip-e7 --skip-e8 \
    --port "$((PORT_BASE + 9))" \
    --out-dir results/kvcomm_ttft_stress/qwen2_5_7b
}

run_e7_smoke() {
  "$PY" benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --manifest results/repo_level_datasets/manifest_500.json \
    --skip-e6 --skip-e8 \
    --agent-max-cases 3 \
    --agent-counts 2 \
    --segment-counts 1 \
    --agent-length-buckets 8000 \
    --disable-hierarchical-cache \
    --port "$((PORT_BASE + 10))" \
    --out-dir results/kvcomm_ttft_stress/qwen2_5_7b_e7_smoke
}

run_e7() {
  "$PY" benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --manifest results/repo_level_datasets/manifest_500.json \
    --skip-e6 --skip-e8 \
    --length-buckets 8000,16000 \
    --disable-hierarchical-cache \
    --port "$((PORT_BASE + 11))" \
    --out-dir results/kvcomm_ttft_stress/qwen2_5_7b_e7
}

run_e8() {
  "$PY" benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --manifest results/repo_level_datasets/manifest_500.json \
    --skip-e6 --skip-e7 \
    --e8-length 16000 \
    --length-buckets 8000,16000 \
    --disable-hierarchical-cache \
    --port "$((PORT_BASE + 12))" \
    --out-dir results/kvcomm_ttft_stress/qwen2_5_7b_e8
}

run_ttft_all() {
  "$PY" benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --model /home/gfy/models/Qwen2.5-7B-Instruct \
    --manifest results/repo_level_datasets/manifest_500.json \
    --max-cases 50 \
    --length-buckets 8000,16000 \
    --disable-hierarchical-cache \
    --port "$((PORT_BASE + 13))" \
    --out-dir results/kvcomm_ttft_stress/qwen2_5_7b
}

case "${1:-all}" in
  cpu)
    run_cpu_experiments
    ;;
  e1-smoke)
    require_gpu
    run_e1_smoke
    ;;
  e1)
    require_gpu
    run_e1_smoke
    run_e1_full
    ;;
  e1-smallctx)
    require_gpu
    run_e1_smallctx_smoke
    run_e1_smallctx_full
    ;;
  e2-smoke)
    require_gpu
    run_e2_smoke
    ;;
  e2)
    require_gpu
    run_e2_smoke
    run_e2_full
    ;;
  e5-smoke)
    require_gpu
    run_e5_smoke
    ;;
  e6-smoke)
    require_gpu
    run_e6_smoke
    ;;
  e6)
    require_gpu
    run_e6_smoke
    run_e6
    ;;
  e7-smoke)
    require_gpu
    run_e7_smoke
    ;;
  e7)
    require_gpu
    run_e7_smoke
    run_e7
    ;;
  e8)
    require_gpu
    run_e8
    ;;
  ttft-all)
    require_gpu
    run_ttft_all
    ;;
  all)
    require_gpu
    run_cpu_experiments
    run_e1_smallctx_smoke
    run_e1_smallctx_full
    run_e2_smoke
    run_e2_full
    run_e5_smoke || true
    run_ttft_all
    ;;
  *)
    echo "usage: $0 [cpu|e1-smoke|e1|e1-smallctx|e2-smoke|e2|e5-smoke|e6-smoke|e6|e7-smoke|e7|e8|ttft-all|all]" >&2
    exit 64
    ;;
esac

"$PY" paper/scripts/generate_paper_figures.py
