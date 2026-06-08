# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/swe_verified_200_instances.json`
- Cases: 4
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 4 | 1653.2 | 11511.5 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 4 | 1338.9 | 11511.5 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.5000 |
| kvflow_style_prefix_plus_hints | 4 | 1823.2 | 11514.5 | 2.0 | 0.0 | 1.00 | 4607.0 | 0.0 | 0.0 | 0.00 | 0.5000 |
| agenttemplatekv_exact_reuse | 4 | 1807.9 | 11512.5 | 2.0 | 0.0 | 1.00 | 14033.0 | 0.0 | 0.0 | 1.00 | 0.4376 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 696.23 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 898.02 | 10547 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2114.3 | 10550 | 1 | 4599 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2203.75 | 10548 | 1 | 14017 | 0 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-13033 | stock_sglang_prefix_only | 1479.02 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_baseline | 1480.27 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_plus_hints | 1497.56 | 10730 | 1 | 4605 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | agenttemplatekv_exact_reuse | 1502.22 | 10728 | 1 | 14029 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13236 | stock_sglang_prefix_only | 2213.73 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_baseline | 2207.25 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_plus_hints | 2226.01 | 11535 | 1 | 4613 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | agenttemplatekv_exact_reuse | 1678.93 | 11533 | 1 | 14045 | 0 | 0 | exact_code_content_signature | 0.6804 |
| astropy__astropy-13398 | stock_sglang_prefix_only | 2223.85 | 13240 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13398 | kvflow_style_prefix_baseline | 770.22 | 13240 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | kvflow_style_prefix_plus_hints | 1455.01 | 13243 | 1 | 4611 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | agenttemplatekv_exact_reuse | 1846.51 | 13241 | 1 | 14041 | 0 | 0 | exact_code_content_signature | 0.0702 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
