# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/swe_verified_200_instances.json`
- Cases: 6
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 6 | 1808.0 | 11724.3 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 6 | 1600.8 | 11724.3 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.6667 |
| kvflow_style_prefix_plus_hints | 6 | 1957.0 | 11727.3 | 2.0 | 0.0 | 1.00 | 4608.3 | 0.0 | 0.0 | 0.00 | 0.6204 |
| agenttemplatekv_exact_reuse | 6 | 1957.3 | 11725.3 | 2.0 | 0.0 | 1.00 | 14035.7 | 0.0 | 0.0 | 1.00 | 0.5788 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 691.62 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 900.83 | 10547 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2112.97 | 10550 | 1 | 4599 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2225.5 | 10548 | 1 | 14017 | 0 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-13033 | stock_sglang_prefix_only | 1479.02 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_baseline | 1487.95 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_plus_hints | 1493.63 | 10730 | 1 | 4605 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | agenttemplatekv_exact_reuse | 1485.04 | 10728 | 1 | 14029 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13236 | stock_sglang_prefix_only | 2207.41 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_baseline | 2204.02 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_plus_hints | 2221.27 | 11535 | 1 | 4613 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | agenttemplatekv_exact_reuse | 1689.47 | 11533 | 1 | 14045 | 0 | 0 | exact_code_content_signature | 0.6804 |
| astropy__astropy-13398 | stock_sglang_prefix_only | 2220.24 | 13240 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13398 | kvflow_style_prefix_baseline | 771.47 | 13240 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | kvflow_style_prefix_plus_hints | 1448.02 | 13243 | 1 | 4611 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | agenttemplatekv_exact_reuse | 1859.79 | 13241 | 1 | 14041 | 0 | 0 | exact_code_content_signature | 0.0702 |
| astropy__astropy-13453 | stock_sglang_prefix_only | 2041.85 | 12492 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_style_prefix_baseline | 2032.04 | 12492 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_style_prefix_plus_hints | 2239.29 | 12495 | 1 | 4611 | 0 | 0 |  | 0.7222 |
| astropy__astropy-13453 | agenttemplatekv_exact_reuse | 2244.9 | 12493 | 1 | 14041 | 0 | 0 | exact_code_content_signature | 0.7222 |
| astropy__astropy-13579 | stock_sglang_prefix_only | 2208.1 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_baseline | 2208.73 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_plus_hints | 2226.83 | 11811 | 1 | 4611 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | agenttemplatekv_exact_reuse | 2239.02 | 11809 | 1 | 14041 | 0 | 0 | exact_code_content_signature | 1.0 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
