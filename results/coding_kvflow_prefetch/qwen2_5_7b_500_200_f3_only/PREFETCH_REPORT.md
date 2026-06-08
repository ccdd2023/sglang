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
| stock_sglang_prefix_only | 6 | 1807.7 | 11724.3 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 6 | 1602.1 | 11724.3 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.6667 |
| kvflow_style_prefix_plus_hints | 6 | 1962.9 | 11727.3 | 2.0 | 0.0 | 1.00 | 4608.3 | 0.0 | 0.0 | 0.00 | 0.6204 |
| agenttemplatekv_exact_reuse | 6 | 1950.6 | 11725.3 | 2.0 | 0.0 | 1.00 | 14035.7 | 0.0 | 0.0 | 1.00 | 0.5788 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_200_postfix/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 691.9 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 894.62 | 10547 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2113.82 | 10550 | 1 | 4599 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2203.05 | 10548 | 1 | 14017 | 0 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-13033 | stock_sglang_prefix_only | 1478.54 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_baseline | 1484.04 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_plus_hints | 1502.35 | 10730 | 1 | 4605 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | agenttemplatekv_exact_reuse | 1486.13 | 10728 | 1 | 14029 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13236 | stock_sglang_prefix_only | 2208.52 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_baseline | 2203.62 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_plus_hints | 2233.33 | 11535 | 1 | 4613 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | agenttemplatekv_exact_reuse | 1689.64 | 11533 | 1 | 14045 | 0 | 0 | exact_code_content_signature | 0.6804 |
| astropy__astropy-13398 | stock_sglang_prefix_only | 2224.7 | 13240 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13398 | kvflow_style_prefix_baseline | 771.96 | 13240 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | kvflow_style_prefix_plus_hints | 1458.13 | 13243 | 1 | 4611 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | agenttemplatekv_exact_reuse | 1843.6 | 13241 | 1 | 14041 | 0 | 0 | exact_code_content_signature | 0.0702 |
| astropy__astropy-13453 | stock_sglang_prefix_only | 2034.07 | 12492 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_style_prefix_baseline | 2036.1 | 12492 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_style_prefix_plus_hints | 2241.56 | 12495 | 1 | 4611 | 0 | 0 |  | 0.7222 |
| astropy__astropy-13453 | agenttemplatekv_exact_reuse | 2246.4 | 12493 | 1 | 14041 | 0 | 0 | exact_code_content_signature | 0.7222 |
| astropy__astropy-13579 | stock_sglang_prefix_only | 2208.56 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_baseline | 2221.99 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_plus_hints | 2228.46 | 11811 | 1 | 4611 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | agenttemplatekv_exact_reuse | 2235.04 | 11809 | 1 | 14041 | 0 | 0 | exact_code_content_signature | 1.0 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
