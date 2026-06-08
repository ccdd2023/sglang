# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/swe_verified_500_instances.json`
- Cases: 3
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 3 | 1458.8 | 10935.3 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 3 | 1525.8 | 10935.3 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.6667 |
| kvflow_style_prefix_plus_hints | 3 | 1943.6 | 10938.3 | 2.0 | 0.0 | 1.00 | 4605.7 | 0.0 | 0.0 | 0.00 | 0.6667 |
| agenttemplatekv_exact_reuse | 3 | 1803.7 | 10936.3 | 2.0 | 0.0 | 1.00 | 14030.3 | 0.0 | 0.0 | 1.00 | 0.5601 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_smoke3/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_smoke3/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_500_smoke3/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 691.15 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 893.33 | 10547 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2113.13 | 10550 | 1 | 4599 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2229.14 | 10548 | 1 | 14017 | 0 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-13033 | stock_sglang_prefix_only | 1477.28 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_baseline | 1477.85 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_plus_hints | 1494.7 | 10730 | 1 | 4605 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | agenttemplatekv_exact_reuse | 1508.09 | 10728 | 1 | 14029 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13236 | stock_sglang_prefix_only | 2207.95 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_baseline | 2206.18 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_plus_hints | 2222.92 | 11535 | 1 | 4613 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | agenttemplatekv_exact_reuse | 1673.98 | 11533 | 1 | 14045 | 0 | 0 | exact_code_content_signature | 0.6804 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
