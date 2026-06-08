# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/swe_verified_10_instances.json`
- Cases: 3
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 3 | 1613.1 | 12846.3 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 3 | 1276.4 | 12846.3 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.2065 |
| kvflow_style_prefix_plus_hints | 3 | 2045.5 | 12849.3 | 2.0 | 0.0 | 1.00 | 4611.3 | 0.0 | 0.0 | 0.00 | 0.3686 |
| agenttemplatekv_exact_reuse | 3 | 2074.1 | 12847.3 | 2.0 | 0.0 | 1.00 | 13987.0 | 0.0 | 0.0 | 1.00 | 0.3686 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/agenttemplatekv_device_prefetch_smoke_3/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/agenttemplatekv_device_prefetch_smoke_3/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/agenttemplatekv_device_prefetch_smoke_3/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 690.64 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 894.31 | 10547 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2116.89 | 10550 | 1 | 4599 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2210.45 | 10548 | 1 | 14017 | 0 | 0 | exact_code_content_signature | 0.0 |
| django__django-10097 | stock_sglang_prefix_only | 1949.3 | 17295 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | kvflow_style_prefix_baseline | 1439.25 | 17295 | 0 | 0 | 0 | 0 |  | 0.5067 |
| django__django-10097 | kvflow_style_prefix_plus_hints | 1799.06 | 17298 | 1 | 4468 | 0 | 0 |  | 0.8132 |
| django__django-10097 | agenttemplatekv_exact_reuse | 1805.72 | 17296 | 1 | 13467 | 0 | 0 | exact_code_content_signature | 0.8132 |
| matplotlib__matplotlib-13989 | stock_sglang_prefix_only | 2199.49 | 10697 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_baseline | 1495.54 | 10697 | 0 | 0 | 0 | 0 |  | 0.1127 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_plus_hints | 2220.46 | 10700 | 1 | 4767 | 0 | 0 |  | 0.2927 |
| matplotlib__matplotlib-13989 | agenttemplatekv_exact_reuse | 2206.2 | 10698 | 1 | 14477 | 0 | 0 | exact_code_content_signature | 0.2927 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
