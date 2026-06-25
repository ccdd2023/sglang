# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/swe_verified_10_instances.json`
- Cases: 10
- Git commit: `5fb934751`
- Command: `benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --reuse-server --port 30000 --max-cases 10 --start-index 0 --files-per-case 3 --disable-hierarchical-cache --emit-ttft --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_ttft_10case_20260616`
- Flush cache per case: `False`
- Concurrent clients: `1`
- Baseline profile: `agenttemplatekv`
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Server extra args: ``
- Resolved server extra args: ``
- LMCache config: ``
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | p50 | p90 | p99 | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 10 | 1430.5 | 1477.0 | 2257.8 | 2260.8 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 10 | 1535.7 | 1958.5 | 2282.9 | 2306.9 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_plus_hints | 10 | 1596.8 | 1694.9 | 2256.7 | 2260.2 | 0.0 | 2.7 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| agenttemplatekv_exact_reuse | 10 | 1368.8 | 1397.8 | 2251.8 | 2260.9 | 0.0 | 2.7 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.90 | 1.0000 |

## Figures

Figure generation skipped because a clean `matplotlib` import is unavailable in this environment.

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 2257.48 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 2255.01 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2256.22 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2250.68 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-10097 | stock_sglang_prefix_only | 2084.14 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | kvflow_style_prefix_baseline | 2309.55 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | kvflow_style_prefix_plus_hints | 1925.36 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | agenttemplatekv_exact_reuse | 1951.7 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-13989 | stock_sglang_prefix_only | 2261.13 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_baseline | 1667.45 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_plus_hints | 1997.11 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | agenttemplatekv_exact_reuse | 1751.84 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| mwaskom__seaborn-3069 | stock_sglang_prefix_only | 822.74 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3069 | kvflow_style_prefix_baseline | 812.88 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3069 | kvflow_style_prefix_plus_hints | 2260.54 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3069 | agenttemplatekv_exact_reuse | 808.92 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| pallets__flask-5014 | stock_sglang_prefix_only | 2250.71 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_style_prefix_baseline | 2249.53 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_style_prefix_plus_hints | 2250.44 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pallets__flask-5014 | agenttemplatekv_exact_reuse | 2261.92 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| psf__requests-1142 | stock_sglang_prefix_only | 61.49 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_style_prefix_baseline | 92.23 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_style_prefix_plus_hints | 61.35 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | agenttemplatekv_exact_reuse | 80.19 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | stock_sglang_prefix_only | 819.99 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_style_prefix_baseline | 819.01 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_style_prefix_plus_hints | 1197.88 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | agenttemplatekv_exact_reuse | 1195.98 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| pylint-dev__pylint-4551 | stock_sglang_prefix_only | 708.02 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4551 | kvflow_style_prefix_baseline | 2279.89 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4551 | kvflow_style_prefix_plus_hints | 1464.43 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4551 | agenttemplatekv_exact_reuse | 1457.25 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-10051 | stock_sglang_prefix_only | 869.9 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_style_prefix_baseline | 599.25 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_style_prefix_plus_hints | 1210.7 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | agenttemplatekv_exact_reuse | 591.52 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| scikit-learn__scikit-learn-10297 | stock_sglang_prefix_only | 2169.08 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10297 | kvflow_style_prefix_baseline | 2271.76 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10297 | kvflow_style_prefix_plus_hints | 1343.57 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10297 | agenttemplatekv_exact_reuse | 1338.33 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
