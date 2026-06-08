# Coding KVFlow Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_10_instances.json`
- Cases: 3
- Safety rule: codebase prefetch may predict future code blocks, but KVCOMM reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg prefetch queued | prefetch success | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 3 | 1344.7 | 3421.7 | 0.0 | 0.00 | 0.00 | 1.0000 |
| kvflow_prefix_only | 3 | 1406.6 | 3421.7 | 0.0 | 0.00 | 0.00 | 0.1240 |
| kvflow_prefix_plus_codebase_prefetch | 3 | 1151.9 | 3424.7 | 0.0 | 0.00 | 0.00 | 0.1441 |
| kvcomm_lossy_plus_codebase_prefetch | 3 | 1100.1 | 6083.7 | 0.0 | 0.00 | 1.00 | 0.0103 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/smoke_3/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/smoke_3/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/smoke_3/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | prefetch queued | match reason | token F1 |
|---|---|---:|---:|---:|---|---:|
| astropy__astropy-12907 | baseline_prefix_cache_only | 1299.81 | 1085 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_prefix_only | 1286.37 | 1085 | 0 |  | 0.0351 |
| astropy__astropy-12907 | kvflow_prefix_plus_codebase_prefetch | 1282.47 | 1088 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvcomm_lossy_plus_codebase_prefetch | 1086.15 | 3522 | 0 | exact_code_content_signature | 0.0 |
| django__django-10097 | baseline_prefix_cache_only | 1433.46 | 8242 | 0 |  | 1.0 |
| django__django-10097 | kvflow_prefix_only | 1428.29 | 8242 | 0 |  | 0.0513 |
| django__django-10097 | kvflow_prefix_plus_codebase_prefetch | 635.19 | 8245 | 0 |  | 0.0909 |
| django__django-10097 | kvcomm_lossy_plus_codebase_prefetch | 1137.56 | 11212 | 0 | exact_code_content_signature | 0.0 |
| matplotlib__matplotlib-13989 | baseline_prefix_cache_only | 1300.95 | 938 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_prefix_only | 1505.28 | 938 | 0 |  | 0.2857 |
| matplotlib__matplotlib-13989 | kvflow_prefix_plus_codebase_prefetch | 1537.99 | 941 | 0 |  | 0.3415 |
| matplotlib__matplotlib-13989 | kvcomm_lossy_plus_codebase_prefetch | 1076.57 | 3517 | 0 | exact_code_content_signature | 0.0308 |

## Interpretation

This benchmark isolates serving-side KVFlow behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.
