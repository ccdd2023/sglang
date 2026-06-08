# Coding KVFlow Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_10_instances.json`
- Cases: 10
- HiCache storage backend: `disabled`
- Safety rule: codebase prefetch may predict future code blocks, but KVCOMM reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | prefetch success | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 10 | 1421.2 | 1773.7 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 |
| kvflow_prefix_only | 10 | 1388.0 | 1773.7 | 0.0 | 0.0 | 0.00 | 0.00 | 0.4853 |
| kvflow_prefix_plus_codebase_prefetch | 10 | 1297.1 | 1776.7 | 1.0 | 0.0 | 0.00 | 0.00 | 0.5149 |
| kvcomm_lossy_plus_codebase_prefetch | 10 | 1280.3 | 2832.1 | 1.0 | 0.0 | 0.00 | 1.00 | 0.3439 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_10/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_10/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_10/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | prefetch queued | match reason | token F1 |
|---|---|---:|---:|---:|---|---:|
| astropy__astropy-12907 | baseline_prefix_cache_only | 1274.81 | 1085 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_prefix_only | 1283.83 | 1085 | 0 |  | 0.0351 |
| astropy__astropy-12907 | kvflow_prefix_plus_codebase_prefetch | 1278.13 | 1088 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvcomm_lossy_plus_codebase_prefetch | 1080.84 | 3522 | 0 | exact_code_content_signature | 0.0 |
| django__django-10097 | baseline_prefix_cache_only | 1437.96 | 8242 | 0 |  | 1.0 |
| django__django-10097 | kvflow_prefix_only | 1425.87 | 8242 | 0 |  | 0.0513 |
| django__django-10097 | kvflow_prefix_plus_codebase_prefetch | 634.62 | 8245 | 0 |  | 0.0909 |
| django__django-10097 | kvcomm_lossy_plus_codebase_prefetch | 1128.21 | 11212 | 0 | exact_code_content_signature | 0.0 |
| matplotlib__matplotlib-13989 | baseline_prefix_cache_only | 1298.21 | 938 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_prefix_only | 1499.37 | 938 | 0 |  | 0.2857 |
| matplotlib__matplotlib-13989 | kvflow_prefix_plus_codebase_prefetch | 1531.4 | 941 | 0 |  | 0.3415 |
| matplotlib__matplotlib-13989 | kvcomm_lossy_plus_codebase_prefetch | 1079.72 | 3517 | 0 | exact_code_content_signature | 0.0308 |
| mwaskom__seaborn-3069 | baseline_prefix_cache_only | 1507.76 | 933 | 0 |  | 1.0 |
| mwaskom__seaborn-3069 | kvflow_prefix_only | 1303.51 | 933 | 0 |  | 0.5714 |
| mwaskom__seaborn-3069 | kvflow_prefix_plus_codebase_prefetch | 1310.8 | 936 | 0 |  | 0.9118 |
| mwaskom__seaborn-3069 | kvcomm_lossy_plus_codebase_prefetch | 1423.43 | 934 | 0 | exact_code_content_signature | 0.7397 |
| pallets__flask-5014 | baseline_prefix_cache_only | 1413.8 | 471 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_prefix_only | 1614.4 | 471 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_prefix_plus_codebase_prefetch | 1291.12 | 474 | 0 |  | 1.0 |
| pallets__flask-5014 | kvcomm_lossy_plus_codebase_prefetch | 1076.25 | 3063 | 0 | exact_code_content_signature | 0.0 |
| psf__requests-1142 | baseline_prefix_cache_only | 1427.7 | 538 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_prefix_only | 1293.61 | 538 | 0 |  | 0.5581 |
| psf__requests-1142 | kvflow_prefix_plus_codebase_prefetch | 1392.26 | 541 | 0 |  | 0.5581 |
| psf__requests-1142 | kvcomm_lossy_plus_codebase_prefetch | 1401.82 | 539 | 0 | exact_code_content_signature | 0.4368 |
| pydata__xarray-2905 | baseline_prefix_cache_only | 1447.06 | 1276 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_prefix_only | 1312.87 | 1276 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_prefix_plus_codebase_prefetch | 1413.63 | 1279 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvcomm_lossy_plus_codebase_prefetch | 1433.73 | 1277 | 0 | exact_code_content_signature | 0.8158 |
| pylint-dev__pylint-4551 | baseline_prefix_cache_only | 1384.58 | 1671 | 0 |  | 1.0 |
| pylint-dev__pylint-4551 | kvflow_prefix_only | 1240.12 | 1671 | 0 |  | 0.5794 |
| pylint-dev__pylint-4551 | kvflow_prefix_plus_codebase_prefetch | 1240.57 | 1674 | 0 |  | 0.5981 |
| pylint-dev__pylint-4551 | kvcomm_lossy_plus_codebase_prefetch | 1346.84 | 1672 | 0 | exact_code_content_signature | 0.5981 |
| pytest-dev__pytest-10051 | baseline_prefix_cache_only | 1452.02 | 1124 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_prefix_only | 1307.52 | 1124 | 0 |  | 0.5 |
| pytest-dev__pytest-10051 | kvflow_prefix_plus_codebase_prefetch | 1416.14 | 1127 | 0 |  | 0.4615 |
| pytest-dev__pytest-10051 | kvcomm_lossy_plus_codebase_prefetch | 1421.9 | 1125 | 0 | exact_code_content_signature | 0.6176 |
| scikit-learn__scikit-learn-10297 | baseline_prefix_cache_only | 1568.05 | 1459 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10297 | kvflow_prefix_only | 1598.9 | 1459 | 0 |  | 0.2716 |
| scikit-learn__scikit-learn-10297 | kvflow_prefix_plus_codebase_prefetch | 1462.3 | 1462 | 0 |  | 0.1875 |
| scikit-learn__scikit-learn-10297 | kvcomm_lossy_plus_codebase_prefetch | 1409.8 | 1460 | 0 | exact_code_content_signature | 0.2 |

## Interpretation

This benchmark isolates serving-side KVFlow behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and exact-content KVCOMM hits, but host load-back counters remain zero. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
