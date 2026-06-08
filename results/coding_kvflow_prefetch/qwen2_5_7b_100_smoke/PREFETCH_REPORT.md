# Coding KVFlow Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 5
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but KVCOMM reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | prefetch success | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 5 | 3852.0 | 2252.8 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 |
| kvflow_prefix_only | 5 | 3846.5 | 2252.8 | 0.0 | 0.0 | 0.00 | 0.00 | 0.5332 |
| kvflow_prefix_plus_codebase_prefetch | 5 | 3855.4 | 2255.8 | 3.0 | 0.0 | 0.00 | 0.00 | 0.4597 |
| kvcomm_lossy_plus_codebase_prefetch | 5 | 3764.9 | 3178.4 | 3.0 | 0.0 | 0.00 | 1.00 | 0.3664 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_smoke/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_smoke/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_smoke/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | prefetch queued | match reason | token F1 |
|---|---|---:|---:|---:|---|---:|
| astropy__astropy-12907 | baseline_prefix_cache_only | 3790.04 | 1101 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_prefix_only | 3785.03 | 1101 | 0 |  | 0.1835 |
| astropy__astropy-12907 | kvflow_prefix_plus_codebase_prefetch | 3784.82 | 1104 | 0 |  | 0.1509 |
| astropy__astropy-12907 | kvcomm_lossy_plus_codebase_prefetch | 3375.77 | 5725 | 0 | exact_code_content_signature | 0.0169 |
| astropy__astropy-13033 | baseline_prefix_cache_only | 3809.41 | 1275 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_prefix_only | 3800.05 | 1275 | 0 |  | 0.9459 |
| astropy__astropy-13033 | kvflow_prefix_plus_codebase_prefetch | 3801.37 | 1278 | 0 |  | 0.6993 |
| astropy__astropy-13033 | kvcomm_lossy_plus_codebase_prefetch | 3803.48 | 1276 | 0 | exact_code_content_signature | 0.5125 |
| astropy__astropy-13236 | baseline_prefix_cache_only | 3842.4 | 2072 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_prefix_only | 3839.87 | 2072 | 0 |  | 0.3947 |
| astropy__astropy-13236 | kvflow_prefix_plus_codebase_prefetch | 3844.09 | 2075 | 0 |  | 0.3165 |
| astropy__astropy-13236 | kvcomm_lossy_plus_codebase_prefetch | 3845.19 | 2073 | 0 | exact_code_content_signature | 0.3497 |
| astropy__astropy-13398 | baseline_prefix_cache_only | 3918.32 | 3782 | 0 |  | 1.0 |
| astropy__astropy-13398 | kvflow_prefix_only | 3918.8 | 3782 | 0 |  | 0.3949 |
| astropy__astropy-13398 | kvflow_prefix_plus_codebase_prefetch | 3937.42 | 3785 | 0 |  | 0.4085 |
| astropy__astropy-13398 | kvcomm_lossy_plus_codebase_prefetch | 3925.39 | 3783 | 0 | exact_code_content_signature | 0.3816 |
| astropy__astropy-13453 | baseline_prefix_cache_only | 3899.9 | 3034 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_prefix_only | 3888.91 | 3034 | 0 |  | 0.7471 |
| astropy__astropy-13453 | kvflow_prefix_plus_codebase_prefetch | 3909.4 | 3037 | 0 |  | 0.7232 |
| astropy__astropy-13453 | kvcomm_lossy_plus_codebase_prefetch | 3874.52 | 3035 | 0 | exact_code_content_signature | 0.5714 |

## Interpretation

This benchmark isolates serving-side KVFlow behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and exact-content KVCOMM hits, but host load-back counters remain zero. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
