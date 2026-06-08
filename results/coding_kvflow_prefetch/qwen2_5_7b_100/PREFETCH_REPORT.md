# Coding KVFlow Prefetch Report — 100-Case Full Results

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 100
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but KVCOMM reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | p50 ms | p90 ms | avg cached tokens | avg hints | exact-content hit rate | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 100 | 3910.6 | 3872.3 | 4139.4 | 1581.8 | 0.0 | 0.00 | 1.0000 |
| kvflow_prefix_only | 100 | 3941.2 | 3878.2 | 4235.0 | 1581.8 | 0.0 | 0.00 | 0.4916 |
| kvflow_prefix_plus_codebase_prefetch | 100 | 3926.5 | 3870.2 | 4275.2 | 1584.8 | 3.0 | 0.00 | 0.4295 |
| kvcomm_lossy_plus_codebase_prefetch | 100 | 3837.6 | 3831.7 | 4209.0 | 2592.5 | 3.0 | 0.99 | 0.3461 |

## Key Takeaways

1. **Exact-content hit rate**: 0.99 (99/100 cases hit exact_code_content_signature)
2. **Cached tokens boost**: 2592.5 vs baseline 1581.8 (1.64×)
3. **Latency**: avg 3837.6ms vs baseline 3910.6ms (1.019×)
4. **P50 latency**: 3831.7ms vs baseline 3872.3ms
5. **P90 latency**: 4209.0ms vs baseline 4139.4ms

## Per-Mode Comparison (Latency)

| mode | avg ms | p50 ms | p90 ms |
|---|---:|---:|---:|
| baseline_prefix_cache_only | 3910.6 | 3872.3 | 4139.4 |
| kvflow_prefix_only | 3941.2 | 3878.2 | 4235.0 |
| kvflow_prefix_plus_codebase_prefetch | 3926.5 | 3870.2 | 4275.2 |
| kvcomm_lossy_plus_codebase_prefetch | 3837.6 | 3831.7 | 4209.0 |

## Per-Mode Comparison (Cache & Reuse)

| mode | avg cached | avg hints | exact hit rate | avg F1 |
|---|---:|---:|---:|---:|
| baseline_prefix_cache_only | 1581.8 | 0.0 | 0.00 | 1.0000 |
| kvflow_prefix_only | 1581.8 | 0.0 | 0.00 | 0.4916 |
| kvflow_prefix_plus_codebase_prefetch | 1584.8 | 3.0 | 0.00 | 0.4295 |
| kvcomm_lossy_plus_codebase_prefetch | 2592.5 | 3.0 | 0.99 | 0.3461 |
