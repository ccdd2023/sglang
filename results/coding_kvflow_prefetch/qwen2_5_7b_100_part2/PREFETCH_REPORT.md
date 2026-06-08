# Coding KVFlow Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 22
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but KVCOMM reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | prefetch success | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 22 | 4199.8 | 1178.0 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 |
| kvflow_prefix_only | 22 | 4249.3 | 1178.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.3937 |
| kvflow_prefix_plus_codebase_prefetch | 22 | 4210.8 | 1180.8 | 2.9 | 0.0 | 0.00 | 0.00 | 0.3898 |
| kvcomm_lossy_plus_codebase_prefetch | 22 | 4234.3 | 1659.1 | 2.9 | 0.0 | 0.00 | 0.95 | 0.3494 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_part2/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_part2/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_part2/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | prefetch queued | match reason | token F1 |
|---|---|---:|---:|---:|---|---:|
| pallets__flask-5014 | baseline_prefix_cache_only | 3803.42 | 485 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_prefix_only | 3804.84 | 485 | 0 |  | 0.2667 |
| pallets__flask-5014 | kvflow_prefix_plus_codebase_prefetch | 3820.64 | 488 | 0 |  | 0.4286 |
| pallets__flask-5014 | kvcomm_lossy_plus_codebase_prefetch | 3809.66 | 486 | 0 | exact_code_content_signature | 0.717 |
| psf__requests-1142 | baseline_prefix_cache_only | 52.03 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_prefix_only | 52.51 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_prefix_plus_codebase_prefetch | 55.05 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvcomm_lossy_plus_codebase_prefetch | 73.76 | 0 | 0 |  | 1.0 |
| psf__requests-1724 | baseline_prefix_cache_only | 6218.19 | 2699 | 0 |  | 1.0 |
| psf__requests-1724 | kvflow_prefix_only | 6211.76 | 2699 | 0 |  | 0.2807 |
| psf__requests-1724 | kvflow_prefix_plus_codebase_prefetch | 6215.48 | 2702 | 0 |  | 0.2987 |
| psf__requests-1724 | kvcomm_lossy_plus_codebase_prefetch | 6220.69 | 2700 | 0 | exact_code_content_signature | 0.239 |
| psf__requests-1766 | baseline_prefix_cache_only | 6065.18 | 769 | 0 |  | 1.0 |
| psf__requests-1766 | kvflow_prefix_only | 6088.54 | 769 | 0 |  | 0.2876 |
| psf__requests-1766 | kvflow_prefix_plus_codebase_prefetch | 6072.21 | 772 | 0 |  | 0.3137 |
| psf__requests-1766 | kvcomm_lossy_plus_codebase_prefetch | 6076.57 | 770 | 0 | exact_code_content_signature | 0.2987 |
| psf__requests-1921 | baseline_prefix_cache_only | 6096.95 | 745 | 0 |  | 1.0 |
| psf__requests-1921 | kvflow_prefix_only | 6097.23 | 745 | 0 |  | 0.3086 |
| psf__requests-1921 | kvflow_prefix_plus_codebase_prefetch | 6099.33 | 748 | 0 |  | 0.3077 |
| psf__requests-1921 | kvcomm_lossy_plus_codebase_prefetch | 6114.61 | 746 | 0 | exact_code_content_signature | 0.4605 |
| psf__requests-2317 | baseline_prefix_cache_only | 6095.33 | 773 | 0 |  | 1.0 |
| psf__requests-2317 | kvflow_prefix_only | 6096.31 | 773 | 0 |  | 0.4667 |
| psf__requests-2317 | kvflow_prefix_plus_codebase_prefetch | 6104.23 | 776 | 0 |  | 0.2636 |
| psf__requests-2317 | kvcomm_lossy_plus_codebase_prefetch | 6115.09 | 774 | 0 | exact_code_content_signature | 0.4431 |
| psf__requests-2931 | baseline_prefix_cache_only | 6096.13 | 550 | 0 |  | 1.0 |
| psf__requests-2931 | kvflow_prefix_only | 6097.74 | 550 | 0 |  | 0.1818 |
| psf__requests-2931 | kvflow_prefix_plus_codebase_prefetch | 6119.69 | 553 | 0 |  | 0.2819 |
| psf__requests-2931 | kvcomm_lossy_plus_codebase_prefetch | 6095.82 | 551 | 0 | exact_code_content_signature | 0.2535 |
| psf__requests-5414 | baseline_prefix_cache_only | 3783.44 | 910 | 0 |  | 1.0 |
| psf__requests-5414 | kvflow_prefix_only | 3782.38 | 910 | 0 |  | 0.6826 |
| psf__requests-5414 | kvflow_prefix_plus_codebase_prefetch | 3786.72 | 913 | 0 |  | 0.618 |
| psf__requests-5414 | kvcomm_lossy_plus_codebase_prefetch | 3768.37 | 911 | 0 | exact_code_content_signature | 0.5698 |
| psf__requests-6028 | baseline_prefix_cache_only | 2594.72 | 939 | 0 |  | 1.0 |
| psf__requests-6028 | kvflow_prefix_only | 3785.67 | 939 | 0 |  | 0.2881 |
| psf__requests-6028 | kvflow_prefix_plus_codebase_prefetch | 2608.56 | 942 | 0 |  | 1.0 |
| psf__requests-6028 | kvcomm_lossy_plus_codebase_prefetch | 3801.36 | 940 | 0 | exact_code_content_signature | 0.2735 |
| pydata__xarray-2905 | baseline_prefix_cache_only | 3932.47 | 1292 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_prefix_only | 3934.56 | 1292 | 0 |  | 0.0221 |
| pydata__xarray-2905 | kvflow_prefix_plus_codebase_prefetch | 3948.28 | 1295 | 0 |  | 0.2619 |
| pydata__xarray-2905 | kvcomm_lossy_plus_codebase_prefetch | 4657.08 | 1293 | 0 | exact_code_content_signature | 0.3951 |
| pydata__xarray-3095 | baseline_prefix_cache_only | 4279.53 | 1062 | 0 |  | 1.0 |
| pydata__xarray-3095 | kvflow_prefix_only | 3877.44 | 1062 | 0 |  | 0.4405 |
| pydata__xarray-3095 | kvflow_prefix_plus_codebase_prefetch | 4025.85 | 1065 | 0 |  | 0.4024 |
| pydata__xarray-3095 | kvcomm_lossy_plus_codebase_prefetch | 3934.66 | 1063 | 0 | exact_code_content_signature | 0.3311 |
| pydata__xarray-3151 | baseline_prefix_cache_only | 3870.54 | 1418 | 0 |  | 1.0 |
| pydata__xarray-3151 | kvflow_prefix_only | 4076.58 | 1418 | 0 |  | 0.2517 |
| pydata__xarray-3151 | kvflow_prefix_plus_codebase_prefetch | 3838.91 | 1421 | 0 |  | 0.3419 |
| pydata__xarray-3151 | kvcomm_lossy_plus_codebase_prefetch | 3901.61 | 1419 | 0 | exact_code_content_signature | 0.5175 |
| pydata__xarray-3305 | baseline_prefix_cache_only | 3872.2 | 1226 | 0 |  | 1.0 |
| pydata__xarray-3305 | kvflow_prefix_only | 3871.12 | 1226 | 0 |  | 0.1871 |
| pydata__xarray-3305 | kvflow_prefix_plus_codebase_prefetch | 3866.81 | 1229 | 0 |  | 0.2468 |
| pydata__xarray-3305 | kvcomm_lossy_plus_codebase_prefetch | 3872.97 | 1227 | 0 | exact_code_content_signature | 0.2468 |
| pydata__xarray-3677 | baseline_prefix_cache_only | 3857.62 | 954 | 0 |  | 1.0 |
| pydata__xarray-3677 | kvflow_prefix_only | 3837.26 | 954 | 0 |  | 0.3886 |
| pydata__xarray-3677 | kvflow_prefix_plus_codebase_prefetch | 3864.1 | 957 | 0 |  | 0.5957 |
| pydata__xarray-3677 | kvcomm_lossy_plus_codebase_prefetch | 3867.55 | 955 | 0 | exact_code_content_signature | 0.3636 |
| pydata__xarray-3993 | baseline_prefix_cache_only | 3974.02 | 828 | 0 |  | 1.0 |
| pydata__xarray-3993 | kvflow_prefix_only | 3970.17 | 828 | 0 |  | 0.2615 |
| pydata__xarray-3993 | kvflow_prefix_plus_codebase_prefetch | 3964.67 | 831 | 0 |  | 0.189 |
| pydata__xarray-3993 | kvcomm_lossy_plus_codebase_prefetch | 3945.63 | 829 | 0 | exact_code_content_signature | 0.2295 |
| pydata__xarray-4075 | baseline_prefix_cache_only | 3868.26 | 1453 | 0 |  | 1.0 |
| pydata__xarray-4075 | kvflow_prefix_only | 3864.41 | 1453 | 0 |  | 0.2953 |
| pydata__xarray-4075 | kvflow_prefix_plus_codebase_prefetch | 3870.03 | 1456 | 0 |  | 0.3375 |
| pydata__xarray-4075 | kvcomm_lossy_plus_codebase_prefetch | 3881.63 | 1454 | 0 | exact_code_content_signature | 0.293 |
| pydata__xarray-4094 | baseline_prefix_cache_only | 3859.4 | 1140 | 0 |  | 1.0 |
| pydata__xarray-4094 | kvflow_prefix_only | 3818.03 | 1140 | 0 |  | 0.3205 |
| pydata__xarray-4094 | kvflow_prefix_plus_codebase_prefetch | 3823.65 | 1143 | 0 |  | 0.2517 |
| pydata__xarray-4094 | kvcomm_lossy_plus_codebase_prefetch | 3820.92 | 1141 | 0 | exact_code_content_signature | 0.2138 |
| pydata__xarray-4356 | baseline_prefix_cache_only | 3810.5 | 1281 | 0 |  | 1.0 |
| pydata__xarray-4356 | kvflow_prefix_only | 3917.25 | 1281 | 0 |  | 0.2874 |
| pydata__xarray-4356 | kvflow_prefix_plus_codebase_prefetch | 4262.42 | 1284 | 0 |  | 0.2527 |
| pydata__xarray-4356 | kvcomm_lossy_plus_codebase_prefetch | 4127.36 | 1282 | 0 | exact_code_content_signature | 0.2561 |
| pydata__xarray-4629 | baseline_prefix_cache_only | 4286.54 | 1431 | 0 |  | 1.0 |
| pydata__xarray-4629 | kvflow_prefix_only | 4327.32 | 1431 | 0 |  | 0.4528 |
| pydata__xarray-4629 | kvflow_prefix_plus_codebase_prefetch | 4309.81 | 1434 | 0 |  | 0.3373 |
| pydata__xarray-4629 | kvcomm_lossy_plus_codebase_prefetch | 4126.21 | 1432 | 0 | exact_code_content_signature | 0.4311 |
| pydata__xarray-4687 | baseline_prefix_cache_only | 4056.93 | 2046 | 0 |  | 1.0 |
| pydata__xarray-4687 | kvflow_prefix_only | 4047.7 | 2046 | 0 |  | 0.5556 |
| pydata__xarray-4687 | kvflow_prefix_plus_codebase_prefetch | 4052.06 | 2049 | 0 |  | 0.3265 |
| pydata__xarray-4687 | kvcomm_lossy_plus_codebase_prefetch | 3533.06 | 7323 | 0 | exact_code_content_signature | 0.0787 |
| pydata__xarray-4695 | baseline_prefix_cache_only | 3954.8 | 1869 | 0 |  | 1.0 |
| pydata__xarray-4695 | kvflow_prefix_only | 3956.86 | 1869 | 0 |  | 0.5503 |
| pydata__xarray-4695 | kvflow_prefix_plus_codebase_prefetch | 3960.05 | 1872 | 0 |  | 0.4903 |
| pydata__xarray-4695 | kvcomm_lossy_plus_codebase_prefetch | 3440.06 | 7159 | 0 | exact_code_content_signature | 0.0625 |
| pydata__xarray-4966 | baseline_prefix_cache_only | 3968.44 | 2045 | 0 |  | 1.0 |
| pydata__xarray-4966 | kvflow_prefix_only | 3969.04 | 2045 | 0 |  | 0.8857 |
| pydata__xarray-4966 | kvflow_prefix_plus_codebase_prefetch | 3968.95 | 2048 | 0 |  | 0.0288 |
| pydata__xarray-4966 | kvcomm_lossy_plus_codebase_prefetch | 3969.2 | 2046 | 0 | exact_code_content_signature | 0.0131 |

## Interpretation

This benchmark isolates serving-side KVFlow behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and exact-content KVCOMM hits, but host load-back counters remain zero. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
