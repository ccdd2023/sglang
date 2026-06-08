# Coding KVFlow Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 31
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but KVCOMM reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | prefetch success | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 31 | 3792.6 | 1760.5 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 |
| kvflow_prefix_only | 31 | 3820.7 | 1760.5 | 0.0 | 0.0 | 0.00 | 0.00 | 0.4940 |
| kvflow_prefix_plus_codebase_prefetch | 31 | 3800.6 | 1763.5 | 3.0 | 0.0 | 0.00 | 0.00 | 0.4177 |
| kvcomm_lossy_plus_codebase_prefetch | 31 | 3639.2 | 3452.0 | 3.0 | 0.0 | 0.00 | 1.00 | 0.2830 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_part3/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_part3/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_part3/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | prefetch queued | match reason | token F1 |
|---|---|---:|---:|---:|---|---:|
| pydata__xarray-6461 | baseline_prefix_cache_only | 3961.49 | 768 | 0 |  | 1.0 |
| pydata__xarray-6461 | kvflow_prefix_only | 3977.76 | 768 | 0 |  | 0.0972 |
| pydata__xarray-6461 | kvflow_prefix_plus_codebase_prefetch | 3965.05 | 771 | 0 |  | 0.3415 |
| pydata__xarray-6461 | kvcomm_lossy_plus_codebase_prefetch | 2900.08 | 11446 | 0 | exact_code_content_signature | 0.0672 |
| pydata__xarray-6599 | baseline_prefix_cache_only | 4050.87 | 2630 | 0 |  | 1.0 |
| pydata__xarray-6599 | kvflow_prefix_only | 4069.12 | 2630 | 0 |  | 0.4552 |
| pydata__xarray-6599 | kvflow_prefix_plus_codebase_prefetch | 4054.58 | 2633 | 0 |  | 0.3974 |
| pydata__xarray-6599 | kvcomm_lossy_plus_codebase_prefetch | 4056.67 | 2631 | 0 | exact_code_content_signature | 0.3636 |
| pylint-dev__pylint-4551 | baseline_prefix_cache_only | 4112.72 | 1684 | 0 |  | 1.0 |
| pylint-dev__pylint-4551 | kvflow_prefix_only | 4114.93 | 1684 | 0 |  | 0.4561 |
| pylint-dev__pylint-4551 | kvflow_prefix_plus_codebase_prefetch | 4117.73 | 1687 | 0 |  | 0.3741 |
| pylint-dev__pylint-4551 | kvcomm_lossy_plus_codebase_prefetch | 3558.57 | 7667 | 0 | exact_code_content_signature | 0.0148 |
| pylint-dev__pylint-4604 | baseline_prefix_cache_only | 4102.01 | 1287 | 0 |  | 1.0 |
| pylint-dev__pylint-4604 | kvflow_prefix_only | 4086.83 | 1287 | 0 |  | 0.4398 |
| pylint-dev__pylint-4604 | kvflow_prefix_plus_codebase_prefetch | 4090.91 | 1290 | 0 |  | 0.4348 |
| pylint-dev__pylint-4604 | kvcomm_lossy_plus_codebase_prefetch | 3521.6 | 7270 | 0 | exact_code_content_signature | 0.0268 |
| pylint-dev__pylint-4661 | baseline_prefix_cache_only | 4080.81 | 721 | 0 |  | 1.0 |
| pylint-dev__pylint-4661 | kvflow_prefix_only | 4064.47 | 721 | 0 |  | 0.0638 |
| pylint-dev__pylint-4661 | kvflow_prefix_plus_codebase_prefetch | 4075.71 | 724 | 0 |  | 0.0323 |
| pylint-dev__pylint-4661 | kvcomm_lossy_plus_codebase_prefetch | 4065.15 | 722 | 0 | exact_code_content_signature | 0.029 |
| pylint-dev__pylint-4970 | baseline_prefix_cache_only | 4078.5 | 656 | 0 |  | 1.0 |
| pylint-dev__pylint-4970 | kvflow_prefix_only | 4075.37 | 656 | 0 |  | 0.5089 |
| pylint-dev__pylint-4970 | kvflow_prefix_plus_codebase_prefetch | 4064.23 | 659 | 0 |  | 0.514 |
| pylint-dev__pylint-4970 | kvcomm_lossy_plus_codebase_prefetch | 4064.47 | 657 | 0 | exact_code_content_signature | 0.3647 |
| pylint-dev__pylint-6386 | baseline_prefix_cache_only | 3846.81 | 729 | 0 |  | 1.0 |
| pylint-dev__pylint-6386 | kvflow_prefix_only | 3834.76 | 729 | 0 |  | 0.8448 |
| pylint-dev__pylint-6386 | kvflow_prefix_plus_codebase_prefetch | 3840.5 | 732 | 0 |  | 1.0 |
| pylint-dev__pylint-6386 | kvcomm_lossy_plus_codebase_prefetch | 3833.29 | 730 | 0 | exact_code_content_signature | 0.378 |
| pylint-dev__pylint-6528 | baseline_prefix_cache_only | 3925.04 | 2370 | 0 |  | 1.0 |
| pylint-dev__pylint-6528 | kvflow_prefix_only | 3911.92 | 2370 | 0 |  | 0.3333 |
| pylint-dev__pylint-6528 | kvflow_prefix_plus_codebase_prefetch | 3926.8 | 2373 | 0 |  | 0.3016 |
| pylint-dev__pylint-6528 | kvcomm_lossy_plus_codebase_prefetch | 3404.28 | 7373 | 0 | exact_code_content_signature | 0.0 |
| pylint-dev__pylint-6903 | baseline_prefix_cache_only | 3884.16 | 1809 | 0 |  | 1.0 |
| pylint-dev__pylint-6903 | kvflow_prefix_only | 3893.39 | 1809 | 0 |  | 0.7974 |
| pylint-dev__pylint-6903 | kvflow_prefix_plus_codebase_prefetch | 3889.14 | 1812 | 0 |  | 0.5256 |
| pylint-dev__pylint-6903 | kvcomm_lossy_plus_codebase_prefetch | 3385.97 | 6812 | 0 | exact_code_content_signature | 0.0157 |
| pylint-dev__pylint-7080 | baseline_prefix_cache_only | 4202.59 | 8318 | 0 |  | 1.0 |
| pylint-dev__pylint-7080 | kvflow_prefix_only | 4197.59 | 8318 | 0 |  | 0.3218 |
| pylint-dev__pylint-7080 | kvflow_prefix_plus_codebase_prefetch | 4189.52 | 8321 | 0 |  | 0.3704 |
| pylint-dev__pylint-7080 | kvcomm_lossy_plus_codebase_prefetch | 4168.16 | 8319 | 0 | exact_code_content_signature | 0.4343 |
| pylint-dev__pylint-7277 | baseline_prefix_cache_only | 3841.46 | 848 | 0 |  | 1.0 |
| pylint-dev__pylint-7277 | kvflow_prefix_only | 3852.3 | 848 | 0 |  | 0.6364 |
| pylint-dev__pylint-7277 | kvflow_prefix_plus_codebase_prefetch | 3843.61 | 851 | 0 |  | 0.4211 |
| pylint-dev__pylint-7277 | kvcomm_lossy_plus_codebase_prefetch | 3824.84 | 849 | 0 | exact_code_content_signature | 0.4476 |
| pylint-dev__pylint-8898 | baseline_prefix_cache_only | 3851.07 | 1925 | 0 |  | 1.0 |
| pylint-dev__pylint-8898 | kvflow_prefix_only | 3847.14 | 1925 | 0 |  | 0.5889 |
| pylint-dev__pylint-8898 | kvflow_prefix_plus_codebase_prefetch | 3849.82 | 1928 | 0 |  | 0.7403 |
| pylint-dev__pylint-8898 | kvcomm_lossy_plus_codebase_prefetch | 2840.78 | 11937 | 0 | exact_code_content_signature | 0.1224 |
| pytest-dev__pytest-10051 | baseline_prefix_cache_only | 3806.16 | 1137 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_prefix_only | 3823.03 | 1137 | 0 |  | 0.9752 |
| pytest-dev__pytest-10051 | kvflow_prefix_plus_codebase_prefetch | 3809.27 | 1140 | 0 |  | 0.4444 |
| pytest-dev__pytest-10051 | kvcomm_lossy_plus_codebase_prefetch | 3791.66 | 1138 | 0 | exact_code_content_signature | 0.0752 |
| pytest-dev__pytest-10081 | baseline_prefix_cache_only | 3835.38 | 1741 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | kvflow_prefix_only | 3838.85 | 1741 | 0 |  | 0.6788 |
| pytest-dev__pytest-10081 | kvflow_prefix_plus_codebase_prefetch | 3839.28 | 1744 | 0 |  | 0.5325 |
| pytest-dev__pytest-10081 | kvcomm_lossy_plus_codebase_prefetch | 3823.53 | 1742 | 0 | exact_code_content_signature | 0.5977 |
| pytest-dev__pytest-10356 | baseline_prefix_cache_only | 3838.94 | 1767 | 0 |  | 1.0 |
| pytest-dev__pytest-10356 | kvflow_prefix_only | 3850.75 | 1767 | 0 |  | 0.0 |
| pytest-dev__pytest-10356 | kvflow_prefix_plus_codebase_prefetch | 3841.0 | 1770 | 0 |  | 0.5311 |
| pytest-dev__pytest-10356 | kvcomm_lossy_plus_codebase_prefetch | 3844.7 | 1768 | 0 | exact_code_content_signature | 0.4757 |
| pytest-dev__pytest-5262 | baseline_prefix_cache_only | 3824.45 | 1643 | 0 |  | 1.0 |
| pytest-dev__pytest-5262 | kvflow_prefix_only | 3842.82 | 1643 | 0 |  | 0.2676 |
| pytest-dev__pytest-5262 | kvflow_prefix_plus_codebase_prefetch | 3830.8 | 1646 | 0 |  | 0.3429 |
| pytest-dev__pytest-5262 | kvcomm_lossy_plus_codebase_prefetch | 3833.39 | 1644 | 0 | exact_code_content_signature | 0.3288 |
| pytest-dev__pytest-5631 | baseline_prefix_cache_only | 3819.87 | 1326 | 0 |  | 1.0 |
| pytest-dev__pytest-5631 | kvflow_prefix_only | 3824.54 | 1326 | 0 |  | 0.4268 |
| pytest-dev__pytest-5631 | kvflow_prefix_plus_codebase_prefetch | 3811.77 | 1329 | 0 |  | 0.3567 |
| pytest-dev__pytest-5631 | kvcomm_lossy_plus_codebase_prefetch | 3796.52 | 1327 | 0 | exact_code_content_signature | 0.3977 |
| pytest-dev__pytest-5787 | baseline_prefix_cache_only | 3903.65 | 2966 | 0 |  | 1.0 |
| pytest-dev__pytest-5787 | kvflow_prefix_only | 3888.75 | 2966 | 0 |  | 0.0245 |
| pytest-dev__pytest-5787 | kvflow_prefix_plus_codebase_prefetch | 1724.3 | 2969 | 0 |  | 0.0 |
| pytest-dev__pytest-5787 | kvcomm_lossy_plus_codebase_prefetch | 3437.51 | 7615 | 0 | exact_code_content_signature | 0.0153 |
| pytest-dev__pytest-5809 | baseline_prefix_cache_only | 3790.7 | 763 | 0 |  | 1.0 |
| pytest-dev__pytest-5809 | kvflow_prefix_only | 3795.47 | 763 | 0 |  | 0.5854 |
| pytest-dev__pytest-5809 | kvflow_prefix_plus_codebase_prefetch | 3789.12 | 766 | 0 |  | 0.3929 |
| pytest-dev__pytest-5809 | kvcomm_lossy_plus_codebase_prefetch | 3772.25 | 764 | 0 | exact_code_content_signature | 0.3025 |
| pytest-dev__pytest-5840 | baseline_prefix_cache_only | 3830.84 | 1453 | 0 |  | 1.0 |
| pytest-dev__pytest-5840 | kvflow_prefix_only | 3831.09 | 1453 | 0 |  | 0.4557 |
| pytest-dev__pytest-5840 | kvflow_prefix_plus_codebase_prefetch | 3834.51 | 1456 | 0 |  | 0.4634 |
| pytest-dev__pytest-5840 | kvcomm_lossy_plus_codebase_prefetch | 3801.3 | 1454 | 0 | exact_code_content_signature | 0.4568 |
| pytest-dev__pytest-6197 | baseline_prefix_cache_only | 1666.52 | 1723 | 0 |  | 1.0 |
| pytest-dev__pytest-6197 | kvflow_prefix_only | 1665.84 | 1723 | 0 |  | 1.0 |
| pytest-dev__pytest-6197 | kvflow_prefix_plus_codebase_prefetch | 3841.28 | 1726 | 0 |  | 0.0 |
| pytest-dev__pytest-6197 | kvcomm_lossy_plus_codebase_prefetch | 1657.56 | 1724 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-6202 | baseline_prefix_cache_only | 3823.39 | 1450 | 0 |  | 1.0 |
| pytest-dev__pytest-6202 | kvflow_prefix_only | 3839.54 | 1450 | 0 |  | 0.1769 |
| pytest-dev__pytest-6202 | kvflow_prefix_plus_codebase_prefetch | 3824.25 | 1453 | 0 |  | 0.1928 |
| pytest-dev__pytest-6202 | kvcomm_lossy_plus_codebase_prefetch | 3823.31 | 1451 | 0 | exact_code_content_signature | 0.1893 |
| pytest-dev__pytest-7205 | baseline_prefix_cache_only | 3871.98 | 2209 | 0 |  | 1.0 |
| pytest-dev__pytest-7205 | kvflow_prefix_only | 3862.5 | 2209 | 0 |  | 0.0 |
| pytest-dev__pytest-7205 | kvflow_prefix_plus_codebase_prefetch | 3877.64 | 2212 | 0 |  | 0.0571 |
| pytest-dev__pytest-7205 | kvcomm_lossy_plus_codebase_prefetch | 3861.43 | 2210 | 0 | exact_code_content_signature | 0.1143 |
| pytest-dev__pytest-7236 | baseline_prefix_cache_only | 3827.87 | 1343 | 0 |  | 1.0 |
| pytest-dev__pytest-7236 | kvflow_prefix_only | 3833.47 | 1343 | 0 |  | 0.8701 |
| pytest-dev__pytest-7236 | kvflow_prefix_plus_codebase_prefetch | 3822.83 | 1346 | 0 |  | 0.454 |
| pytest-dev__pytest-7236 | kvcomm_lossy_plus_codebase_prefetch | 3806.03 | 1344 | 0 | exact_code_content_signature | 0.3373 |
| pytest-dev__pytest-7324 | baseline_prefix_cache_only | 3862.06 | 582 | 0 |  | 1.0 |
| pytest-dev__pytest-7324 | kvflow_prefix_only | 3855.61 | 582 | 0 |  | 0.8767 |
| pytest-dev__pytest-7324 | kvflow_prefix_plus_codebase_prefetch | 3851.28 | 585 | 0 |  | 0.1887 |
| pytest-dev__pytest-7324 | kvcomm_lossy_plus_codebase_prefetch | 3838.62 | 583 | 0 | exact_code_content_signature | 0.15 |
| pytest-dev__pytest-7432 | baseline_prefix_cache_only | 1695.46 | 823 | 0 |  | 1.0 |
| pytest-dev__pytest-7432 | kvflow_prefix_only | 1698.89 | 823 | 0 |  | 1.0 |
| pytest-dev__pytest-7432 | kvflow_prefix_plus_codebase_prefetch | 1695.44 | 826 | 0 |  | 1.0 |
| pytest-dev__pytest-7432 | kvcomm_lossy_plus_codebase_prefetch | 1680.79 | 824 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-7490 | baseline_prefix_cache_only | 4077.89 | 5019 | 0 |  | 1.0 |
| pytest-dev__pytest-7490 | kvflow_prefix_only | 4052.65 | 5019 | 0 |  | 0.859 |
| pytest-dev__pytest-7490 | kvflow_prefix_plus_codebase_prefetch | 4072.97 | 5022 | 0 |  | 0.6463 |
| pytest-dev__pytest-7490 | kvcomm_lossy_plus_codebase_prefetch | 4057.31 | 5020 | 0 | exact_code_content_signature | 0.2838 |
| scikit-learn__scikit-learn-10297 | baseline_prefix_cache_only | 3992.37 | 1476 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10297 | kvflow_prefix_only | 4597.98 | 1476 | 0 |  | 0.2769 |
| scikit-learn__scikit-learn-10297 | kvflow_prefix_plus_codebase_prefetch | 4095.68 | 1479 | 0 |  | 0.2791 |
| scikit-learn__scikit-learn-10297 | kvcomm_lossy_plus_codebase_prefetch | 4348.38 | 1477 | 0 | exact_code_content_signature | 0.2468 |
| scikit-learn__scikit-learn-10844 | baseline_prefix_cache_only | 4094.06 | 1322 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10844 | kvflow_prefix_only | 4399.49 | 1322 | 0 |  | 0.8429 |
| scikit-learn__scikit-learn-10844 | kvflow_prefix_plus_codebase_prefetch | 4319.84 | 1325 | 0 |  | 0.4937 |
| scikit-learn__scikit-learn-10844 | kvcomm_lossy_plus_codebase_prefetch | 4501.2 | 1323 | 0 | exact_code_content_signature | 0.5217 |
| scikit-learn__scikit-learn-10908 | baseline_prefix_cache_only | 4080.22 | 1254 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10908 | kvflow_prefix_only | 4025.5 | 1254 | 0 |  | 0.032 |
| scikit-learn__scikit-learn-10908 | kvflow_prefix_plus_codebase_prefetch | 4036.43 | 1257 | 0 |  | 0.5 |
| scikit-learn__scikit-learn-10908 | kvcomm_lossy_plus_codebase_prefetch | 4008.25 | 1255 | 0 | exact_code_content_signature | 0.0 |
| scikit-learn__scikit-learn-11310 | baseline_prefix_cache_only | 3989.81 | 833 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-11310 | kvflow_prefix_only | 3990.05 | 833 | 0 |  | 0.4216 |
| scikit-learn__scikit-learn-11310 | kvflow_prefix_plus_codebase_prefetch | 3992.72 | 836 | 0 |  | 0.6207 |
| scikit-learn__scikit-learn-11310 | kvcomm_lossy_plus_codebase_prefetch | 3508.64 | 5937 | 0 | exact_code_content_signature | 0.0161 |

## Interpretation

This benchmark isolates serving-side KVFlow behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and exact-content KVCOMM hits, but host load-back counters remain zero. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
