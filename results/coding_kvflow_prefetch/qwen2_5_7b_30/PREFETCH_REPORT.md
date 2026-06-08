# Coding KVFlow Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/swebench_local_envs/expanded_30_discriminative_instances.json`
- Cases: 28
- HiCache storage backend: `disabled`
- Safety rule: codebase prefetch may predict future code blocks, but KVCOMM reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | avg cached tokens | avg hints | avg prefetch queued | prefetch success | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 28 | 1372.0 | 1253.3 | 0.0 | 0.0 | 0.00 | 0.00 | 1.0000 |
| kvflow_prefix_only | 28 | 1353.8 | 1253.3 | 0.0 | 0.0 | 0.00 | 0.00 | 0.5887 |
| kvflow_prefix_plus_codebase_prefetch | 28 | 1387.7 | 1256.3 | 1.0 | 0.0 | 0.00 | 0.00 | 0.4970 |
| kvcomm_lossy_plus_codebase_prefetch | 28 | 1355.2 | 1606.1 | 1.0 | 0.0 | 0.00 | 1.00 | 0.3927 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_30/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_30/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_30/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | prefetch queued | match reason | token F1 |
|---|---|---:|---:|---:|---|---:|
| astropy__astropy-12907 | baseline_prefix_cache_only | 1285.4 | 1085 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_prefix_only | 1274.46 | 1085 | 0 |  | 0.0351 |
| astropy__astropy-12907 | kvflow_prefix_plus_codebase_prefetch | 1278.39 | 1088 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvcomm_lossy_plus_codebase_prefetch | 1081.92 | 3522 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-13033 | baseline_prefix_cache_only | 1110.95 | 1259 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_prefix_only | 1112.67 | 1259 | 0 |  | 0.9362 |
| astropy__astropy-13033 | kvflow_prefix_plus_codebase_prefetch | 1111.78 | 1262 | 0 |  | 0.2667 |
| astropy__astropy-13033 | kvcomm_lossy_plus_codebase_prefetch | 1063.79 | 1932 | 0 | exact_code_content_signature | 0.0784 |
| astropy__astropy-13236 | baseline_prefix_cache_only | 1323.22 | 2054 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_prefix_only | 1322.31 | 2054 | 0 |  | 0.3099 |
| astropy__astropy-13236 | kvflow_prefix_plus_codebase_prefetch | 1313.38 | 2057 | 0 |  | 0.3333 |
| astropy__astropy-13236 | kvcomm_lossy_plus_codebase_prefetch | 1324.2 | 2055 | 0 | exact_code_content_signature | 0.3934 |
| django__django-10554 | baseline_prefix_cache_only | 1295.55 | 1705 | 0 |  | 1.0 |
| django__django-10554 | kvflow_prefix_only | 1392.49 | 1705 | 0 |  | 0.0625 |
| django__django-10554 | kvflow_prefix_plus_codebase_prefetch | 1399.07 | 1708 | 0 |  | 0.0 |
| django__django-10554 | kvcomm_lossy_plus_codebase_prefetch | 1401.88 | 1706 | 0 | exact_code_content_signature | 0.0 |
| django__django-10880 | baseline_prefix_cache_only | 1167.79 | 600 | 0 |  | 1.0 |
| django__django-10880 | kvflow_prefix_only | 1266.91 | 600 | 0 |  | 0.973 |
| django__django-10880 | kvflow_prefix_plus_codebase_prefetch | 1267.75 | 603 | 0 |  | 0.7714 |
| django__django-10880 | kvcomm_lossy_plus_codebase_prefetch | 1164.08 | 601 | 0 | exact_code_content_signature | 0.6957 |
| matplotlib__matplotlib-13989 | baseline_prefix_cache_only | 1452.67 | 938 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_prefix_only | 1440.51 | 938 | 0 |  | 0.1563 |
| matplotlib__matplotlib-13989 | kvflow_prefix_plus_codebase_prefetch | 1286.63 | 941 | 0 |  | 0.4 |
| matplotlib__matplotlib-13989 | kvcomm_lossy_plus_codebase_prefetch | 1410.6 | 939 | 0 | exact_code_content_signature | 0.2373 |
| matplotlib__matplotlib-14623 | baseline_prefix_cache_only | 1304.19 | 831 | 0 |  | 1.0 |
| matplotlib__matplotlib-14623 | kvflow_prefix_only | 1519.23 | 831 | 0 |  | 0.2899 |
| matplotlib__matplotlib-14623 | kvflow_prefix_plus_codebase_prefetch | 1522.08 | 834 | 0 |  | 0.3284 |
| matplotlib__matplotlib-14623 | kvcomm_lossy_plus_codebase_prefetch | 1552.72 | 832 | 0 | exact_code_content_signature | 0.2182 |
| matplotlib__matplotlib-20488 | baseline_prefix_cache_only | 1443.72 | 1293 | 0 |  | 1.0 |
| matplotlib__matplotlib-20488 | kvflow_prefix_only | 1521.01 | 1293 | 0 |  | 1.0 |
| matplotlib__matplotlib-20488 | kvflow_prefix_plus_codebase_prefetch | 1453.74 | 1296 | 0 |  | 0.8454 |
| matplotlib__matplotlib-20488 | kvcomm_lossy_plus_codebase_prefetch | 1463.33 | 1294 | 0 | exact_code_content_signature | 0.64 |
| mwaskom__seaborn-3069 | baseline_prefix_cache_only | 1441.17 | 933 | 0 |  | 1.0 |
| mwaskom__seaborn-3069 | kvflow_prefix_only | 1299.06 | 933 | 0 |  | 0.5714 |
| mwaskom__seaborn-3069 | kvflow_prefix_plus_codebase_prefetch | 1522.43 | 936 | 0 |  | 0.9118 |
| mwaskom__seaborn-3069 | kvcomm_lossy_plus_codebase_prefetch | 1085.51 | 3635 | 0 | exact_code_content_signature | 0.0 |
| mwaskom__seaborn-3187 | baseline_prefix_cache_only | 1414.71 | 1101 | 0 |  | 1.0 |
| mwaskom__seaborn-3187 | kvflow_prefix_only | 1409.96 | 1101 | 0 |  | 0.6863 |
| mwaskom__seaborn-3187 | kvflow_prefix_plus_codebase_prefetch | 1290.8 | 1104 | 0 |  | 0.5234 |
| mwaskom__seaborn-3187 | kvcomm_lossy_plus_codebase_prefetch | 1447.4 | 1102 | 0 | exact_code_content_signature | 0.5714 |
| pallets__flask-5014 | baseline_prefix_cache_only | 1413.37 | 471 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_prefix_only | 1412.18 | 471 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_prefix_plus_codebase_prefetch | 1411.47 | 474 | 0 |  | 1.0 |
| pallets__flask-5014 | kvcomm_lossy_plus_codebase_prefetch | 1442.77 | 472 | 0 | exact_code_content_signature | 1.0 |
| psf__requests-1142 | baseline_prefix_cache_only | 1436.38 | 538 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_prefix_only | 1288.41 | 538 | 0 |  | 0.5581 |
| psf__requests-1142 | kvflow_prefix_plus_codebase_prefetch | 1399.25 | 541 | 0 |  | 0.5581 |
| psf__requests-1142 | kvcomm_lossy_plus_codebase_prefetch | 1430.37 | 539 | 0 | exact_code_content_signature | 0.4368 |
| psf__requests-1724 | baseline_prefix_cache_only | 1431.45 | 2678 | 0 |  | 1.0 |
| psf__requests-1724 | kvflow_prefix_only | 1430.27 | 2678 | 0 |  | 0.4632 |
| psf__requests-1724 | kvflow_prefix_plus_codebase_prefetch | 1431.19 | 2681 | 0 |  | 0.6471 |
| psf__requests-1724 | kvcomm_lossy_plus_codebase_prefetch | 1466.81 | 2679 | 0 | exact_code_content_signature | 0.3918 |
| psf__requests-1766 | baseline_prefix_cache_only | 1186.58 | 747 | 0 |  | 1.0 |
| psf__requests-1766 | kvflow_prefix_only | 1293.15 | 747 | 0 |  | 1.0 |
| psf__requests-1766 | kvflow_prefix_plus_codebase_prefetch | 1295.25 | 750 | 0 |  | 1.0 |
| psf__requests-1766 | kvcomm_lossy_plus_codebase_prefetch | 1074.82 | 2275 | 0 | exact_code_content_signature | 0.0656 |
| pydata__xarray-2905 | baseline_prefix_cache_only | 1430.03 | 1276 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_prefix_only | 1430.44 | 1276 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_prefix_plus_codebase_prefetch | 1410.42 | 1279 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvcomm_lossy_plus_codebase_prefetch | 1422.08 | 1277 | 0 | exact_code_content_signature | 0.8158 |
| pydata__xarray-3095 | baseline_prefix_cache_only | 1439.2 | 1046 | 0 |  | 1.0 |
| pydata__xarray-3095 | kvflow_prefix_only | 1304.84 | 1046 | 0 |  | 1.0 |
| pydata__xarray-3095 | kvflow_prefix_plus_codebase_prefetch | 1412.98 | 1049 | 0 |  | 0.4632 |
| pydata__xarray-3095 | kvcomm_lossy_plus_codebase_prefetch | 1417.76 | 1047 | 0 | exact_code_content_signature | 0.4632 |
| pydata__xarray-3151 | baseline_prefix_cache_only | 1368.84 | 1402 | 0 |  | 1.0 |
| pydata__xarray-3151 | kvflow_prefix_only | 1346.03 | 1402 | 0 |  | 0.9111 |
| pydata__xarray-3151 | kvflow_prefix_plus_codebase_prefetch | 1439.48 | 1405 | 0 |  | 0.9111 |
| pydata__xarray-3151 | kvcomm_lossy_plus_codebase_prefetch | 1423.12 | 1403 | 0 | exact_code_content_signature | 0.7556 |
| pylint-dev__pylint-4551 | baseline_prefix_cache_only | 1377.95 | 1671 | 0 |  | 1.0 |
| pylint-dev__pylint-4551 | kvflow_prefix_only | 1241.36 | 1671 | 0 |  | 0.5794 |
| pylint-dev__pylint-4551 | kvflow_prefix_plus_codebase_prefetch | 1240.14 | 1674 | 0 |  | 0.5981 |
| pylint-dev__pylint-4551 | kvcomm_lossy_plus_codebase_prefetch | 1350.96 | 1672 | 0 | exact_code_content_signature | 0.5981 |
| pylint-dev__pylint-4604 | baseline_prefix_cache_only | 1582.31 | 1273 | 0 |  | 1.0 |
| pylint-dev__pylint-4604 | kvflow_prefix_only | 1617.18 | 1273 | 0 |  | 1.0 |
| pylint-dev__pylint-4604 | kvflow_prefix_plus_codebase_prefetch | 1501.97 | 1276 | 0 |  | 0.2752 |
| pylint-dev__pylint-4604 | kvcomm_lossy_plus_codebase_prefetch | 1436.44 | 1274 | 0 | exact_code_content_signature | 0.3301 |
| pylint-dev__pylint-4661 | baseline_prefix_cache_only | 1274.76 | 707 | 0 |  | 1.0 |
| pylint-dev__pylint-4661 | kvflow_prefix_only | 939.74 | 707 | 0 |  | 0.381 |
| pylint-dev__pylint-4661 | kvflow_prefix_plus_codebase_prefetch | 1181.55 | 710 | 0 |  | 0.3514 |
| pylint-dev__pylint-4661 | kvcomm_lossy_plus_codebase_prefetch | 1280.54 | 708 | 0 | exact_code_content_signature | 0.4048 |
| pytest-dev__pytest-10051 | baseline_prefix_cache_only | 1434.3 | 1124 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_prefix_only | 1470.01 | 1124 | 0 |  | 0.4894 |
| pytest-dev__pytest-10051 | kvflow_prefix_plus_codebase_prefetch | 1494.26 | 1127 | 0 |  | 0.3457 |
| pytest-dev__pytest-10051 | kvcomm_lossy_plus_codebase_prefetch | 1372.59 | 1125 | 0 | exact_code_content_signature | 0.38 |
| pytest-dev__pytest-10081 | baseline_prefix_cache_only | 1341.63 | 1729 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | kvflow_prefix_only | 1312.91 | 1729 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | kvflow_prefix_plus_codebase_prefetch | 1444.14 | 1732 | 0 |  | 0.8679 |
| pytest-dev__pytest-10081 | kvcomm_lossy_plus_codebase_prefetch | 1445.99 | 1730 | 0 | exact_code_content_signature | 0.8679 |
| pytest-dev__pytest-10356 | baseline_prefix_cache_only | 1434.35 | 1756 | 0 |  | 1.0 |
| pytest-dev__pytest-10356 | kvflow_prefix_only | 1439.72 | 1756 | 0 |  | 0.6813 |
| pytest-dev__pytest-10356 | kvflow_prefix_plus_codebase_prefetch | 1441.05 | 1759 | 0 |  | 0.0632 |
| pytest-dev__pytest-10356 | kvcomm_lossy_plus_codebase_prefetch | 1452.23 | 1757 | 0 | exact_code_content_signature | 0.6596 |
| scikit-learn__scikit-learn-10297 | baseline_prefix_cache_only | 1384.3 | 1459 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10297 | kvflow_prefix_only | 1556.48 | 1459 | 0 |  | 0.2716 |
| scikit-learn__scikit-learn-10297 | kvflow_prefix_plus_codebase_prefetch | 1590.38 | 1462 | 0 |  | 0.1875 |
| scikit-learn__scikit-learn-10297 | kvcomm_lossy_plus_codebase_prefetch | 1485.67 | 1460 | 0 | exact_code_content_signature | 0.2 |
| scikit-learn__scikit-learn-10844 | baseline_prefix_cache_only | 1403.41 | 1306 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10844 | kvflow_prefix_only | 1339.57 | 1306 | 0 |  | 0.0938 |
| scikit-learn__scikit-learn-10844 | kvflow_prefix_plus_codebase_prefetch | 1563.99 | 1309 | 0 |  | 0.1212 |
| scikit-learn__scikit-learn-10844 | kvcomm_lossy_plus_codebase_prefetch | 1562.17 | 1307 | 0 | exact_code_content_signature | 0.0513 |
| scikit-learn__scikit-learn-10908 | baseline_prefix_cache_only | 1461.75 | 1235 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10908 | kvflow_prefix_only | 1300.71 | 1235 | 0 |  | 0.4304 |
| scikit-learn__scikit-learn-10908 | kvflow_prefix_plus_codebase_prefetch | 1486.56 | 1238 | 0 |  | 0.4557 |
| scikit-learn__scikit-learn-10908 | kvcomm_lossy_plus_codebase_prefetch | 1467.99 | 1236 | 0 | exact_code_content_signature | 0.425 |
| sphinx-doc__sphinx-10323 | baseline_prefix_cache_only | 1459.07 | 1351 | 0 |  | 1.0 |
| sphinx-doc__sphinx-10323 | kvflow_prefix_only | 1296.82 | 1351 | 0 |  | 0.3956 |
| sphinx-doc__sphinx-10323 | kvflow_prefix_plus_codebase_prefetch | 1463.94 | 1354 | 0 |  | 0.5 |
| sphinx-doc__sphinx-10323 | kvcomm_lossy_plus_codebase_prefetch | 1087.34 | 3866 | 0 | exact_code_content_signature | 0.0 |
| sphinx-doc__sphinx-10449 | baseline_prefix_cache_only | 1317.87 | 1525 | 0 |  | 1.0 |
| sphinx-doc__sphinx-10449 | kvflow_prefix_only | 1327.98 | 1525 | 0 |  | 0.2069 |
| sphinx-doc__sphinx-10449 | kvflow_prefix_plus_codebase_prefetch | 1202.2 | 1528 | 0 |  | 0.1905 |
| sphinx-doc__sphinx-10449 | kvcomm_lossy_plus_codebase_prefetch | 1330.93 | 1526 | 0 | exact_code_content_signature | 0.3158 |

## Interpretation

This benchmark isolates serving-side KVFlow behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and exact-content KVCOMM hits, but host load-back counters remain zero. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
