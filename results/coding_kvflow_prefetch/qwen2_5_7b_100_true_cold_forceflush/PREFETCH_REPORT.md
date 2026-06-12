# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 100
- Git commit: `3d709f3ce`
- Command: `benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --model /home/gfy/models/Qwen2.5-7B-Instruct --dataset results/repo_level_datasets/swe_verified_100_instances.json --manifest results/repo_level_datasets/manifest_100.json --max-cases 100 --flush-cache-per-case --disable-hierarchical-cache --port 31321 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_true_cold_forceflush`
- Flush cache per case: `True`
- Concurrent clients: `1`
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | p50 | p90 | p99 | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 100 | 1292.5 | 1176.2 | 2285.5 | 2422.5 | 12316.2 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 100 | 1240.7 | 1131.4 | 2101.6 | 2304.9 | 12316.2 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.6851 |
| kvflow_style_prefix_plus_hints | 100 | 1367.5 | 1208.2 | 2276.6 | 2305.6 | 12319.2 | 2.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.5997 |
| agenttemplatekv_exact_reuse | 100 | 1326.8 | 1180.8 | 2260.8 | 2315.4 | 12317.2 | 2.0 | 0.0 | 0.00 | 10698.1 | 0.0 | 0.0 | 1.00 | 0.6070 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_true_cold_forceflush/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_true_cold_forceflush/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100_true_cold_forceflush/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 720.52 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 924.21 | 10547 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2168.48 | 10550 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2285.69 | 10548 | 0 | 9418 | 0 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-13033 | stock_sglang_prefix_only | 1534.34 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_baseline | 1530.09 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_plus_hints | 1532.74 | 10730 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | agenttemplatekv_exact_reuse | 1542.18 | 10728 | 0 | 9424 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13236 | stock_sglang_prefix_only | 2285.27 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_baseline | 2281.77 | 11532 | 0 | 0 | 0 | 0 |  | 0.8264 |
| astropy__astropy-13236 | kvflow_style_prefix_plus_hints | 2282.07 | 11535 | 0 | 0 | 0 | 0 |  | 0.9194 |
| astropy__astropy-13236 | agenttemplatekv_exact_reuse | 2293.68 | 11533 | 0 | 9432 | 0 | 0 | exact_code_content_signature | 0.7667 |
| astropy__astropy-13398 | stock_sglang_prefix_only | 2299.33 | 13240 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13398 | kvflow_style_prefix_baseline | 1130.99 | 13240 | 0 | 0 | 0 | 0 |  | 0.0833 |
| astropy__astropy-13398 | kvflow_style_prefix_plus_hints | 1446.16 | 13243 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | agenttemplatekv_exact_reuse | 1907.42 | 13241 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 0.0702 |
| astropy__astropy-13453 | stock_sglang_prefix_only | 2105.34 | 12492 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_style_prefix_baseline | 2159.57 | 12492 | 0 | 0 | 0 | 0 |  | 0.8116 |
| astropy__astropy-13453 | kvflow_style_prefix_plus_hints | 2295.05 | 12495 | 0 | 0 | 0 | 0 |  | 0.7246 |
| astropy__astropy-13453 | agenttemplatekv_exact_reuse | 2299.99 | 12493 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 0.7222 |
| astropy__astropy-13579 | stock_sglang_prefix_only | 2287.95 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_baseline | 2296.36 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_plus_hints | 2282.4 | 11811 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | agenttemplatekv_exact_reuse | 2293.77 | 11809 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13977 | stock_sglang_prefix_only | 1186.07 | 12895 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13977 | kvflow_style_prefix_baseline | 1186.8 | 12895 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13977 | kvflow_style_prefix_plus_hints | 1186.35 | 12898 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13977 | agenttemplatekv_exact_reuse | 1188.94 | 12896 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14096 | stock_sglang_prefix_only | 960.06 | 10393 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14096 | kvflow_style_prefix_baseline | 939.59 | 10393 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-14096 | kvflow_style_prefix_plus_hints | 957.51 | 10396 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14096 | agenttemplatekv_exact_reuse | 953.22 | 10394 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14182 | stock_sglang_prefix_only | 928.8 | 10766 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14182 | kvflow_style_prefix_baseline | 928.63 | 10766 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14182 | kvflow_style_prefix_plus_hints | 1067.77 | 10769 | 0 | 0 | 0 | 0 |  | 0.4667 |
| astropy__astropy-14182 | agenttemplatekv_exact_reuse | 1055.06 | 10767 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 0.4667 |
| astropy__astropy-14309 | stock_sglang_prefix_only | 1812.89 | 11476 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14309 | kvflow_style_prefix_baseline | 1503.83 | 11476 | 0 | 0 | 0 | 0 |  | 0.64 |
| astropy__astropy-14309 | kvflow_style_prefix_plus_hints | 1817.74 | 11479 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14309 | agenttemplatekv_exact_reuse | 1824.82 | 11477 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14365 | stock_sglang_prefix_only | 1432.05 | 10755 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14365 | kvflow_style_prefix_baseline | 1410.26 | 10755 | 0 | 0 | 0 | 0 |  | 0.9412 |
| astropy__astropy-14365 | kvflow_style_prefix_plus_hints | 1345.68 | 10758 | 0 | 0 | 0 | 0 |  | 0.875 |
| astropy__astropy-14365 | agenttemplatekv_exact_reuse | 1438.22 | 10756 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14369 | stock_sglang_prefix_only | 1134.81 | 11146 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | kvflow_style_prefix_baseline | 1136.33 | 11146 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | kvflow_style_prefix_plus_hints | 1139.92 | 11149 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | agenttemplatekv_exact_reuse | 1142.46 | 11147 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14508 | stock_sglang_prefix_only | 1430.49 | 11088 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14508 | kvflow_style_prefix_baseline | 1430.49 | 11088 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14508 | kvflow_style_prefix_plus_hints | 1431.35 | 11091 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14508 | agenttemplatekv_exact_reuse | 1438.43 | 11089 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14539 | stock_sglang_prefix_only | 2288.26 | 11658 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14539 | kvflow_style_prefix_baseline | 2283.9 | 11658 | 0 | 0 | 0 | 0 |  | 0.2909 |
| astropy__astropy-14539 | kvflow_style_prefix_plus_hints | 2279.3 | 11661 | 0 | 0 | 0 | 0 |  | 0.2909 |
| astropy__astropy-14539 | agenttemplatekv_exact_reuse | 2290.34 | 11659 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.2909 |
| astropy__astropy-14598 | stock_sglang_prefix_only | 1205.3 | 10966 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14598 | kvflow_style_prefix_baseline | 1153.87 | 10966 | 0 | 0 | 0 | 0 |  | 0.9444 |
| astropy__astropy-14598 | kvflow_style_prefix_plus_hints | 1209.57 | 10969 | 0 | 0 | 0 | 0 |  | 0.9444 |
| astropy__astropy-14598 | agenttemplatekv_exact_reuse | 1304.93 | 10967 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.65 |
| django__django-10097 | stock_sglang_prefix_only | 2149.88 | 17295 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | kvflow_style_prefix_baseline | 1495.54 | 17295 | 0 | 0 | 0 | 0 |  | 0.075 |
| django__django-10097 | kvflow_style_prefix_plus_hints | 2154.25 | 17298 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | agenttemplatekv_exact_reuse | 2042.82 | 17296 | 0 | 8999 | 0 | 0 | exact_code_content_signature | 0.3636 |
| django__django-10554 | stock_sglang_prefix_only | 1495.67 | 10700 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10554 | kvflow_style_prefix_baseline | 1309.29 | 10700 | 0 | 0 | 0 | 0 |  | 0.8372 |
| django__django-10554 | kvflow_style_prefix_plus_hints | 1309.61 | 10703 | 0 | 0 | 0 | 0 |  | 0.8372 |
| django__django-10554 | agenttemplatekv_exact_reuse | 1316.09 | 10701 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.8372 |
| django__django-10880 | stock_sglang_prefix_only | 925.01 | 9661 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10880 | kvflow_style_prefix_baseline | 962.92 | 9661 | 0 | 0 | 0 | 0 |  | 0.8387 |
| django__django-10880 | kvflow_style_prefix_plus_hints | 1235.55 | 9664 | 0 | 0 | 0 | 0 |  | 0.8125 |
| django__django-10880 | agenttemplatekv_exact_reuse | 980.94 | 9662 | 0 | 9010 | 0 | 0 | exact_code_content_signature | 0.8667 |
| django__django-10914 | stock_sglang_prefix_only | 1008.24 | 9720 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10914 | kvflow_style_prefix_baseline | 1011.67 | 9720 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10914 | kvflow_style_prefix_plus_hints | 768.67 | 9723 | 0 | 0 | 0 | 0 |  | 0.3333 |
| django__django-10914 | agenttemplatekv_exact_reuse | 774.0 | 9721 | 0 | 9017 | 0 | 0 | exact_code_content_signature | 0.3333 |
| django__django-10973 | stock_sglang_prefix_only | 2269.13 | 10568 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10973 | kvflow_style_prefix_baseline | 2279.51 | 10568 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10973 | kvflow_style_prefix_plus_hints | 2277.16 | 10571 | 0 | 0 | 0 | 0 |  | 0.7761 |
| django__django-10973 | agenttemplatekv_exact_reuse | 2259.77 | 10569 | 0 | 9017 | 0 | 0 | exact_code_content_signature | 0.75 |
| django__django-10999 | stock_sglang_prefix_only | 1298.82 | 9895 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10999 | kvflow_style_prefix_baseline | 1342.91 | 9895 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-10999 | kvflow_style_prefix_plus_hints | 2269.25 | 9898 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-10999 | agenttemplatekv_exact_reuse | 2254.91 | 9896 | 0 | 9017 | 0 | 0 | exact_code_content_signature | 0.0 |
| django__django-11066 | stock_sglang_prefix_only | 1222.74 | 10182 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11066 | kvflow_style_prefix_baseline | 1217.02 | 10182 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11066 | kvflow_style_prefix_plus_hints | 1216.45 | 10185 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11066 | agenttemplatekv_exact_reuse | 1208.92 | 10183 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-11087 | stock_sglang_prefix_only | 1166.62 | 12554 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11087 | kvflow_style_prefix_baseline | 1165.8 | 12554 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11087 | kvflow_style_prefix_plus_hints | 1106.0 | 12557 | 0 | 0 | 0 | 0 |  | 0.9444 |
| django__django-11087 | agenttemplatekv_exact_reuse | 1295.99 | 12555 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.7222 |
| django__django-11095 | stock_sglang_prefix_only | 1302.06 | 9734 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11095 | kvflow_style_prefix_baseline | 2146.08 | 9734 | 0 | 0 | 0 | 0 |  | 0.4 |
| django__django-11095 | kvflow_style_prefix_plus_hints | 2268.75 | 9737 | 0 | 0 | 0 | 0 |  | 0.0816 |
| django__django-11095 | agenttemplatekv_exact_reuse | 1298.72 | 9735 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-11099 | stock_sglang_prefix_only | 836.65 | 9833 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11099 | kvflow_style_prefix_baseline | 837.05 | 9833 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11099 | kvflow_style_prefix_plus_hints | 837.63 | 9836 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11099 | agenttemplatekv_exact_reuse | 839.66 | 9834 | 0 | 9004 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-11119 | stock_sglang_prefix_only | 1129.85 | 9515 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11119 | kvflow_style_prefix_baseline | 1122.54 | 9515 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11119 | kvflow_style_prefix_plus_hints | 1981.23 | 9518 | 0 | 0 | 0 | 0 |  | 0.4103 |
| django__django-11119 | agenttemplatekv_exact_reuse | 1135.08 | 9516 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.6667 |
| django__django-11133 | stock_sglang_prefix_only | 811.12 | 9629 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11133 | kvflow_style_prefix_baseline | 804.6 | 9629 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11133 | kvflow_style_prefix_plus_hints | 1526.94 | 9632 | 0 | 0 | 0 | 0 |  | 0.0606 |
| django__django-11133 | agenttemplatekv_exact_reuse | 1384.52 | 9630 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.0645 |
| django__django-11138 | stock_sglang_prefix_only | 1335.34 | 11193 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11138 | kvflow_style_prefix_baseline | 1529.2 | 11193 | 0 | 0 | 0 | 0 |  | 0.7368 |
| django__django-11138 | kvflow_style_prefix_plus_hints | 1436.11 | 11196 | 0 | 0 | 0 | 0 |  | 0.8947 |
| django__django-11138 | agenttemplatekv_exact_reuse | 1525.54 | 11194 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.7368 |
| django__django-11141 | stock_sglang_prefix_only | 923.27 | 9818 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11141 | kvflow_style_prefix_baseline | 650.94 | 9818 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-11141 | kvflow_style_prefix_plus_hints | 1117.29 | 9821 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-11141 | agenttemplatekv_exact_reuse | 1344.44 | 9819 | 0 | 8914 | 0 | 0 | exact_code_content_signature | 0.1053 |
| django__django-11149 | stock_sglang_prefix_only | 1311.4 | 10734 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11149 | kvflow_style_prefix_baseline | 1310.01 | 10734 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11149 | kvflow_style_prefix_plus_hints | 2285.21 | 10737 | 0 | 0 | 0 | 0 |  | 0.3158 |
| django__django-11149 | agenttemplatekv_exact_reuse | 1297.59 | 10735 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-13989 | stock_sglang_prefix_only | 2279.49 | 10697 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_baseline | 1549.38 | 10697 | 0 | 0 | 0 | 0 |  | 0.1127 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_plus_hints | 2274.52 | 10700 | 0 | 0 | 0 | 0 |  | 0.2927 |
| matplotlib__matplotlib-13989 | agenttemplatekv_exact_reuse | 2270.16 | 10698 | 0 | 9710 | 0 | 0 | exact_code_content_signature | 0.2927 |
| matplotlib__matplotlib-14623 | stock_sglang_prefix_only | 892.18 | 10528 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-14623 | kvflow_style_prefix_baseline | 892.36 | 10528 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-14623 | kvflow_style_prefix_plus_hints | 950.54 | 10531 | 0 | 0 | 0 | 0 |  | 0.7778 |
| matplotlib__matplotlib-14623 | agenttemplatekv_exact_reuse | 935.07 | 10529 | 0 | 9647 | 0 | 0 | exact_code_content_signature | 0.7778 |
| matplotlib__matplotlib-20488 | stock_sglang_prefix_only | 1017.13 | 11128 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20488 | kvflow_style_prefix_baseline | 810.92 | 11128 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-20488 | kvflow_style_prefix_plus_hints | 1451.66 | 11131 | 0 | 0 | 0 | 0 |  | 0.5 |
| matplotlib__matplotlib-20488 | agenttemplatekv_exact_reuse | 1385.19 | 11129 | 0 | 9778 | 0 | 0 | exact_code_content_signature | 0.5 |
| matplotlib__matplotlib-20676 | stock_sglang_prefix_only | 1605.11 | 10977 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20676 | kvflow_style_prefix_baseline | 2063.59 | 10977 | 0 | 0 | 0 | 0 |  | 0.087 |
| matplotlib__matplotlib-20676 | kvflow_style_prefix_plus_hints | 1389.55 | 10980 | 0 | 0 | 0 | 0 |  | 0.0889 |
| matplotlib__matplotlib-20676 | agenttemplatekv_exact_reuse | 1561.22 | 10978 | 0 | 9787 | 0 | 0 | exact_code_content_signature | 0.05 |
| matplotlib__matplotlib-20826 | stock_sglang_prefix_only | 619.37 | 11134 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20826 | kvflow_style_prefix_baseline | 619.27 | 11134 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20826 | kvflow_style_prefix_plus_hints | 982.15 | 11137 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-20826 | agenttemplatekv_exact_reuse | 990.1 | 11135 | 0 | 9787 | 0 | 0 | exact_code_content_signature | 0.0 |
| matplotlib__matplotlib-20859 | stock_sglang_prefix_only | 894.09 | 10809 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20859 | kvflow_style_prefix_baseline | 1276.87 | 10809 | 0 | 0 | 0 | 0 |  | 0.5833 |
| matplotlib__matplotlib-20859 | kvflow_style_prefix_plus_hints | 1482.69 | 10812 | 0 | 0 | 0 | 0 |  | 0.4364 |
| matplotlib__matplotlib-20859 | agenttemplatekv_exact_reuse | 1283.67 | 10810 | 0 | 9589 | 0 | 0 | exact_code_content_signature | 0.5833 |
| matplotlib__matplotlib-21568 | stock_sglang_prefix_only | 1238.56 | 11430 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-21568 | kvflow_style_prefix_baseline | 1088.57 | 11430 | 0 | 0 | 0 | 0 |  | 0.8 |
| matplotlib__matplotlib-21568 | kvflow_style_prefix_plus_hints | 1296.16 | 11433 | 0 | 0 | 0 | 0 |  | 0.7 |
| matplotlib__matplotlib-21568 | agenttemplatekv_exact_reuse | 1093.87 | 11431 | 0 | 9576 | 0 | 0 | exact_code_content_signature | 0.8 |
| matplotlib__matplotlib-22719 | stock_sglang_prefix_only | 1185.71 | 12837 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22719 | kvflow_style_prefix_baseline | 1570.66 | 12837 | 0 | 0 | 0 | 0 |  | 0.5455 |
| matplotlib__matplotlib-22719 | kvflow_style_prefix_plus_hints | 1538.0 | 12840 | 0 | 0 | 0 | 0 |  | 0.5455 |
| matplotlib__matplotlib-22719 | agenttemplatekv_exact_reuse | 1540.99 | 12838 | 0 | 9581 | 0 | 0 | exact_code_content_signature | 0.5455 |
| matplotlib__matplotlib-22865 | stock_sglang_prefix_only | 1454.49 | 10951 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22865 | kvflow_style_prefix_baseline | 2053.87 | 10951 | 0 | 0 | 0 | 0 |  | 0.0976 |
| matplotlib__matplotlib-22865 | kvflow_style_prefix_plus_hints | 1466.94 | 10954 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-22865 | agenttemplatekv_exact_reuse | 2060.85 | 10952 | 0 | 9612 | 0 | 0 | exact_code_content_signature | 0.0976 |
| matplotlib__matplotlib-22871 | stock_sglang_prefix_only | 1341.83 | 10573 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22871 | kvflow_style_prefix_baseline | 1342.59 | 10573 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22871 | kvflow_style_prefix_plus_hints | 1339.37 | 10576 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22871 | agenttemplatekv_exact_reuse | 1355.23 | 10574 | 0 | 9612 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-23299 | stock_sglang_prefix_only | 627.21 | 10660 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23299 | kvflow_style_prefix_baseline | 621.04 | 10660 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23299 | kvflow_style_prefix_plus_hints | 625.09 | 10663 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-23299 | agenttemplatekv_exact_reuse | 608.6 | 10661 | 0 | 9622 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-23314 | stock_sglang_prefix_only | 1330.15 | 10579 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23314 | kvflow_style_prefix_baseline | 1323.7 | 10579 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23314 | kvflow_style_prefix_plus_hints | 1327.55 | 10582 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23314 | agenttemplatekv_exact_reuse | 1319.93 | 10580 | 0 | 9622 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-23412 | stock_sglang_prefix_only | 1423.2 | 11172 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23412 | kvflow_style_prefix_baseline | 1617.08 | 11172 | 0 | 0 | 0 | 0 |  | 0.2 |
| matplotlib__matplotlib-23412 | kvflow_style_prefix_plus_hints | 1643.63 | 11175 | 0 | 0 | 0 | 0 |  | 0.2667 |
| matplotlib__matplotlib-23412 | agenttemplatekv_exact_reuse | 1631.17 | 11173 | 0 | 9622 | 0 | 0 | exact_code_content_signature | 0.2667 |
| matplotlib__matplotlib-23476 | stock_sglang_prefix_only | 1107.78 | 11435 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23476 | kvflow_style_prefix_baseline | 1111.65 | 11435 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23476 | kvflow_style_prefix_plus_hints | 1162.52 | 11438 | 0 | 0 | 0 | 0 |  | 0.3636 |
| matplotlib__matplotlib-23476 | agenttemplatekv_exact_reuse | 1062.69 | 11436 | 0 | 9622 | 0 | 0 | exact_code_content_signature | 0.1765 |
| matplotlib__matplotlib-24026 | stock_sglang_prefix_only | 1052.5 | 10866 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-24026 | kvflow_style_prefix_baseline | 1057.33 | 10866 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-24026 | kvflow_style_prefix_plus_hints | 1052.13 | 10869 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-24026 | agenttemplatekv_exact_reuse | 1045.92 | 10867 | 0 | 9627 | 0 | 0 | exact_code_content_signature | 1.0 |
| mwaskom__seaborn-3069 | stock_sglang_prefix_only | 2276.39 | 10253 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3069 | kvflow_style_prefix_baseline | 828.49 | 10253 | 0 | 0 | 0 | 0 |  | 0.1538 |
| mwaskom__seaborn-3069 | kvflow_style_prefix_plus_hints | 826.11 | 10256 | 0 | 0 | 0 | 0 |  | 0.1538 |
| mwaskom__seaborn-3069 | agenttemplatekv_exact_reuse | 2275.46 | 10254 | 0 | 9281 | 0 | 0 | exact_code_content_signature | 0.16 |
| mwaskom__seaborn-3187 | stock_sglang_prefix_only | 1015.49 | 10422 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3187 | kvflow_style_prefix_baseline | 1025.45 | 10422 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3187 | kvflow_style_prefix_plus_hints | 2276.57 | 10425 | 0 | 0 | 0 | 0 |  | 0.439 |
| mwaskom__seaborn-3187 | agenttemplatekv_exact_reuse | 1089.44 | 10423 | 0 | 9282 | 0 | 0 | exact_code_content_signature | 0.7692 |
| pallets__flask-5014 | stock_sglang_prefix_only | 846.1 | 10368 | 0 | 0 | 0 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_style_prefix_baseline | 853.9 | 10368 | 0 | 0 | 0 | 0 |  | 0.5714 |
| pallets__flask-5014 | kvflow_style_prefix_plus_hints | 848.62 | 10371 | 0 | 0 | 0 | 0 |  | 0.5714 |
| pallets__flask-5014 | agenttemplatekv_exact_reuse | 830.69 | 10369 | 0 | 9854 | 0 | 0 | exact_code_content_signature | 0.5714 |
| psf__requests-1142 | stock_sglang_prefix_only | 770.53 | 25580 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_style_prefix_baseline | 789.35 | 25580 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_style_prefix_plus_hints | 1206.87 | 25583 | 0 | 0 | 0 | 0 |  | 0.2308 |
| psf__requests-1142 | agenttemplatekv_exact_reuse | 1172.64 | 25581 | 0 | 24993 | 0 | 0 | exact_code_content_signature | 0.3077 |
| psf__requests-1724 | stock_sglang_prefix_only | 971.64 | 27884 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1724 | kvflow_style_prefix_baseline | 719.99 | 27884 | 0 | 0 | 0 | 0 |  | 0.4 |
| psf__requests-1724 | kvflow_style_prefix_plus_hints | 971.32 | 27887 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1724 | agenttemplatekv_exact_reuse | 965.02 | 27885 | 0 | 25158 | 0 | 0 | exact_code_content_signature | 1.0 |
| psf__requests-1766 | stock_sglang_prefix_only | 633.7 | 25978 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1766 | kvflow_style_prefix_baseline | 634.44 | 25978 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1766 | kvflow_style_prefix_plus_hints | 635.28 | 25981 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1766 | agenttemplatekv_exact_reuse | 640.96 | 25979 | 0 | 25181 | 0 | 0 | exact_code_content_signature | 1.0 |
| psf__requests-1921 | stock_sglang_prefix_only | 702.61 | 26036 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1921 | kvflow_style_prefix_baseline | 704.92 | 26036 | 0 | 0 | 0 | 0 |  | 0.5 |
| psf__requests-1921 | kvflow_style_prefix_plus_hints | 690.44 | 26039 | 0 | 0 | 0 | 0 |  | 0.5 |
| psf__requests-1921 | agenttemplatekv_exact_reuse | 693.09 | 26037 | 0 | 25263 | 0 | 0 | exact_code_content_signature | 0.5 |
| psf__requests-2317 | stock_sglang_prefix_only | 2422.49 | 26119 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-2317 | kvflow_style_prefix_baseline | 1013.26 | 26119 | 0 | 0 | 0 | 0 |  | 0.1951 |
| psf__requests-2317 | kvflow_style_prefix_plus_hints | 654.43 | 26122 | 0 | 0 | 0 | 0 |  | 0.2051 |
| psf__requests-2317 | agenttemplatekv_exact_reuse | 665.43 | 26120 | 0 | 25318 | 0 | 0 | exact_code_content_signature | 0.2051 |
| psf__requests-2931 | stock_sglang_prefix_only | 2423.96 | 25942 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-2931 | kvflow_style_prefix_baseline | 872.94 | 25942 | 0 | 0 | 0 | 0 |  | 0.1562 |
| psf__requests-2931 | kvflow_style_prefix_plus_hints | 2427.81 | 25945 | 0 | 0 | 0 | 0 |  | 0.9107 |
| psf__requests-2931 | agenttemplatekv_exact_reuse | 627.77 | 25943 | 0 | 25364 | 0 | 0 | exact_code_content_signature | 0.0 |
| psf__requests-5414 | stock_sglang_prefix_only | 985.67 | 10849 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-5414 | kvflow_style_prefix_baseline | 989.61 | 10849 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-5414 | kvflow_style_prefix_plus_hints | 987.24 | 10852 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-5414 | agenttemplatekv_exact_reuse | 993.87 | 10850 | 0 | 9911 | 0 | 0 | exact_code_content_signature | 1.0 |
| psf__requests-6028 | stock_sglang_prefix_only | 1595.05 | 10881 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-6028 | kvflow_style_prefix_baseline | 1593.17 | 10881 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-6028 | kvflow_style_prefix_plus_hints | 1479.99 | 10884 | 0 | 0 | 0 | 0 |  | 0.7667 |
| psf__requests-6028 | agenttemplatekv_exact_reuse | 1546.29 | 10882 | 0 | 9914 | 0 | 0 | exact_code_content_signature | 0.6102 |
| pydata__xarray-2905 | stock_sglang_prefix_only | 1241.15 | 11893 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_style_prefix_baseline | 1226.63 | 11893 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_style_prefix_plus_hints | 1227.9 | 11896 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | agenttemplatekv_exact_reuse | 1219.74 | 11894 | 0 | 10569 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-3095 | stock_sglang_prefix_only | 1328.78 | 10634 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3095 | kvflow_style_prefix_baseline | 1105.69 | 10634 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-3095 | kvflow_style_prefix_plus_hints | 1115.98 | 10637 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-3095 | agenttemplatekv_exact_reuse | 1096.22 | 10635 | 0 | 9541 | 0 | 0 | exact_code_content_signature | 0.0 |
| pydata__xarray-3151 | stock_sglang_prefix_only | 720.11 | 11000 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3151 | kvflow_style_prefix_baseline | 717.11 | 11000 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3151 | kvflow_style_prefix_plus_hints | 709.33 | 11003 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3151 | agenttemplatekv_exact_reuse | 787.74 | 11001 | 0 | 9551 | 0 | 0 | exact_code_content_signature | 0.7273 |
| pydata__xarray-3305 | stock_sglang_prefix_only | 993.48 | 10923 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3305 | kvflow_style_prefix_baseline | 995.4 | 10923 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3305 | kvflow_style_prefix_plus_hints | 983.45 | 10926 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3305 | agenttemplatekv_exact_reuse | 972.56 | 10924 | 0 | 9665 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-3677 | stock_sglang_prefix_only | 739.09 | 10748 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3677 | kvflow_style_prefix_baseline | 1341.54 | 10748 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-3677 | kvflow_style_prefix_plus_hints | 1350.9 | 10751 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-3677 | agenttemplatekv_exact_reuse | 1403.62 | 10749 | 0 | 9762 | 0 | 0 | exact_code_content_signature | 0.0 |
| pydata__xarray-3993 | stock_sglang_prefix_only | 1048.6 | 11429 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3993 | kvflow_style_prefix_baseline | 754.05 | 11429 | 0 | 0 | 0 | 0 |  | 0.2222 |
| pydata__xarray-3993 | kvflow_style_prefix_plus_hints | 947.92 | 11432 | 0 | 0 | 0 | 0 |  | 0.8667 |
| pydata__xarray-3993 | agenttemplatekv_exact_reuse | 941.31 | 11430 | 0 | 10569 | 0 | 0 | exact_code_content_signature | 0.8667 |
| pydata__xarray-4075 | stock_sglang_prefix_only | 1215.43 | 11270 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4075 | kvflow_style_prefix_baseline | 1219.49 | 11270 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4075 | kvflow_style_prefix_plus_hints | 1183.52 | 11273 | 0 | 0 | 0 | 0 |  | 0.8333 |
| pydata__xarray-4075 | agenttemplatekv_exact_reuse | 1201.46 | 11271 | 0 | 9785 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-4094 | stock_sglang_prefix_only | 2130.51 | 10959 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4094 | kvflow_style_prefix_baseline | 1637.74 | 10959 | 0 | 0 | 0 | 0 |  | 0.459 |
| pydata__xarray-4094 | kvflow_style_prefix_plus_hints | 1984.61 | 10962 | 0 | 0 | 0 | 0 |  | 0.9091 |
| pydata__xarray-4094 | agenttemplatekv_exact_reuse | 1446.17 | 10960 | 0 | 9788 | 0 | 0 | exact_code_content_signature | 0.459 |
| pydata__xarray-4356 | stock_sglang_prefix_only | 1031.07 | 11103 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4356 | kvflow_style_prefix_baseline | 674.41 | 11103 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-4356 | kvflow_style_prefix_plus_hints | 1021.65 | 11106 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4356 | agenttemplatekv_exact_reuse | 1010.07 | 11104 | 0 | 9791 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-4629 | stock_sglang_prefix_only | 665.37 | 12022 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4629 | kvflow_style_prefix_baseline | 662.23 | 12022 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4629 | kvflow_style_prefix_plus_hints | 1023.76 | 12025 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-4629 | agenttemplatekv_exact_reuse | 1016.29 | 12023 | 0 | 10559 | 0 | 0 | exact_code_content_signature | 0.0 |
| pydata__xarray-4687 | stock_sglang_prefix_only | 2308.08 | 12615 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4687 | kvflow_style_prefix_baseline | 2299.83 | 12615 | 0 | 0 | 0 | 0 |  | 0.0215 |
| pydata__xarray-4687 | kvflow_style_prefix_plus_hints | 957.81 | 12618 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-4687 | agenttemplatekv_exact_reuse | 950.05 | 12616 | 0 | 10540 | 0 | 0 | exact_code_content_signature | 0.0 |
| pydata__xarray-4695 | stock_sglang_prefix_only | 2294.8 | 12464 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4695 | kvflow_style_prefix_baseline | 1496.24 | 12464 | 0 | 0 | 0 | 0 |  | 0.0274 |
| pydata__xarray-4695 | kvflow_style_prefix_plus_hints | 2298.73 | 12467 | 0 | 0 | 0 | 0 |  | 0.6265 |
| pydata__xarray-4695 | agenttemplatekv_exact_reuse | 2330.16 | 12465 | 0 | 10564 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-4966 | stock_sglang_prefix_only | 1533.55 | 12646 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4966 | kvflow_style_prefix_baseline | 1531.64 | 12646 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4966 | kvflow_style_prefix_plus_hints | 1216.96 | 12649 | 0 | 0 | 0 | 0 |  | 0.1017 |
| pydata__xarray-4966 | agenttemplatekv_exact_reuse | 1193.9 | 12647 | 0 | 10569 | 0 | 0 | exact_code_content_signature | 0.1017 |
| pydata__xarray-6461 | stock_sglang_prefix_only | 1002.69 | 11435 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6461 | kvflow_style_prefix_baseline | 1008.54 | 11435 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6461 | kvflow_style_prefix_plus_hints | 1072.11 | 11438 | 0 | 0 | 0 | 0 |  | 0.9444 |
| pydata__xarray-6461 | agenttemplatekv_exact_reuse | 1078.45 | 11436 | 0 | 10634 | 0 | 0 | exact_code_content_signature | 0.9444 |
| pydata__xarray-6599 | stock_sglang_prefix_only | 592.44 | 13300 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6599 | kvflow_style_prefix_baseline | 595.64 | 13300 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6599 | kvflow_style_prefix_plus_hints | 596.0 | 13303 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6599 | agenttemplatekv_exact_reuse | 599.69 | 13301 | 0 | 10636 | 0 | 0 | exact_code_content_signature | 1.0 |
| pylint-dev__pylint-4551 | stock_sglang_prefix_only | 2302.99 | 12914 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4551 | kvflow_style_prefix_baseline | 2299.66 | 12914 | 0 | 0 | 0 | 0 |  | 0.8958 |
| pylint-dev__pylint-4551 | kvflow_style_prefix_plus_hints | 2304.32 | 12917 | 0 | 0 | 0 | 0 |  | 0.5581 |
| pylint-dev__pylint-4551 | agenttemplatekv_exact_reuse | 2290.93 | 12915 | 0 | 11201 | 0 | 0 | exact_code_content_signature | 0.5581 |
| pylint-dev__pylint-4604 | stock_sglang_prefix_only | 608.02 | 12517 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4604 | kvflow_style_prefix_baseline | 601.32 | 12517 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4604 | kvflow_style_prefix_plus_hints | 1044.22 | 12520 | 0 | 0 | 0 | 0 |  | 0.0769 |
| pylint-dev__pylint-4604 | agenttemplatekv_exact_reuse | 1039.1 | 12518 | 0 | 11201 | 0 | 0 | exact_code_content_signature | 0.0769 |
| pylint-dev__pylint-4661 | stock_sglang_prefix_only | 796.81 | 11971 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4661 | kvflow_style_prefix_baseline | 1178.74 | 11971 | 0 | 0 | 0 | 0 |  | 0.0 |
| pylint-dev__pylint-4661 | kvflow_style_prefix_plus_hints | 1183.83 | 11974 | 0 | 0 | 0 | 0 |  | 0.0 |
| pylint-dev__pylint-4661 | agenttemplatekv_exact_reuse | 1171.63 | 11972 | 0 | 11222 | 0 | 0 | exact_code_content_signature | 0.0 |
| pylint-dev__pylint-4970 | stock_sglang_prefix_only | 1371.52 | 11884 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4970 | kvflow_style_prefix_baseline | 1144.84 | 11884 | 0 | 0 | 0 | 0 |  | 0.0 |
| pylint-dev__pylint-4970 | kvflow_style_prefix_plus_hints | 1265.5 | 11887 | 0 | 0 | 0 | 0 |  | 0.0 |
| pylint-dev__pylint-4970 | agenttemplatekv_exact_reuse | 1258.44 | 11885 | 0 | 11199 | 0 | 0 | exact_code_content_signature | 0.0 |
| pylint-dev__pylint-6386 | stock_sglang_prefix_only | 893.47 | 10593 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6386 | kvflow_style_prefix_baseline | 1589.74 | 10593 | 0 | 0 | 0 | 0 |  | 0.3396 |
| pylint-dev__pylint-6386 | kvflow_style_prefix_plus_hints | 1111.12 | 10596 | 0 | 0 | 0 | 0 |  | 0.0 |
| pylint-dev__pylint-6386 | agenttemplatekv_exact_reuse | 1091.74 | 10594 | 0 | 9831 | 0 | 0 | exact_code_content_signature | 0.0 |
| pylint-dev__pylint-6528 | stock_sglang_prefix_only | 2266.53 | 12421 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6528 | kvflow_style_prefix_baseline | 2304.66 | 12421 | 0 | 0 | 0 | 0 |  | 0.9589 |
| pylint-dev__pylint-6528 | kvflow_style_prefix_plus_hints | 2098.58 | 12424 | 0 | 0 | 0 | 0 |  | 0.8493 |
| pylint-dev__pylint-6528 | agenttemplatekv_exact_reuse | 2102.86 | 12422 | 0 | 10020 | 0 | 0 | exact_code_content_signature | 0.8493 |
| pylint-dev__pylint-6903 | stock_sglang_prefix_only | 1022.35 | 11862 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6903 | kvflow_style_prefix_baseline | 1025.78 | 11862 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6903 | kvflow_style_prefix_plus_hints | 1026.4 | 11865 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6903 | agenttemplatekv_exact_reuse | 1015.69 | 11863 | 0 | 10022 | 0 | 0 | exact_code_content_signature | 1.0 |
| pylint-dev__pylint-7080 | stock_sglang_prefix_only | 619.19 | 18382 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7080 | kvflow_style_prefix_baseline | 616.52 | 18382 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7080 | kvflow_style_prefix_plus_hints | 601.88 | 18385 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7080 | agenttemplatekv_exact_reuse | 698.59 | 18383 | 0 | 10033 | 0 | 0 | exact_code_content_signature | 0.0 |
| pylint-dev__pylint-7277 | stock_sglang_prefix_only | 941.4 | 10935 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7277 | kvflow_style_prefix_baseline | 931.14 | 10935 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7277 | kvflow_style_prefix_plus_hints | 1571.63 | 10938 | 0 | 0 | 0 | 0 |  | 0.0541 |
| pylint-dev__pylint-7277 | agenttemplatekv_exact_reuse | 1564.82 | 10936 | 0 | 10056 | 0 | 0 | exact_code_content_signature | 0.0541 |
| pylint-dev__pylint-8898 | stock_sglang_prefix_only | 1006.57 | 11924 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-8898 | kvflow_style_prefix_baseline | 1021.89 | 11924 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-8898 | kvflow_style_prefix_plus_hints | 1012.19 | 11927 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-8898 | agenttemplatekv_exact_reuse | 1012.86 | 11925 | 0 | 9966 | 0 | 0 | exact_code_content_signature | 0.9375 |
| pytest-dev__pytest-10051 | stock_sglang_prefix_only | 931.32 | 11091 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_style_prefix_baseline | 932.94 | 11091 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_style_prefix_plus_hints | 932.62 | 11094 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | agenttemplatekv_exact_reuse | 939.7 | 11092 | 0 | 9926 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-10081 | stock_sglang_prefix_only | 1129.87 | 11695 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | kvflow_style_prefix_baseline | 1131.87 | 11695 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | kvflow_style_prefix_plus_hints | 1129.44 | 11698 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | agenttemplatekv_exact_reuse | 1139.26 | 11696 | 0 | 9926 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-10356 | stock_sglang_prefix_only | 1594.69 | 11721 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10356 | kvflow_style_prefix_baseline | 1600.41 | 11721 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10356 | kvflow_style_prefix_plus_hints | 1667.61 | 11724 | 0 | 0 | 0 | 0 |  | 0.9836 |
| pytest-dev__pytest-10356 | agenttemplatekv_exact_reuse | 1674.75 | 11722 | 0 | 9926 | 0 | 0 | exact_code_content_signature | 0.9836 |
| pytest-dev__pytest-5262 | stock_sglang_prefix_only | 875.89 | 11631 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5262 | kvflow_style_prefix_baseline | 885.73 | 11631 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5262 | kvflow_style_prefix_plus_hints | 957.25 | 11634 | 0 | 0 | 0 | 0 |  | 0.9032 |
| pytest-dev__pytest-5262 | agenttemplatekv_exact_reuse | 945.72 | 11632 | 0 | 9958 | 0 | 0 | exact_code_content_signature | 0.9032 |
| pytest-dev__pytest-5631 | stock_sglang_prefix_only | 1126.41 | 11267 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5631 | kvflow_style_prefix_baseline | 1125.32 | 11267 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5631 | kvflow_style_prefix_plus_hints | 1139.36 | 11270 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5631 | agenttemplatekv_exact_reuse | 1132.17 | 11268 | 0 | 9912 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-5787 | stock_sglang_prefix_only | 1187.06 | 12939 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5787 | kvflow_style_prefix_baseline | 1187.33 | 12939 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5787 | kvflow_style_prefix_plus_hints | 2302.5 | 12942 | 0 | 0 | 0 | 0 |  | 0.3871 |
| pytest-dev__pytest-5787 | agenttemplatekv_exact_reuse | 2315.26 | 12940 | 0 | 9943 | 0 | 0 | exact_code_content_signature | 0.339 |
| pytest-dev__pytest-5809 | stock_sglang_prefix_only | 826.92 | 10734 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5809 | kvflow_style_prefix_baseline | 837.17 | 10734 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5809 | kvflow_style_prefix_plus_hints | 828.04 | 10737 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5809 | agenttemplatekv_exact_reuse | 813.7 | 10735 | 0 | 9941 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-5840 | stock_sglang_prefix_only | 911.74 | 11421 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5840 | kvflow_style_prefix_baseline | 1012.27 | 11421 | 0 | 0 | 0 | 0 |  | 0.8667 |
| pytest-dev__pytest-5840 | kvflow_style_prefix_plus_hints | 890.22 | 11424 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5840 | agenttemplatekv_exact_reuse | 890.59 | 11422 | 0 | 9939 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-6197 | stock_sglang_prefix_only | 1283.44 | 11677 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-6197 | kvflow_style_prefix_baseline | 960.17 | 11677 | 0 | 0 | 0 | 0 |  | 0.0 |
| pytest-dev__pytest-6197 | kvflow_style_prefix_plus_hints | 1176.56 | 11680 | 0 | 0 | 0 | 0 |  | 0.0 |
| pytest-dev__pytest-6197 | agenttemplatekv_exact_reuse | 1169.34 | 11678 | 0 | 9925 | 0 | 0 | exact_code_content_signature | 0.0 |
| pytest-dev__pytest-6202 | stock_sglang_prefix_only | 979.6 | 11404 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-6202 | kvflow_style_prefix_baseline | 982.12 | 11404 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-6202 | kvflow_style_prefix_plus_hints | 980.07 | 11407 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-6202 | agenttemplatekv_exact_reuse | 958.78 | 11405 | 0 | 9925 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-7205 | stock_sglang_prefix_only | 1271.32 | 12275 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7205 | kvflow_style_prefix_baseline | 1266.53 | 12275 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7205 | kvflow_style_prefix_plus_hints | 1498.94 | 12278 | 0 | 0 | 0 | 0 |  | 0.7647 |
| pytest-dev__pytest-7205 | agenttemplatekv_exact_reuse | 1504.75 | 12276 | 0 | 10037 | 0 | 0 | exact_code_content_signature | 0.7647 |
| pytest-dev__pytest-7236 | stock_sglang_prefix_only | 1188.81 | 11417 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7236 | kvflow_style_prefix_baseline | 1136.45 | 11417 | 0 | 0 | 0 | 0 |  | 0.9302 |
| pytest-dev__pytest-7236 | kvflow_style_prefix_plus_hints | 1191.62 | 11420 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7236 | agenttemplatekv_exact_reuse | 1168.71 | 11418 | 0 | 10046 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-7324 | stock_sglang_prefix_only | 998.94 | 10665 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7324 | kvflow_style_prefix_baseline | 1008.87 | 10665 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7324 | kvflow_style_prefix_plus_hints | 1013.02 | 10668 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7324 | agenttemplatekv_exact_reuse | 989.71 | 10666 | 0 | 10054 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-7432 | stock_sglang_prefix_only | 2288.0 | 10906 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7432 | kvflow_style_prefix_baseline | 2096.61 | 10906 | 0 | 0 | 0 | 0 |  | 0.7429 |
| pytest-dev__pytest-7432 | kvflow_style_prefix_plus_hints | 2098.46 | 10909 | 0 | 0 | 0 | 0 |  | 0.7429 |
| pytest-dev__pytest-7432 | agenttemplatekv_exact_reuse | 2101.18 | 10907 | 0 | 10054 | 0 | 0 | exact_code_content_signature | 0.7429 |
| pytest-dev__pytest-7490 | stock_sglang_prefix_only | 625.22 | 15104 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7490 | kvflow_style_prefix_baseline | 2326.64 | 15104 | 0 | 0 | 0 | 0 |  | 0.0 |
| pytest-dev__pytest-7490 | kvflow_style_prefix_plus_hints | 997.45 | 15107 | 0 | 0 | 0 | 0 |  | 0.0 |
| pytest-dev__pytest-7490 | agenttemplatekv_exact_reuse | 964.0 | 15105 | 0 | 10057 | 0 | 0 | exact_code_content_signature | 0.0 |
| scikit-learn__scikit-learn-10297 | stock_sglang_prefix_only | 2290.4 | 12062 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10297 | kvflow_style_prefix_baseline | 1929.14 | 12062 | 0 | 0 | 0 | 0 |  | 0.0 |
| scikit-learn__scikit-learn-10297 | kvflow_style_prefix_plus_hints | 1934.47 | 12065 | 0 | 0 | 0 | 0 |  | 0.0 |
| scikit-learn__scikit-learn-10297 | agenttemplatekv_exact_reuse | 1920.52 | 12063 | 0 | 10552 | 0 | 0 | exact_code_content_signature | 0.0 |
| scikit-learn__scikit-learn-10844 | stock_sglang_prefix_only | 1269.52 | 11909 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10844 | kvflow_style_prefix_baseline | 1266.56 | 11909 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10844 | kvflow_style_prefix_plus_hints | 1633.31 | 11912 | 0 | 0 | 0 | 0 |  | 0.6129 |
| scikit-learn__scikit-learn-10844 | agenttemplatekv_exact_reuse | 1621.97 | 11910 | 0 | 10555 | 0 | 0 | exact_code_content_signature | 0.6129 |
| scikit-learn__scikit-learn-10908 | stock_sglang_prefix_only | 1093.58 | 11832 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10908 | kvflow_style_prefix_baseline | 1106.91 | 11832 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10908 | kvflow_style_prefix_plus_hints | 1176.95 | 11835 | 0 | 0 | 0 | 0 |  | 0.1765 |
| scikit-learn__scikit-learn-10908 | agenttemplatekv_exact_reuse | 1084.09 | 11833 | 0 | 10545 | 0 | 0 | exact_code_content_signature | 1.0 |
| scikit-learn__scikit-learn-11310 | stock_sglang_prefix_only | 1197.06 | 11421 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-11310 | kvflow_style_prefix_baseline | 790.92 | 11421 | 0 | 0 | 0 | 0 |  | 0.2222 |
| scikit-learn__scikit-learn-11310 | kvflow_style_prefix_plus_hints | 1071.79 | 11424 | 0 | 0 | 0 | 0 |  | 0.1429 |
| scikit-learn__scikit-learn-11310 | agenttemplatekv_exact_reuse | 992.68 | 11422 | 0 | 10556 | 0 | 0 | exact_code_content_signature | 0.1538 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
