# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 48
- Git commit: `5fb934751`
- Command: `/home/gfy/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --model /home/gfy/models/Qwen2.5-7B-Instruct --dataset results/repo_level_datasets/swe_verified_100_instances.json --manifest results/repo_level_datasets/manifest_100.json --max-cases 100 --max-tokens 128 --baseline-profile lmcache --port 31345 --server-timeout 600 --eval-timeout 3600 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613`
- Flush cache per case: `False`
- Concurrent clients: `1`
- Baseline profile: `lmcache`
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Server extra args: ``
- Resolved server extra args: `--enable-lmcache`
- LMCache config: `/home/gfy/CodeMAS_Project/sglang-kvflow/python/sglang/srt/mem_cache/storage/lmcache/example_config.yaml`
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | p50 | p90 | p99 | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 48 | 1422.4 | 1304.7 | 2295.2 | 2398.4 | 11027.9 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 48 | 1340.2 | 1287.0 | 2173.6 | 2377.6 | 11027.9 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.6562 |
| kvflow_style_prefix_plus_hints | 48 | 1532.4 | 1440.6 | 2307.5 | 2363.6 | 11030.9 | 2.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.6095 |
| agenttemplatekv_exact_reuse | 48 | 1554.4 | 1439.4 | 2321.2 | 2477.3 | 11028.9 | 2.0 | 0.0 | 0.00 | 9367.9 | 0.0 | 0.0 | 1.00 | 0.5688 |

## Figures

Figure generation skipped because a clean `matplotlib` import is unavailable in this environment.

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 736.89 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 930.77 | 10547 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2186.16 | 10550 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2279.08 | 10548 | 0 | 9418 | 0 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-13033 | stock_sglang_prefix_only | 1559.78 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_baseline | 1583.75 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_plus_hints | 1564.88 | 10730 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | agenttemplatekv_exact_reuse | 1602.36 | 10728 | 0 | 9424 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13236 | stock_sglang_prefix_only | 2307.22 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_baseline | 2307.71 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_plus_hints | 2303.06 | 11535 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | agenttemplatekv_exact_reuse | 1754.83 | 11533 | 0 | 9432 | 0 | 0 | exact_code_content_signature | 0.6804 |
| astropy__astropy-13398 | stock_sglang_prefix_only | 2366.97 | 13240 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13398 | kvflow_style_prefix_baseline | 833.67 | 13240 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | kvflow_style_prefix_plus_hints | 1535.99 | 13243 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | agenttemplatekv_exact_reuse | 1954.32 | 13241 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 0.0702 |
| astropy__astropy-13453 | stock_sglang_prefix_only | 2149.06 | 12492 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_style_prefix_baseline | 2155.76 | 12492 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_style_prefix_plus_hints | 2343.42 | 12495 | 0 | 0 | 0 | 0 |  | 0.7222 |
| astropy__astropy-13453 | agenttemplatekv_exact_reuse | 2558.34 | 12493 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 0.7222 |
| astropy__astropy-13579 | stock_sglang_prefix_only | 2290.49 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_baseline | 2303.14 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_plus_hints | 2317.94 | 11811 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | agenttemplatekv_exact_reuse | 2386.0 | 11809 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13977 | stock_sglang_prefix_only | 1209.09 | 12895 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13977 | kvflow_style_prefix_baseline | 1210.57 | 12895 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13977 | kvflow_style_prefix_plus_hints | 1257.48 | 12898 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13977 | agenttemplatekv_exact_reuse | 1234.81 | 12896 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14096 | stock_sglang_prefix_only | 988.28 | 10393 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14096 | kvflow_style_prefix_baseline | 977.48 | 10393 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-14096 | kvflow_style_prefix_plus_hints | 970.25 | 10396 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14096 | agenttemplatekv_exact_reuse | 982.11 | 10394 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-14182 | stock_sglang_prefix_only | 1075.22 | 10766 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14182 | kvflow_style_prefix_baseline | 1149.78 | 10766 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14182 | kvflow_style_prefix_plus_hints | 1080.31 | 10769 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14182 | agenttemplatekv_exact_reuse | 1088.14 | 10767 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14309 | stock_sglang_prefix_only | 1855.03 | 11476 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14309 | kvflow_style_prefix_baseline | 1620.08 | 11476 | 0 | 0 | 0 | 0 |  | 0.64 |
| astropy__astropy-14309 | kvflow_style_prefix_plus_hints | 1880.97 | 11479 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14309 | agenttemplatekv_exact_reuse | 1844.33 | 11477 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14365 | stock_sglang_prefix_only | 1450.26 | 10755 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14365 | kvflow_style_prefix_baseline | 1447.96 | 10755 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14365 | kvflow_style_prefix_plus_hints | 1389.11 | 10758 | 0 | 0 | 0 | 0 |  | 0.8571 |
| astropy__astropy-14365 | agenttemplatekv_exact_reuse | 1464.98 | 10756 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.9412 |
| astropy__astropy-14369 | stock_sglang_prefix_only | 1174.64 | 11146 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | kvflow_style_prefix_baseline | 1168.23 | 11146 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | kvflow_style_prefix_plus_hints | 1153.19 | 11149 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | agenttemplatekv_exact_reuse | 1183.98 | 11147 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14508 | stock_sglang_prefix_only | 1540.89 | 11088 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14508 | kvflow_style_prefix_baseline | 1474.81 | 11088 | 0 | 0 | 0 | 0 |  | 0.7391 |
| astropy__astropy-14508 | kvflow_style_prefix_plus_hints | 1461.24 | 11091 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14508 | agenttemplatekv_exact_reuse | 1503.9 | 11089 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.7391 |
| astropy__astropy-14539 | stock_sglang_prefix_only | 2314.05 | 11658 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14539 | kvflow_style_prefix_baseline | 2381.4 | 11658 | 0 | 0 | 0 | 0 |  | 0.2909 |
| astropy__astropy-14539 | kvflow_style_prefix_plus_hints | 2338.21 | 11661 | 0 | 0 | 0 | 0 |  | 0.2909 |
| astropy__astropy-14539 | agenttemplatekv_exact_reuse | 2353.62 | 11659 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.2909 |
| astropy__astropy-14598 | stock_sglang_prefix_only | 1195.37 | 10966 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14598 | kvflow_style_prefix_baseline | 1237.78 | 10966 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14598 | kvflow_style_prefix_plus_hints | 1244.85 | 10969 | 0 | 0 | 0 | 0 |  | 0.9444 |
| astropy__astropy-14598 | agenttemplatekv_exact_reuse | 1228.06 | 10967 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.7179 |
| django__django-10097 | stock_sglang_prefix_only | 2135.93 | 17295 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | kvflow_style_prefix_baseline | 1522.97 | 17295 | 0 | 0 | 0 | 0 |  | 0.5067 |
| django__django-10097 | kvflow_style_prefix_plus_hints | 1885.62 | 17298 | 0 | 0 | 0 | 0 |  | 0.8132 |
| django__django-10097 | agenttemplatekv_exact_reuse | 1906.16 | 17296 | 0 | 8999 | 0 | 0 | exact_code_content_signature | 0.8132 |
| django__django-10554 | stock_sglang_prefix_only | 1355.32 | 10700 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10554 | kvflow_style_prefix_baseline | 1354.02 | 10700 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10554 | kvflow_style_prefix_plus_hints | 1337.92 | 10703 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10554 | agenttemplatekv_exact_reuse | 1357.77 | 10701 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-10880 | stock_sglang_prefix_only | 1111.1 | 9661 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10880 | kvflow_style_prefix_baseline | 1645.23 | 9661 | 0 | 0 | 0 | 0 |  | 0.6842 |
| django__django-10880 | kvflow_style_prefix_plus_hints | 1279.51 | 9664 | 0 | 0 | 0 | 0 |  | 0.7429 |
| django__django-10880 | agenttemplatekv_exact_reuse | 1030.99 | 9662 | 0 | 9010 | 0 | 0 | exact_code_content_signature | 0.7879 |
| django__django-10914 | stock_sglang_prefix_only | 1058.45 | 9720 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10914 | kvflow_style_prefix_baseline | 1102.39 | 9720 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10914 | kvflow_style_prefix_plus_hints | 1043.57 | 9723 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10914 | agenttemplatekv_exact_reuse | 1201.54 | 9721 | 0 | 9017 | 0 | 0 | exact_code_content_signature | 0.0769 |
| django__django-10973 | stock_sglang_prefix_only | 2254.06 | 10568 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10973 | kvflow_style_prefix_baseline | 2215.09 | 10568 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10973 | kvflow_style_prefix_plus_hints | 2370.6 | 10571 | 0 | 0 | 0 | 0 |  | 0.6479 |
| django__django-10973 | agenttemplatekv_exact_reuse | 2312.59 | 10569 | 0 | 9017 | 0 | 0 | exact_code_content_signature | 0.6571 |
| django__django-10999 | stock_sglang_prefix_only | 2306.13 | 9895 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10999 | kvflow_style_prefix_baseline | 2373.36 | 9895 | 0 | 0 | 0 | 0 |  | 0.1429 |
| django__django-10999 | kvflow_style_prefix_plus_hints | 1568.68 | 9898 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-10999 | agenttemplatekv_exact_reuse | 2314.67 | 9896 | 0 | 9017 | 0 | 0 | exact_code_content_signature | 0.087 |
| django__django-11066 | stock_sglang_prefix_only | 1263.94 | 10182 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11066 | kvflow_style_prefix_baseline | 855.48 | 10182 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-11066 | kvflow_style_prefix_plus_hints | 1264.56 | 10185 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11066 | agenttemplatekv_exact_reuse | 1336.83 | 10183 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-11087 | stock_sglang_prefix_only | 1134.68 | 12554 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11087 | kvflow_style_prefix_baseline | 1336.31 | 12554 | 0 | 0 | 0 | 0 |  | 0.7647 |
| django__django-11087 | kvflow_style_prefix_plus_hints | 1400.74 | 12557 | 0 | 0 | 0 | 0 |  | 0.7222 |
| django__django-11087 | agenttemplatekv_exact_reuse | 1234.95 | 12555 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.6667 |
| django__django-11095 | stock_sglang_prefix_only | 1345.5 | 9734 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11095 | kvflow_style_prefix_baseline | 1339.28 | 9734 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11095 | kvflow_style_prefix_plus_hints | 1539.85 | 9737 | 0 | 0 | 0 | 0 |  | 0.05 |
| django__django-11095 | agenttemplatekv_exact_reuse | 1513.54 | 9735 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.05 |
| django__django-11099 | stock_sglang_prefix_only | 861.43 | 9833 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11099 | kvflow_style_prefix_baseline | 906.85 | 9833 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11099 | kvflow_style_prefix_plus_hints | 1212.24 | 9836 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-11099 | agenttemplatekv_exact_reuse | 884.82 | 9834 | 0 | 9004 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-11119 | stock_sglang_prefix_only | 961.81 | 9515 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11119 | kvflow_style_prefix_baseline | 1066.97 | 9515 | 0 | 0 | 0 | 0 |  | 0.8571 |
| django__django-11119 | kvflow_style_prefix_plus_hints | 1486.8 | 9518 | 0 | 0 | 0 | 0 |  | 0.5714 |
| django__django-11119 | agenttemplatekv_exact_reuse | 1172.39 | 9516 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.6087 |
| django__django-11133 | stock_sglang_prefix_only | 930.82 | 9629 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11133 | kvflow_style_prefix_baseline | 801.57 | 9629 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-11133 | kvflow_style_prefix_plus_hints | 1482.88 | 9632 | 0 | 0 | 0 | 0 |  | 0.5789 |
| django__django-11133 | agenttemplatekv_exact_reuse | 1517.32 | 9630 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.5789 |
| django__django-11138 | stock_sglang_prefix_only | 1630.94 | 11193 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11138 | kvflow_style_prefix_baseline | 1589.87 | 11193 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11138 | kvflow_style_prefix_plus_hints | 1573.66 | 11196 | 0 | 0 | 0 | 0 |  | 0.8947 |
| django__django-11138 | agenttemplatekv_exact_reuse | 1632.87 | 11194 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 0.8947 |
| django__django-11141 | stock_sglang_prefix_only | 985.08 | 9818 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11141 | kvflow_style_prefix_baseline | 680.62 | 9818 | 0 | 0 | 0 | 0 |  | 0.0 |
| django__django-11141 | kvflow_style_prefix_plus_hints | 2271.7 | 9821 | 0 | 0 | 0 | 0 |  | 0.0435 |
| django__django-11141 | agenttemplatekv_exact_reuse | 2274.81 | 9819 | 0 | 8914 | 0 | 0 | exact_code_content_signature | 0.0645 |
| django__django-11149 | stock_sglang_prefix_only | 1357.7 | 10734 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11149 | kvflow_style_prefix_baseline | 1370.49 | 10734 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11149 | kvflow_style_prefix_plus_hints | 1356.54 | 10737 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-11149 | agenttemplatekv_exact_reuse | 1380.45 | 10735 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-13989 | stock_sglang_prefix_only | 2426.28 | 10697 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_baseline | 1607.32 | 10697 | 0 | 0 | 0 | 0 |  | 0.1127 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_plus_hints | 2355.7 | 10700 | 0 | 0 | 0 | 0 |  | 0.2927 |
| matplotlib__matplotlib-13989 | agenttemplatekv_exact_reuse | 2336.37 | 10698 | 0 | 9710 | 0 | 0 | exact_code_content_signature | 0.2927 |
| matplotlib__matplotlib-14623 | stock_sglang_prefix_only | 1400.79 | 10528 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-14623 | kvflow_style_prefix_baseline | 922.12 | 10528 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-14623 | kvflow_style_prefix_plus_hints | 958.36 | 10531 | 0 | 0 | 0 | 0 |  | 0.7778 |
| matplotlib__matplotlib-14623 | agenttemplatekv_exact_reuse | 968.09 | 10529 | 0 | 9647 | 0 | 0 | exact_code_content_signature | 0.7778 |
| matplotlib__matplotlib-20488 | stock_sglang_prefix_only | 1062.04 | 11128 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20488 | kvflow_style_prefix_baseline | 843.41 | 11128 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-20488 | kvflow_style_prefix_plus_hints | 1469.23 | 11131 | 0 | 0 | 0 | 0 |  | 0.5 |
| matplotlib__matplotlib-20488 | agenttemplatekv_exact_reuse | 1413.9 | 11129 | 0 | 9778 | 0 | 0 | exact_code_content_signature | 0.5 |
| matplotlib__matplotlib-20676 | stock_sglang_prefix_only | 1652.77 | 10977 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20676 | kvflow_style_prefix_baseline | 2149.24 | 10977 | 0 | 0 | 0 | 0 |  | 0.087 |
| matplotlib__matplotlib-20676 | kvflow_style_prefix_plus_hints | 1418.5 | 10980 | 0 | 0 | 0 | 0 |  | 0.0889 |
| matplotlib__matplotlib-20676 | agenttemplatekv_exact_reuse | 1229.48 | 10978 | 0 | 9787 | 0 | 0 | exact_code_content_signature | 0.0976 |
| matplotlib__matplotlib-20826 | stock_sglang_prefix_only | 654.17 | 11134 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20826 | kvflow_style_prefix_baseline | 651.78 | 11134 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20826 | kvflow_style_prefix_plus_hints | 1016.98 | 11137 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-20826 | agenttemplatekv_exact_reuse | 1025.12 | 11135 | 0 | 9787 | 0 | 0 | exact_code_content_signature | 0.0 |
| matplotlib__matplotlib-20859 | stock_sglang_prefix_only | 930.13 | 10809 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20859 | kvflow_style_prefix_baseline | 918.07 | 10809 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-20859 | kvflow_style_prefix_plus_hints | 1315.36 | 10812 | 0 | 0 | 0 | 0 |  | 0.5417 |
| matplotlib__matplotlib-20859 | agenttemplatekv_exact_reuse | 1306.23 | 10810 | 0 | 9589 | 0 | 0 | exact_code_content_signature | 0.5833 |
| matplotlib__matplotlib-21568 | stock_sglang_prefix_only | 1195.64 | 11430 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-21568 | kvflow_style_prefix_baseline | 1116.22 | 11430 | 0 | 0 | 0 | 0 |  | 0.8 |
| matplotlib__matplotlib-21568 | kvflow_style_prefix_plus_hints | 1119.39 | 11433 | 0 | 0 | 0 | 0 |  | 0.8 |
| matplotlib__matplotlib-21568 | agenttemplatekv_exact_reuse | 1227.98 | 11431 | 0 | 9576 | 0 | 0 | exact_code_content_signature | 0.8571 |
| matplotlib__matplotlib-22719 | stock_sglang_prefix_only | 1257.9 | 12837 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22719 | kvflow_style_prefix_baseline | 1559.83 | 12837 | 0 | 0 | 0 | 0 |  | 0.5185 |
| matplotlib__matplotlib-22719 | kvflow_style_prefix_plus_hints | 1541.09 | 12840 | 0 | 0 | 0 | 0 |  | 0.5185 |
| matplotlib__matplotlib-22719 | agenttemplatekv_exact_reuse | 1572.26 | 12838 | 0 | 9581 | 0 | 0 | exact_code_content_signature | 0.6545 |
| matplotlib__matplotlib-22865 | stock_sglang_prefix_only | 1546.68 | 10951 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22865 | kvflow_style_prefix_baseline | 1493.93 | 10951 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22865 | kvflow_style_prefix_plus_hints | 1506.89 | 10954 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-22865 | agenttemplatekv_exact_reuse | 2118.69 | 10952 | 0 | 9612 | 0 | 0 | exact_code_content_signature | 0.0976 |
| matplotlib__matplotlib-22871 | stock_sglang_prefix_only | 1354.31 | 10573 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22871 | kvflow_style_prefix_baseline | 1352.6 | 10573 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22871 | kvflow_style_prefix_plus_hints | 1420.04 | 10576 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-22871 | agenttemplatekv_exact_reuse | 1877.78 | 10574 | 0 | 9612 | 0 | 0 | exact_code_content_signature | 0.0 |
| matplotlib__matplotlib-23299 | stock_sglang_prefix_only | 645.0 | 10660 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23299 | kvflow_style_prefix_baseline | 767.62 | 10660 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-23299 | kvflow_style_prefix_plus_hints | 653.07 | 10663 | 0 | 0 | 0 | 0 |  | 0.0 |
| matplotlib__matplotlib-23299 | agenttemplatekv_exact_reuse | 641.14 | 10661 | 0 | 9622 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-23314 | stock_sglang_prefix_only | 1356.7 | 10579 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23314 | kvflow_style_prefix_baseline | 1353.38 | 10579 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23314 | kvflow_style_prefix_plus_hints | 1353.26 | 10582 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23314 | agenttemplatekv_exact_reuse | 1364.71 | 10580 | 0 | 9622 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-23412 | stock_sglang_prefix_only | 1442.8 | 11172 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23412 | kvflow_style_prefix_baseline | 1689.04 | 11172 | 0 | 0 | 0 | 0 |  | 0.2 |
| matplotlib__matplotlib-23412 | kvflow_style_prefix_plus_hints | 1704.42 | 11175 | 0 | 0 | 0 | 0 |  | 0.2667 |
| matplotlib__matplotlib-23412 | agenttemplatekv_exact_reuse | 1711.86 | 11173 | 0 | 9622 | 0 | 0 | exact_code_content_signature | 0.2667 |
| matplotlib__matplotlib-23476 | stock_sglang_prefix_only | 1136.42 | 11435 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23476 | kvflow_style_prefix_baseline | 1131.73 | 11435 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23476 | kvflow_style_prefix_plus_hints | 1135.37 | 11438 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-23476 | agenttemplatekv_exact_reuse | 1160.04 | 11436 | 0 | 9622 | 0 | 0 | exact_code_content_signature | 0.1765 |
| matplotlib__matplotlib-24026 | stock_sglang_prefix_only | 1084.2 | 10866 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-24026 | kvflow_style_prefix_baseline | 1065.85 | 10866 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-24026 | kvflow_style_prefix_plus_hints | 1085.92 | 10869 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-24026 | agenttemplatekv_exact_reuse | 1086.52 | 10867 | 0 | 9627 | 0 | 0 | exact_code_content_signature | 1.0 |
| mwaskom__seaborn-3069 | stock_sglang_prefix_only | 2283.76 | 10253 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3069 | kvflow_style_prefix_baseline | 845.12 | 10253 | 0 | 0 | 0 | 0 |  | 0.1538 |
| mwaskom__seaborn-3069 | kvflow_style_prefix_plus_hints | 2281.86 | 10256 | 0 | 0 | 0 | 0 |  | 0.16 |
| mwaskom__seaborn-3069 | agenttemplatekv_exact_reuse | 2344.74 | 10254 | 0 | 9281 | 0 | 0 | exact_code_content_signature | 0.0988 |
| mwaskom__seaborn-3187 | stock_sglang_prefix_only | 1013.13 | 10422 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3187 | kvflow_style_prefix_baseline | 1042.43 | 10422 | 0 | 0 | 0 | 0 |  | 1.0 |
| mwaskom__seaborn-3187 | kvflow_style_prefix_plus_hints | 2261.84 | 10425 | 0 | 0 | 0 | 0 |  | 0.4286 |
| mwaskom__seaborn-3187 | agenttemplatekv_exact_reuse | 1887.57 | 10423 | 0 | 9282 | 0 | 0 | exact_code_content_signature | 0.4516 |
| pallets__flask-5014 | stock_sglang_prefix_only | 924.54 | 10368 | 0 | 0 | 0 | 0 |  | 1.0 |
| pallets__flask-5014 | kvflow_style_prefix_baseline | 874.5 | 10368 | 0 | 0 | 0 | 0 |  | 0.0 |
| pallets__flask-5014 | kvflow_style_prefix_plus_hints | 853.9 | 10371 | 0 | 0 | 0 | 0 |  | 0.0 |
| pallets__flask-5014 | agenttemplatekv_exact_reuse | 847.63 | 10369 | 0 | 9854 | 0 | 0 | exact_code_content_signature | 0.0 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
