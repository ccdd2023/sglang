# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 20
- Git commit: `3d709f3ce`
- Command: `benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --model /home/gfy/models/Qwen2.5-7B-Instruct --dataset results/repo_level_datasets/swe_verified_100_instances.json --manifest results/repo_level_datasets/manifest_100.json --max-cases 20 --concurrent-clients 4 --disable-hierarchical-cache --port 31331 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_20_concurrent4_smoke`
- Flush cache per case: `False`
- Concurrent clients: `4`
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | p50 | p90 | p99 | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 20 | 2176.2 | 1811.4 | 4082.0 | 4788.2 | 11471.6 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 20 | 1850.4 | 1611.5 | 2607.5 | 3320.1 | 11471.6 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.7910 |
| kvflow_style_prefix_plus_hints | 20 | 2064.1 | 1891.7 | 3247.4 | 4136.1 | 11474.6 | 2.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.7019 |
| agenttemplatekv_exact_reuse | 20 | 2292.5 | 1933.2 | 4346.9 | 4680.1 | 11472.6 | 2.0 | 0.0 | 0.00 | 9346.2 | 0.0 | 0.0 | 1.00 | 0.7197 |

## Figures

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_20_concurrent4_smoke/fig_latency.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_20_concurrent4_smoke/fig_cached_tokens.png)

![Prefetch tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_20_concurrent4_smoke/fig_prefetch_tokens.png)

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 854.86 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 1054.47 | 10547 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2089.86 | 10550 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 1948.76 | 10548 | 0 | 9418 | 0 | 0 | exact_code_content_signature | 0.0 |
| astropy__astropy-13033 | stock_sglang_prefix_only | 1762.08 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_baseline | 1761.56 | 10727 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | kvflow_style_prefix_plus_hints | 1764.84 | 10730 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13033 | agenttemplatekv_exact_reuse | 2080.74 | 10728 | 0 | 9424 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13236 | stock_sglang_prefix_only | 2645.4 | 11532 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13236 | kvflow_style_prefix_baseline | 2602.39 | 11532 | 0 | 0 | 0 | 0 |  | 0.8099 |
| astropy__astropy-13236 | kvflow_style_prefix_plus_hints | 3199.46 | 11535 | 0 | 0 | 0 | 0 |  | 0.7899 |
| astropy__astropy-13236 | agenttemplatekv_exact_reuse | 4680.1 | 11533 | 0 | 9432 | 0 | 0 | exact_code_content_signature | 0.9032 |
| astropy__astropy-13398 | stock_sglang_prefix_only | 2645.55 | 13240 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13398 | kvflow_style_prefix_baseline | 1297.37 | 13240 | 0 | 0 | 0 | 0 |  | 0.0833 |
| astropy__astropy-13398 | kvflow_style_prefix_plus_hints | 1932.48 | 13243 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-13398 | agenttemplatekv_exact_reuse | 2743.27 | 13241 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 0.0702 |
| astropy__astropy-13453 | stock_sglang_prefix_only | 4902.91 | 12492 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13453 | kvflow_style_prefix_baseline | 2497.78 | 12492 | 0 | 0 | 0 | 0 |  | 0.8533 |
| astropy__astropy-13453 | kvflow_style_prefix_plus_hints | 2506.88 | 12495 | 0 | 0 | 0 | 0 |  | 0.72 |
| astropy__astropy-13453 | agenttemplatekv_exact_reuse | 4309.88 | 12493 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13579 | stock_sglang_prefix_only | 4299.07 | 11808 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | kvflow_style_prefix_baseline | 2604.15 | 11808 | 0 | 0 | 0 | 0 |  | 0.8462 |
| astropy__astropy-13579 | kvflow_style_prefix_plus_hints | 2988.67 | 11811 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13579 | agenttemplatekv_exact_reuse | 4050.2 | 11809 | 0 | 9430 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-13977 | stock_sglang_prefix_only | 1350.82 | 12895 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13977 | kvflow_style_prefix_baseline | 1433.66 | 12895 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-13977 | kvflow_style_prefix_plus_hints | 1280.58 | 12898 | 0 | 0 | 0 | 0 |  | 0.9655 |
| astropy__astropy-13977 | agenttemplatekv_exact_reuse | 1289.19 | 12896 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 0.9655 |
| astropy__astropy-14096 | stock_sglang_prefix_only | 1040.21 | 10393 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14096 | kvflow_style_prefix_baseline | 1044.17 | 10393 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14096 | kvflow_style_prefix_plus_hints | 1023.89 | 10396 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14096 | agenttemplatekv_exact_reuse | 1152.16 | 10394 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14182 | stock_sglang_prefix_only | 1138.64 | 10766 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14182 | kvflow_style_prefix_baseline | 1550.5 | 10766 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14182 | kvflow_style_prefix_plus_hints | 1331.72 | 10769 | 0 | 0 | 0 | 0 |  | 0.1 |
| astropy__astropy-14182 | agenttemplatekv_exact_reuse | 1202.46 | 10767 | 0 | 9479 | 0 | 0 | exact_code_content_signature | 0.1 |
| astropy__astropy-14309 | stock_sglang_prefix_only | 2191.97 | 11476 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14309 | kvflow_style_prefix_baseline | 1982.84 | 11476 | 0 | 0 | 0 | 0 |  | 0.64 |
| astropy__astropy-14309 | kvflow_style_prefix_plus_hints | 1851.01 | 11479 | 0 | 0 | 0 | 0 |  | 0.9333 |
| astropy__astropy-14309 | agenttemplatekv_exact_reuse | 1917.54 | 11477 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14365 | stock_sglang_prefix_only | 1860.69 | 10755 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14365 | kvflow_style_prefix_baseline | 1499.69 | 10755 | 0 | 0 | 0 | 0 |  | 0.8571 |
| astropy__astropy-14365 | kvflow_style_prefix_plus_hints | 1448.6 | 10758 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14365 | agenttemplatekv_exact_reuse | 1491.36 | 10756 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.875 |
| astropy__astropy-14369 | stock_sglang_prefix_only | 1213.88 | 11146 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | kvflow_style_prefix_baseline | 1199.4 | 11146 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | kvflow_style_prefix_plus_hints | 1229.38 | 11149 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14369 | agenttemplatekv_exact_reuse | 1203.15 | 11147 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 1.0 |
| astropy__astropy-14508 | stock_sglang_prefix_only | 1583.96 | 11088 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14508 | kvflow_style_prefix_baseline | 1522.63 | 11088 | 0 | 0 | 0 | 0 |  | 0.8 |
| astropy__astropy-14508 | kvflow_style_prefix_plus_hints | 1978.21 | 11091 | 0 | 0 | 0 | 0 |  | 0.8 |
| astropy__astropy-14508 | agenttemplatekv_exact_reuse | 4220.47 | 11089 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.8 |
| astropy__astropy-14539 | stock_sglang_prefix_only | 3820.65 | 11658 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14539 | kvflow_style_prefix_baseline | 2637.31 | 11658 | 0 | 0 | 0 | 0 |  | 0.6667 |
| astropy__astropy-14539 | kvflow_style_prefix_plus_hints | 4243.34 | 11661 | 0 | 0 | 0 | 0 |  | 0.2 |
| astropy__astropy-14539 | agenttemplatekv_exact_reuse | 4680.16 | 11659 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.2 |
| astropy__astropy-14598 | stock_sglang_prefix_only | 2123.8 | 10966 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-14598 | kvflow_style_prefix_baseline | 2352.12 | 10966 | 0 | 0 | 0 | 0 |  | 0.9444 |
| astropy__astropy-14598 | kvflow_style_prefix_plus_hints | 1349.56 | 10969 | 0 | 0 | 0 | 0 |  | 0.9444 |
| astropy__astropy-14598 | agenttemplatekv_exact_reuse | 1338.44 | 10967 | 0 | 9490 | 0 | 0 | exact_code_content_signature | 0.6667 |
| django__django-10097 | stock_sglang_prefix_only | 4057.89 | 17295 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | kvflow_style_prefix_baseline | 1672.43 | 17295 | 0 | 0 | 0 | 0 |  | 0.5067 |
| django__django-10097 | kvflow_style_prefix_plus_hints | 3678.84 | 17298 | 0 | 0 | 0 | 0 |  | 0.8132 |
| django__django-10097 | agenttemplatekv_exact_reuse | 2100.82 | 17296 | 0 | 8999 | 0 | 0 | exact_code_content_signature | 0.8132 |
| django__django-10554 | stock_sglang_prefix_only | 1478.26 | 10700 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10554 | kvflow_style_prefix_baseline | 1500.48 | 10700 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10554 | kvflow_style_prefix_plus_hints | 3114.19 | 10703 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10554 | agenttemplatekv_exact_reuse | 1511.52 | 10701 | 0 | 8941 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-10880 | stock_sglang_prefix_only | 1062.73 | 9661 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10880 | kvflow_style_prefix_baseline | 3480.27 | 9661 | 0 | 0 | 0 | 0 |  | 0.8125 |
| django__django-10880 | kvflow_style_prefix_plus_hints | 1341.51 | 9664 | 0 | 0 | 0 | 0 |  | 0.8125 |
| django__django-10880 | agenttemplatekv_exact_reuse | 958.46 | 9662 | 0 | 9010 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-10914 | stock_sglang_prefix_only | 1080.38 | 9720 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10914 | kvflow_style_prefix_baseline | 1033.97 | 9720 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10914 | kvflow_style_prefix_plus_hints | 794.08 | 9723 | 0 | 0 | 0 | 0 |  | 0.3333 |
| django__django-10914 | agenttemplatekv_exact_reuse | 778.39 | 9721 | 0 | 9017 | 0 | 0 | exact_code_content_signature | 0.3333 |
| django__django-10973 | stock_sglang_prefix_only | 2410.83 | 10568 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10973 | kvflow_style_prefix_baseline | 2280.57 | 10568 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10973 | kvflow_style_prefix_plus_hints | 2134.66 | 10571 | 0 | 0 | 0 | 0 |  | 0.6269 |
| django__django-10973 | agenttemplatekv_exact_reuse | 2192.29 | 10569 | 0 | 9017 | 0 | 0 | exact_code_content_signature | 0.6667 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
