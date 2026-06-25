# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 20
- Git commit: `5fb934751`
- Command: `/home/gfy/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --model /home/gfy/models/Qwen2.5-7B-Instruct --dataset results/repo_level_datasets/swe_verified_100_instances.json --manifest results/repo_level_datasets/manifest_100.json --start-index 68 --max-cases 20 --max-tokens 128 --baseline-profile lmcache --server-extra-args --disable-overlap-schedule --max-running-requests 1 --flush-cache-per-case --port 31352 --server-timeout 600 --eval-timeout 3600 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard68_20_flush_noidlecheck`
- Flush cache per case: `True`
- Concurrent clients: `1`
- Baseline profile: `lmcache`
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Server extra args: `--disable-overlap-schedule --max-running-requests 1`
- Resolved server extra args: `--disable-overlap-schedule --max-running-requests 1 --enable-lmcache`
- LMCache config: `/home/gfy/CodeMAS_Project/sglang-kvflow/python/sglang/srt/mem_cache/storage/lmcache/example_config.yaml`
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | p50 | p90 | p99 | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 20 | 1217.3 | 1070.5 | 1821.0 | 2449.6 | 12193.1 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 20 | 1287.8 | 1190.9 | 1770.3 | 2469.5 | 12193.1 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.8907 |
| kvflow_style_prefix_plus_hints | 20 | 1339.3 | 1197.3 | 2252.7 | 2457.2 | 12196.1 | 2.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.6394 |
| agenttemplatekv_exact_reuse | 20 | 1347.6 | 1205.5 | 2277.2 | 2465.9 | 12194.1 | 2.0 | 0.0 | 0.00 | 10306.1 | 0.0 | 0.0 | 1.00 | 0.6339 |

## Figures

Figure generation skipped because a clean `matplotlib` import is unavailable in this environment.

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| pydata__xarray-4966 | stock_sglang_prefix_only | 1614.27 | 12646 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4966 | kvflow_style_prefix_baseline | 1612.93 | 12646 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4966 | kvflow_style_prefix_plus_hints | 1274.12 | 12649 | 0 | 0 | 0 | 0 |  | 0.1017 |
| pydata__xarray-4966 | agenttemplatekv_exact_reuse | 1266.08 | 12647 | 0 | 10569 | 0 | 0 | exact_code_content_signature | 0.1017 |
| pydata__xarray-6461 | stock_sglang_prefix_only | 1064.16 | 11435 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6461 | kvflow_style_prefix_baseline | 1037.74 | 11435 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6461 | kvflow_style_prefix_plus_hints | 1134.7 | 11438 | 0 | 0 | 0 | 0 |  | 0.9444 |
| pydata__xarray-6461 | agenttemplatekv_exact_reuse | 1147.93 | 11436 | 0 | 10634 | 0 | 0 | exact_code_content_signature | 0.9444 |
| pydata__xarray-6599 | stock_sglang_prefix_only | 674.89 | 13300 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6599 | kvflow_style_prefix_baseline | 689.64 | 13300 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6599 | kvflow_style_prefix_plus_hints | 679.26 | 13303 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-6599 | agenttemplatekv_exact_reuse | 642.12 | 13301 | 0 | 10636 | 0 | 0 | exact_code_content_signature | 1.0 |
| pylint-dev__pylint-4551 | stock_sglang_prefix_only | 2452.77 | 12914 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4551 | kvflow_style_prefix_baseline | 2425.43 | 12914 | 0 | 0 | 0 | 0 |  | 0.8958 |
| pylint-dev__pylint-4551 | kvflow_style_prefix_plus_hints | 2429.88 | 12917 | 0 | 0 | 0 | 0 |  | 0.5581 |
| pylint-dev__pylint-4551 | agenttemplatekv_exact_reuse | 2469.01 | 12915 | 0 | 11201 | 0 | 0 | exact_code_content_signature | 0.5581 |
| pylint-dev__pylint-4604 | stock_sglang_prefix_only | 666.45 | 12517 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4604 | kvflow_style_prefix_baseline | 644.7 | 12517 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4604 | kvflow_style_prefix_plus_hints | 1127.18 | 12520 | 0 | 0 | 0 | 0 |  | 0.0769 |
| pylint-dev__pylint-4604 | agenttemplatekv_exact_reuse | 1122.49 | 12518 | 0 | 11201 | 0 | 0 | exact_code_content_signature | 0.0769 |
| pylint-dev__pylint-4661 | stock_sglang_prefix_only | 836.94 | 11971 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4661 | kvflow_style_prefix_baseline | 1349.15 | 11971 | 0 | 0 | 0 | 0 |  | 0.0 |
| pylint-dev__pylint-4661 | kvflow_style_prefix_plus_hints | 1288.6 | 11974 | 0 | 0 | 0 | 0 |  | 0.0 |
| pylint-dev__pylint-4661 | agenttemplatekv_exact_reuse | 1277.05 | 11972 | 0 | 11222 | 0 | 0 | exact_code_content_signature | 0.0 |
| pylint-dev__pylint-4970 | stock_sglang_prefix_only | 1448.86 | 11884 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-4970 | kvflow_style_prefix_baseline | 1247.46 | 11884 | 0 | 0 | 0 | 0 |  | 0.7143 |
| pylint-dev__pylint-4970 | kvflow_style_prefix_plus_hints | 1341.38 | 11887 | 0 | 0 | 0 | 0 |  | 0.05 |
| pylint-dev__pylint-4970 | agenttemplatekv_exact_reuse | 1339.78 | 11885 | 0 | 11199 | 0 | 0 | exact_code_content_signature | 0.05 |
| pylint-dev__pylint-6386 | stock_sglang_prefix_only | 933.22 | 10593 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6386 | kvflow_style_prefix_baseline | 1672.02 | 10593 | 0 | 0 | 0 | 0 |  | 0.3396 |
| pylint-dev__pylint-6386 | kvflow_style_prefix_plus_hints | 1198.4 | 10596 | 0 | 0 | 0 | 0 |  | 0.0 |
| pylint-dev__pylint-6386 | agenttemplatekv_exact_reuse | 1179.49 | 10594 | 0 | 9831 | 0 | 0 | exact_code_content_signature | 0.0 |
| pylint-dev__pylint-6528 | stock_sglang_prefix_only | 2436.21 | 12421 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6528 | kvflow_style_prefix_baseline | 2479.83 | 12421 | 0 | 0 | 0 | 0 |  | 0.8649 |
| pylint-dev__pylint-6528 | kvflow_style_prefix_plus_hints | 2232.97 | 12424 | 0 | 0 | 0 | 0 |  | 0.7297 |
| pylint-dev__pylint-6528 | agenttemplatekv_exact_reuse | 2257.69 | 12422 | 0 | 10020 | 0 | 0 | exact_code_content_signature | 0.7297 |
| pylint-dev__pylint-6903 | stock_sglang_prefix_only | 1123.51 | 11862 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6903 | kvflow_style_prefix_baseline | 1081.81 | 11862 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6903 | kvflow_style_prefix_plus_hints | 1093.17 | 11865 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-6903 | agenttemplatekv_exact_reuse | 1081.01 | 11863 | 0 | 10022 | 0 | 0 | exact_code_content_signature | 1.0 |
| pylint-dev__pylint-7080 | stock_sglang_prefix_only | 645.87 | 18382 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7080 | kvflow_style_prefix_baseline | 703.78 | 18382 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7080 | kvflow_style_prefix_plus_hints | 657.08 | 18385 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7080 | agenttemplatekv_exact_reuse | 662.9 | 18383 | 0 | 10033 | 0 | 0 | exact_code_content_signature | 1.0 |
| pylint-dev__pylint-7277 | stock_sglang_prefix_only | 988.39 | 10935 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7277 | kvflow_style_prefix_baseline | 983.11 | 10935 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-7277 | kvflow_style_prefix_plus_hints | 1678.3 | 10938 | 0 | 0 | 0 | 0 |  | 0.0541 |
| pylint-dev__pylint-7277 | agenttemplatekv_exact_reuse | 1699.76 | 10936 | 0 | 10056 | 0 | 0 | exact_code_content_signature | 0.0541 |
| pylint-dev__pylint-8898 | stock_sglang_prefix_only | 1076.86 | 11924 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-8898 | kvflow_style_prefix_baseline | 1076.63 | 11924 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-8898 | kvflow_style_prefix_plus_hints | 1063.25 | 11927 | 0 | 0 | 0 | 0 |  | 1.0 |
| pylint-dev__pylint-8898 | agenttemplatekv_exact_reuse | 1142.55 | 11925 | 0 | 9966 | 0 | 0 | exact_code_content_signature | 0.9375 |
| pytest-dev__pytest-10051 | stock_sglang_prefix_only | 974.52 | 11091 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_style_prefix_baseline | 1006.77 | 11091 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | kvflow_style_prefix_plus_hints | 1001.49 | 11094 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10051 | agenttemplatekv_exact_reuse | 1007.41 | 11092 | 0 | 9926 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-10081 | stock_sglang_prefix_only | 1192.84 | 11695 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | kvflow_style_prefix_baseline | 1192.58 | 11695 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | kvflow_style_prefix_plus_hints | 1196.12 | 11698 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10081 | agenttemplatekv_exact_reuse | 1261.24 | 11696 | 0 | 9926 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-10356 | stock_sglang_prefix_only | 1752.63 | 11721 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10356 | kvflow_style_prefix_baseline | 1697.54 | 11721 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-10356 | kvflow_style_prefix_plus_hints | 1794.28 | 11724 | 0 | 0 | 0 | 0 |  | 0.9836 |
| pytest-dev__pytest-10356 | agenttemplatekv_exact_reuse | 1801.87 | 11722 | 0 | 9926 | 0 | 0 | exact_code_content_signature | 0.9836 |
| pytest-dev__pytest-5262 | stock_sglang_prefix_only | 944.31 | 11631 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5262 | kvflow_style_prefix_baseline | 1450.59 | 11631 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5262 | kvflow_style_prefix_plus_hints | 1015.32 | 11634 | 0 | 0 | 0 | 0 |  | 0.9032 |
| pytest-dev__pytest-5262 | agenttemplatekv_exact_reuse | 1011.71 | 11632 | 0 | 9958 | 0 | 0 | exact_code_content_signature | 0.9032 |
| pytest-dev__pytest-5631 | stock_sglang_prefix_only | 1293.09 | 11267 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5631 | kvflow_style_prefix_baseline | 1189.15 | 11267 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5631 | kvflow_style_prefix_plus_hints | 1199.93 | 11270 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5631 | agenttemplatekv_exact_reuse | 1231.48 | 11268 | 0 | 9912 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-5787 | stock_sglang_prefix_only | 1301.83 | 12939 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5787 | kvflow_style_prefix_baseline | 1272.98 | 12939 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5787 | kvflow_style_prefix_plus_hints | 2463.57 | 12942 | 0 | 0 | 0 | 0 |  | 0.3871 |
| pytest-dev__pytest-5787 | agenttemplatekv_exact_reuse | 2452.52 | 12940 | 0 | 9943 | 0 | 0 | exact_code_content_signature | 0.339 |
| pytest-dev__pytest-5809 | stock_sglang_prefix_only | 923.59 | 10734 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5809 | kvflow_style_prefix_baseline | 941.38 | 10734 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5809 | kvflow_style_prefix_plus_hints | 917.39 | 10737 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5809 | agenttemplatekv_exact_reuse | 897.44 | 10735 | 0 | 9941 | 0 | 0 | exact_code_content_signature | 1.0 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
