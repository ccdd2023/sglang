# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 20
- Git commit: `5fb934751`
- Command: `/home/gfy/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --model /home/gfy/models/Qwen2.5-7B-Instruct --dataset results/repo_level_datasets/swe_verified_100_instances.json --manifest results/repo_level_datasets/manifest_100.json --start-index 48 --max-cases 20 --max-tokens 128 --baseline-profile lmcache --server-extra-args --disable-overlap-schedule --max-running-requests 1 --flush-cache-per-case --port 31351 --server-timeout 600 --eval-timeout 3600 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard48_20_flush_noidlecheck`
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
| stock_sglang_prefix_only | 20 | 1347.8 | 1074.1 | 2424.3 | 2542.8 | 15816.5 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 20 | 1254.8 | 1063.9 | 1796.8 | 2538.4 | 15816.5 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.5866 |
| kvflow_style_prefix_plus_hints | 20 | 1114.7 | 1061.0 | 1455.0 | 2031.5 | 15819.5 | 2.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.6348 |
| agenttemplatekv_exact_reuse | 20 | 1244.5 | 1057.1 | 1724.4 | 2615.1 | 15817.5 | 2.0 | 0.0 | 0.00 | 14589.3 | 0.0 | 0.0 | 1.00 | 0.6545 |

## Figures

Figure generation skipped because a clean `matplotlib` import is unavailable in this environment.

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| psf__requests-1142 | stock_sglang_prefix_only | 909.48 | 25580 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_style_prefix_baseline | 1001.0 | 25580 | 0 | 0 | 0 | 0 |  | 0.25 |
| psf__requests-1142 | kvflow_style_prefix_plus_hints | 923.21 | 25583 | 0 | 0 | 0 | 0 |  | 0.125 |
| psf__requests-1142 | agenttemplatekv_exact_reuse | 1045.27 | 25581 | 0 | 24993 | 0 | 0 | exact_code_content_signature | 0.25 |
| psf__requests-1724 | stock_sglang_prefix_only | 1068.38 | 27884 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1724 | kvflow_style_prefix_baseline | 827.55 | 27884 | 0 | 0 | 0 | 0 |  | 0.4 |
| psf__requests-1724 | kvflow_style_prefix_plus_hints | 1127.06 | 27887 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1724 | agenttemplatekv_exact_reuse | 890.51 | 27885 | 0 | 25158 | 0 | 0 | exact_code_content_signature | 0.375 |
| psf__requests-1766 | stock_sglang_prefix_only | 701.42 | 25978 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1766 | kvflow_style_prefix_baseline | 699.61 | 25978 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1766 | kvflow_style_prefix_plus_hints | 704.06 | 25981 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1766 | agenttemplatekv_exact_reuse | 698.76 | 25979 | 0 | 25181 | 0 | 0 | exact_code_content_signature | 1.0 |
| psf__requests-1921 | stock_sglang_prefix_only | 760.12 | 26036 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1921 | kvflow_style_prefix_baseline | 874.88 | 26036 | 0 | 0 | 0 | 0 |  | 0.5 |
| psf__requests-1921 | kvflow_style_prefix_plus_hints | 766.66 | 26039 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1921 | agenttemplatekv_exact_reuse | 754.5 | 26037 | 0 | 25263 | 0 | 0 | exact_code_content_signature | 1.0 |
| psf__requests-2317 | stock_sglang_prefix_only | 2567.26 | 26119 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-2317 | kvflow_style_prefix_baseline | 2568.41 | 26119 | 0 | 0 | 0 | 0 |  | 0.1159 |
| psf__requests-2317 | kvflow_style_prefix_plus_hints | 731.55 | 26122 | 0 | 0 | 0 | 0 |  | 0.1951 |
| psf__requests-2317 | agenttemplatekv_exact_reuse | 2641.83 | 26120 | 0 | 25318 | 0 | 0 | exact_code_content_signature | 0.1159 |
| psf__requests-2931 | stock_sglang_prefix_only | 1028.61 | 25942 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-2931 | kvflow_style_prefix_baseline | 947.26 | 25942 | 0 | 0 | 0 | 0 |  | 0.7368 |
| psf__requests-2931 | kvflow_style_prefix_plus_hints | 1030.85 | 25945 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-2931 | agenttemplatekv_exact_reuse | 1039.88 | 25943 | 0 | 25364 | 0 | 0 | exact_code_content_signature | 0.7368 |
| psf__requests-5414 | stock_sglang_prefix_only | 1050.4 | 10849 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-5414 | kvflow_style_prefix_baseline | 1046.31 | 10849 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-5414 | kvflow_style_prefix_plus_hints | 1035.37 | 10852 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-5414 | agenttemplatekv_exact_reuse | 1058.14 | 10850 | 0 | 9911 | 0 | 0 | exact_code_content_signature | 1.0 |
| psf__requests-6028 | stock_sglang_prefix_only | 1661.76 | 10881 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-6028 | kvflow_style_prefix_baseline | 1665.82 | 10881 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-6028 | kvflow_style_prefix_plus_hints | 1556.32 | 10884 | 0 | 0 | 0 | 0 |  | 0.7667 |
| psf__requests-6028 | agenttemplatekv_exact_reuse | 1638.11 | 10882 | 0 | 9914 | 0 | 0 | exact_code_content_signature | 0.6102 |
| pydata__xarray-2905 | stock_sglang_prefix_only | 1304.53 | 11893 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_style_prefix_baseline | 1317.86 | 11893 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | kvflow_style_prefix_plus_hints | 1299.93 | 11896 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-2905 | agenttemplatekv_exact_reuse | 1317.0 | 11894 | 0 | 10569 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-3095 | stock_sglang_prefix_only | 1892.19 | 10634 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3095 | kvflow_style_prefix_baseline | 1170.2 | 10634 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-3095 | kvflow_style_prefix_plus_hints | 1175.76 | 10637 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-3095 | agenttemplatekv_exact_reuse | 1163.75 | 10635 | 0 | 9541 | 0 | 0 | exact_code_content_signature | 0.0 |
| pydata__xarray-3151 | stock_sglang_prefix_only | 786.67 | 11000 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3151 | kvflow_style_prefix_baseline | 749.63 | 11000 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3151 | kvflow_style_prefix_plus_hints | 763.85 | 11003 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3151 | agenttemplatekv_exact_reuse | 815.91 | 11001 | 0 | 9551 | 0 | 0 | exact_code_content_signature | 0.8333 |
| pydata__xarray-3305 | stock_sglang_prefix_only | 1039.11 | 10923 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3305 | kvflow_style_prefix_baseline | 1044.41 | 10923 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3305 | kvflow_style_prefix_plus_hints | 1051.79 | 10926 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3305 | agenttemplatekv_exact_reuse | 1048.88 | 10924 | 0 | 9665 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-3677 | stock_sglang_prefix_only | 874.72 | 10748 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3677 | kvflow_style_prefix_baseline | 1455.85 | 10748 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-3677 | kvflow_style_prefix_plus_hints | 1443.72 | 10751 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-3677 | agenttemplatekv_exact_reuse | 1513.78 | 10749 | 0 | 9762 | 0 | 0 | exact_code_content_signature | 0.0 |
| pydata__xarray-3993 | stock_sglang_prefix_only | 1101.87 | 11429 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-3993 | kvflow_style_prefix_baseline | 795.2 | 11429 | 0 | 0 | 0 | 0 |  | 0.2222 |
| pydata__xarray-3993 | kvflow_style_prefix_plus_hints | 989.16 | 11432 | 0 | 0 | 0 | 0 |  | 0.8667 |
| pydata__xarray-3993 | agenttemplatekv_exact_reuse | 1056.14 | 11430 | 0 | 10569 | 0 | 0 | exact_code_content_signature | 0.8667 |
| pydata__xarray-4075 | stock_sglang_prefix_only | 1278.83 | 11270 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4075 | kvflow_style_prefix_baseline | 1285.9 | 11270 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4075 | kvflow_style_prefix_plus_hints | 1252.38 | 11273 | 0 | 0 | 0 | 0 |  | 0.8333 |
| pydata__xarray-4075 | agenttemplatekv_exact_reuse | 1325.93 | 11271 | 0 | 9785 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-4094 | stock_sglang_prefix_only | 2267.44 | 10959 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4094 | kvflow_style_prefix_baseline | 1728.56 | 10959 | 0 | 0 | 0 | 0 |  | 0.459 |
| pydata__xarray-4094 | kvflow_style_prefix_plus_hints | 2142.99 | 10962 | 0 | 0 | 0 | 0 |  | 0.9091 |
| pydata__xarray-4094 | agenttemplatekv_exact_reuse | 1562.81 | 10960 | 0 | 9788 | 0 | 0 | exact_code_content_signature | 0.459 |
| pydata__xarray-4356 | stock_sglang_prefix_only | 1079.74 | 11103 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4356 | kvflow_style_prefix_baseline | 1081.57 | 11103 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4356 | kvflow_style_prefix_plus_hints | 1082.95 | 11106 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4356 | agenttemplatekv_exact_reuse | 1064.18 | 11104 | 0 | 9791 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-4629 | stock_sglang_prefix_only | 721.65 | 12022 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4629 | kvflow_style_prefix_baseline | 729.61 | 12022 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4629 | kvflow_style_prefix_plus_hints | 1111.47 | 12025 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-4629 | agenttemplatekv_exact_reuse | 737.94 | 12023 | 0 | 10559 | 0 | 0 | exact_code_content_signature | 1.0 |
| pydata__xarray-4687 | stock_sglang_prefix_only | 2422.72 | 12615 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4687 | kvflow_style_prefix_baseline | 2410.57 | 12615 | 0 | 0 | 0 | 0 |  | 0.0215 |
| pydata__xarray-4687 | kvflow_style_prefix_plus_hints | 1035.34 | 12618 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-4687 | agenttemplatekv_exact_reuse | 1015.09 | 12616 | 0 | 10540 | 0 | 0 | exact_code_content_signature | 0.0 |
| pydata__xarray-4695 | stock_sglang_prefix_only | 2438.41 | 12464 | 0 | 0 | 0 | 0 |  | 1.0 |
| pydata__xarray-4695 | kvflow_style_prefix_baseline | 1695.48 | 12464 | 0 | 0 | 0 | 0 |  | 0.026 |
| pydata__xarray-4695 | kvflow_style_prefix_plus_hints | 1070.22 | 12467 | 0 | 0 | 0 | 0 |  | 0.0 |
| pydata__xarray-4695 | agenttemplatekv_exact_reuse | 2500.9 | 12465 | 0 | 10564 | 0 | 0 | exact_code_content_signature | 0.8434 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
