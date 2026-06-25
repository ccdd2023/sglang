# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 12
- Git commit: `5fb934751`
- Command: `/home/gfy/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --model /home/gfy/models/Qwen2.5-7B-Instruct --dataset results/repo_level_datasets/swe_verified_100_instances.json --manifest results/repo_level_datasets/manifest_100.json --start-index 88 --max-cases 12 --max-tokens 128 --baseline-profile lmcache --server-extra-args --disable-overlap-schedule --max-running-requests 1 --flush-cache-per-case --port 31353 --server-timeout 600 --eval-timeout 3600 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard88_12_flush_noidlecheck`
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
| stock_sglang_prefix_only | 12 | 1289.7 | 1169.7 | 1896.3 | 2424.5 | 11841.1 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 12 | 1253.1 | 1104.7 | 1964.2 | 2250.0 | 11841.1 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.7685 |
| kvflow_style_prefix_plus_hints | 12 | 1325.8 | 1189.5 | 1988.8 | 2191.8 | 11844.1 | 2.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.5732 |
| agenttemplatekv_exact_reuse | 12 | 1441.5 | 1250.6 | 1997.9 | 2203.3 | 11842.1 | 2.0 | 0.0 | 0.00 | 10187.1 | 0.0 | 0.0 | 1.00 | 0.5425 |

## Figures

Figure generation skipped because a clean `matplotlib` import is unavailable in this environment.

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| pytest-dev__pytest-5840 | stock_sglang_prefix_only | 959.45 | 11421 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5840 | kvflow_style_prefix_baseline | 1052.21 | 11421 | 0 | 0 | 0 | 0 |  | 0.8667 |
| pytest-dev__pytest-5840 | kvflow_style_prefix_plus_hints | 948.38 | 11424 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-5840 | agenttemplatekv_exact_reuse | 958.29 | 11422 | 0 | 9939 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-6197 | stock_sglang_prefix_only | 977.2 | 11677 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-6197 | kvflow_style_prefix_baseline | 1050.76 | 11677 | 0 | 0 | 0 | 0 |  | 0.9375 |
| pytest-dev__pytest-6197 | kvflow_style_prefix_plus_hints | 982.51 | 11680 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-6197 | agenttemplatekv_exact_reuse | 1254.27 | 11678 | 0 | 9925 | 0 | 0 | exact_code_content_signature | 0.6842 |
| pytest-dev__pytest-6202 | stock_sglang_prefix_only | 1026.28 | 11404 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-6202 | kvflow_style_prefix_baseline | 1022.42 | 11404 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-6202 | kvflow_style_prefix_plus_hints | 1098.73 | 11407 | 0 | 0 | 0 | 0 |  | 0.8333 |
| pytest-dev__pytest-6202 | agenttemplatekv_exact_reuse | 1157.21 | 11405 | 0 | 9925 | 0 | 0 | exact_code_content_signature | 0.8333 |
| pytest-dev__pytest-7205 | stock_sglang_prefix_only | 1371.66 | 12275 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7205 | kvflow_style_prefix_baseline | 1361.44 | 12275 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7205 | kvflow_style_prefix_plus_hints | 1580.87 | 12278 | 0 | 0 | 0 | 0 |  | 0.7647 |
| pytest-dev__pytest-7205 | agenttemplatekv_exact_reuse | 1627.26 | 12276 | 0 | 10037 | 0 | 0 | exact_code_content_signature | 0.7647 |
| pytest-dev__pytest-7236 | stock_sglang_prefix_only | 1180.31 | 11417 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7236 | kvflow_style_prefix_baseline | 1190.44 | 11417 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7236 | kvflow_style_prefix_plus_hints | 1242.84 | 11420 | 0 | 0 | 0 | 0 |  | 0.9302 |
| pytest-dev__pytest-7236 | agenttemplatekv_exact_reuse | 1246.84 | 11418 | 0 | 10046 | 0 | 0 | exact_code_content_signature | 0.9302 |
| pytest-dev__pytest-7324 | stock_sglang_prefix_only | 1049.66 | 10665 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7324 | kvflow_style_prefix_baseline | 1043.31 | 10665 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7324 | kvflow_style_prefix_plus_hints | 1065.18 | 10668 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7324 | agenttemplatekv_exact_reuse | 1051.06 | 10666 | 0 | 10054 | 0 | 0 | exact_code_content_signature | 1.0 |
| pytest-dev__pytest-7432 | stock_sglang_prefix_only | 1954.59 | 10906 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7432 | kvflow_style_prefix_baseline | 2277.1 | 10906 | 0 | 0 | 0 | 0 |  | 0.4179 |
| pytest-dev__pytest-7432 | kvflow_style_prefix_plus_hints | 2213.23 | 10909 | 0 | 0 | 0 | 0 |  | 0.4179 |
| pytest-dev__pytest-7432 | agenttemplatekv_exact_reuse | 2224.96 | 10907 | 0 | 10054 | 0 | 0 | exact_code_content_signature | 0.4179 |
| pytest-dev__pytest-7490 | stock_sglang_prefix_only | 664.04 | 15104 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7490 | kvflow_style_prefix_baseline | 675.27 | 15104 | 0 | 0 | 0 | 0 |  | 1.0 |
| pytest-dev__pytest-7490 | kvflow_style_prefix_plus_hints | 644.97 | 15107 | 0 | 0 | 0 | 0 |  | 0.0 |
| pytest-dev__pytest-7490 | agenttemplatekv_exact_reuse | 1062.95 | 15105 | 0 | 10057 | 0 | 0 | exact_code_content_signature | 0.0 |
| scikit-learn__scikit-learn-10297 | stock_sglang_prefix_only | 2482.55 | 12062 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10297 | kvflow_style_prefix_baseline | 2031.13 | 12062 | 0 | 0 | 0 | 0 |  | 0.0 |
| scikit-learn__scikit-learn-10297 | kvflow_style_prefix_plus_hints | 2018.28 | 12065 | 0 | 0 | 0 | 0 |  | 0.0 |
| scikit-learn__scikit-learn-10297 | agenttemplatekv_exact_reuse | 2028.28 | 12063 | 0 | 10552 | 0 | 0 | exact_code_content_signature | 0.0 |
| scikit-learn__scikit-learn-10844 | stock_sglang_prefix_only | 1335.04 | 11909 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10844 | kvflow_style_prefix_baseline | 1330.42 | 11909 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10844 | kvflow_style_prefix_plus_hints | 1723.15 | 11912 | 0 | 0 | 0 | 0 |  | 0.6129 |
| scikit-learn__scikit-learn-10844 | agenttemplatekv_exact_reuse | 1724.45 | 11910 | 0 | 10555 | 0 | 0 | exact_code_content_signature | 0.6129 |
| scikit-learn__scikit-learn-10908 | stock_sglang_prefix_only | 1159.02 | 11832 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10908 | kvflow_style_prefix_baseline | 1157.12 | 11832 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-10908 | kvflow_style_prefix_plus_hints | 1255.82 | 11835 | 0 | 0 | 0 | 0 |  | 0.1765 |
| scikit-learn__scikit-learn-10908 | agenttemplatekv_exact_reuse | 1238.03 | 11833 | 0 | 10545 | 0 | 0 | exact_code_content_signature | 0.1765 |
| scikit-learn__scikit-learn-11310 | stock_sglang_prefix_only | 1316.0 | 11421 | 0 | 0 | 0 | 0 |  | 1.0 |
| scikit-learn__scikit-learn-11310 | kvflow_style_prefix_baseline | 846.04 | 11421 | 0 | 0 | 0 | 0 |  | 0.0 |
| scikit-learn__scikit-learn-11310 | kvflow_style_prefix_plus_hints | 1136.21 | 11424 | 0 | 0 | 0 | 0 |  | 0.1429 |
| scikit-learn__scikit-learn-11310 | agenttemplatekv_exact_reuse | 1724.04 | 11422 | 0 | 10556 | 0 | 0 | exact_code_content_signature | 0.0909 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
