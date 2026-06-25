# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 2
- Git commit: `5fb934751`
- Command: `/home/gfy/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --model /home/gfy/models/Qwen2.5-7B-Instruct --dataset results/repo_level_datasets/swe_verified_100_instances.json --manifest results/repo_level_datasets/manifest_100.json --start-index 48 --max-cases 2 --max-tokens 128 --baseline-profile lmcache --server-extra-args --disable-overlap-schedule --max-running-requests 1 --flush-cache-per-case --port 31350 --server-timeout 600 --eval-timeout 3600 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_probe48_2_flush_noidlecheck`
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
| stock_sglang_prefix_only | 2 | 986.7 | 986.7 | 1053.0 | 1067.9 | 26732.0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 2 | 903.8 | 903.8 | 974.2 | 990.0 | 26732.0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.3250 |
| kvflow_style_prefix_plus_hints | 2 | 1012.5 | 1012.5 | 1063.3 | 1074.7 | 26735.0 | 2.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.5625 |
| agenttemplatekv_exact_reuse | 2 | 964.9 | 964.9 | 1012.9 | 1023.7 | 26733.0 | 2.0 | 0.0 | 0.00 | 25075.5 | 0.0 | 0.0 | 1.00 | 0.3125 |

## Figures

Figure generation skipped because a clean `matplotlib` import is unavailable in this environment.

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| psf__requests-1142 | stock_sglang_prefix_only | 903.8 | 25580 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1142 | kvflow_style_prefix_baseline | 991.79 | 25580 | 0 | 0 | 0 | 0 |  | 0.25 |
| psf__requests-1142 | kvflow_style_prefix_plus_hints | 949.07 | 25583 | 0 | 0 | 0 | 0 |  | 0.125 |
| psf__requests-1142 | agenttemplatekv_exact_reuse | 1024.92 | 25581 | 0 | 24993 | 0 | 0 | exact_code_content_signature | 0.25 |
| psf__requests-1724 | stock_sglang_prefix_only | 1069.57 | 27884 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1724 | kvflow_style_prefix_baseline | 815.76 | 27884 | 0 | 0 | 0 | 0 |  | 0.4 |
| psf__requests-1724 | kvflow_style_prefix_plus_hints | 1075.98 | 27887 | 0 | 0 | 0 | 0 |  | 1.0 |
| psf__requests-1724 | agenttemplatekv_exact_reuse | 904.93 | 27885 | 0 | 25158 | 0 | 0 | exact_code_content_signature | 0.375 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
