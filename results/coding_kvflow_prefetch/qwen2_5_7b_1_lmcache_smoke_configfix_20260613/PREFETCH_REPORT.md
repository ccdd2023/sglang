# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Cases: 1
- Git commit: `5fb934751`
- Command: `/home/gfy/CodeMAS_Project/sglang-kvflow/benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --model /home/gfy/models/Qwen2.5-7B-Instruct --dataset results/repo_level_datasets/swe_verified_100_instances.json --manifest results/repo_level_datasets/manifest_100.json --max-cases 1 --max-tokens 8 --baseline-profile lmcache --port 31344 --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_1_lmcache_smoke_configfix_20260613`
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
| stock_sglang_prefix_only | 1 | 225.2 | 225.2 | 225.2 | 225.2 | 10547.0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 1 | 217.9 | 217.9 | 217.9 | 217.9 | 10547.0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_plus_hints | 1 | 215.9 | 215.9 | 215.9 | 215.9 | 10550.0 | 2.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0000 |
| agenttemplatekv_exact_reuse | 1 | 219.0 | 219.0 | 219.0 | 219.0 | 10548.0 | 2.0 | 0.0 | 0.00 | 9418.0 | 0.0 | 0.0 | 1.00 | 0.0000 |

## Figures

Figure generation skipped because a clean `matplotlib` import is unavailable in this environment.

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 225.25 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 217.89 | 10547 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 215.86 | 10550 | 0 | 0 | 0 | 0 |  | 0.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 218.99 | 10548 | 0 | 9418 | 0 | 0 | exact_code_content_signature | 0.0 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
