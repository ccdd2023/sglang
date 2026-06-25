# AgentTemplateKV Coding Prefetch Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/swe_verified_10_instances.json`
- Cases: 3
- Git commit: `5fb934751`
- Command: `benchmark/multi_workflow/bench_coding_kvflow_prefetch.py --reuse-server --port 30000 --max-cases 3 --start-index 0 --files-per-case 3 --disable-hierarchical-cache --emit-ttft --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_ttft_smoke_20260615`
- Flush cache per case: `False`
- Concurrent clients: `1`
- Baseline profile: `agenttemplatekv`
- HiCache storage backend: `disabled`
- Hierarchical cache: `False`
- Server extra args: ``
- Resolved server extra args: ``
- LMCache config: ``
- Safety rule: codebase prefetch may predict future code blocks, but AgentTemplateKV reuse still requires `exact_code_content_signature`.

## Main Table

| mode | cases | avg latency ms | p50 | p90 | p99 | avg cached tokens | avg hints | avg prefetch queued | protected hit | protected toks | expired toks | large-gap rejects | exact-content hit | avg token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 3 | 2203.4 | 2263.7 | 2264.5 | 2264.7 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_baseline | 3 | 2080.8 | 2256.7 | 2314.5 | 2327.6 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| kvflow_style_prefix_plus_hints | 3 | 2061.1 | 1997.1 | 2206.5 | 2253.7 | 0.0 | 3.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | 1.0000 |
| agenttemplatekv_exact_reuse | 3 | 1981.5 | 1937.2 | 2185.8 | 2241.8 | 0.0 | 3.0 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 1.00 | 1.0000 |

## Figures

Figure generation skipped because a clean `matplotlib` import is unavailable in this environment.

## Per-Case Table

| instance_id | mode | latency ms | cached | protected hit | protected toks | expired toks | large-gap rejects | match reason | token F1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| astropy__astropy-12907 | stock_sglang_prefix_only | 2264.75 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_baseline | 2256.67 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | kvflow_style_prefix_plus_hints | 2258.89 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| astropy__astropy-12907 | agenttemplatekv_exact_reuse | 2247.98 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| django__django-10097 | stock_sglang_prefix_only | 2081.8 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | kvflow_style_prefix_baseline | 2329.01 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | kvflow_style_prefix_plus_hints | 1927.37 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| django__django-10097 | agenttemplatekv_exact_reuse | 1937.16 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |
| matplotlib__matplotlib-13989 | stock_sglang_prefix_only | 2263.74 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_baseline | 1656.61 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | kvflow_style_prefix_plus_hints | 1997.13 | 0 | 0 | 0 | 0 | 0 |  | 1.0 |
| matplotlib__matplotlib-13989 | agenttemplatekv_exact_reuse | 1759.36 | 0 | 0 | 0 | 0 | 0 | exact_code_content_signature | 1.0 |

## Interpretation

This benchmark isolates serving-side AgentTemplateKV behavior. It does not replace the SWE-bench pass@1 table; use it to show whether coding-aware prefetch improves cached-token/latency behavior before running expensive candidate tests.

When `hicache_storage_backend` is disabled, `codebase_prefetch_hints` still verifies template-to-engine guidance and AgentTemplateKV exact-content hits. The device-first protected-anchor counters (`agenttemplatekv_*`) report whether hints become protected device anchors without host load-back. Enable `--hicache-storage-backend file` only for storage-specific debugging; the current local file backend can trip SGLang's runtime memory checker on long coding prompts.
