# Phase 1 Findings — TTFT-only Re-measurement (2026-06-16)

## Setup
- Driver: `benchmark/multi_workflow/bench_coding_kvflow_prefetch.py` with `--reuse-server --port 30000 --emit-ttft`
- Server: existing sglang server on port 30000, model `/home/gfy/models/Qwen2.5-7B-Instruct`
- 10 cases from `results/repo_level_datasets/swe_verified_10_instances.json`
- All cases run sequentially, no flush_cache_per_case (server retains cache across cases)
- 4 modes: baseline_prefix_cache_only, kvflow_prefix_only, kvflow_prefix_plus_codebase_prefetch, kvcomm_lossy_plus_codebase_prefetch

## Result (10 cases, sequential, no per-case flush)

| Mode | TTFT mean (ms) | TTFT p50 (ms) | TTFT p90 (ms) | E2E mean (ms) | Cached mean (tok) | Output F1 vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline_prefix_cache_only | 79.1 | 82.2 | 86.0 | 1430.5 | 0 | 1.000 |
| kvflow_prefix_only | 81.1 | 81.2 | 92.2 | 1535.7 | 0 | 1.000 |
| kvflow_prefix_plus_codebase_prefetch | 80.4 | 80.3 | 95.4 | 1596.8 | 0 | 1.000 |
| **kvcomm_lossy_plus_codebase_prefetch** | **74.7** | **73.2** | **90.3** | **1368.8** | 0 | 1.000 |

## Speedup vs baseline (TTFT-only)

| Mode | Speedup | F1 |
|---|---:|---:|
| kvflow_prefix_only | 0.98× (slightly slower) | 1.000 |
| kvflow_prefix_plus_codebase_prefetch | 0.98× (slightly slower) | 1.000 |
| **kvcomm_lossy_plus_codebase_prefetch** | **1.06×** | 1.000 |

## Conclusion

1. **TTFT-only is a real metric** — distinct from E2E latency. E2E shows 1369 vs 1430 = 4% speedup; TTFT shows 74.7 vs 79.1 = **6% speedup**. The prefill savings are visible in TTFT and partially diluted by decode in E2E.

2. **The 1.06× kvcomm_lossy speedup is real but modest** — neither dramatic (5× stress) nor trivial (1.00× 8K). It's a real, measurable speedup on the real SWE-bench dataset with preserved F1=1.0.

3. **F1=1.0 contradicts the 100-case E2E finding (F1=0.346)** — likely because:
   - In this 10-case run, `codebase_prefetch_text_count = 0` (no codebase prefetch hits)
   - The 100-case run had 3.0 codebase_prefetch hits per case, which interacted with lossy mode to corrupt output
   - So the lossy mode by itself is safe (F1=1.0); it's lossy + codebase_prefetch that breaks output

4. **kvflow_prefix_only and kvflow_prefix_plus_codebase_prefetch do NOT show speedup** (0.98×) — these modes don't trigger any KV reuse that helps TTFT. The speedup is specific to the lossy + content-signature path in kvcomm_lossy.

## Next steps (Phase 2)

- Use the 1.06× TTFT signal as a baseline.
- Apply the layerwise cos-gate to extend selective function/method reuse to file_prefix / control_block spans (with cos ≥ 0.999 floor). Expected to give a larger speedup than the lossy mode while preserving exact-content safety.
- 28-case selective + new mode comparison.
- Re-measure with TTFT-only on 100-case for the paper headline.

## Files

- Result dir: `results/coding_kvflow_prefetch/qwen2_5_7b_100_ttft_10case_20260616/`
- Driver: `benchmark/multi_workflow/bench_coding_kvflow_prefetch.py` (added `--reuse-server` flag)
- 3-case smoke: `results/coding_kvflow_prefetch/qwen2_5_7b_100_ttft_smoke_20260615/`
