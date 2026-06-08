# Full-Dataset Speedup + Accuracy (AgentTemplateKV)

Consolidated view of the 100-case serving speedup
(`results/coding_kvflow_prefetch/qwen2_5_7b_100/prefetch_table.csv`) and the 28-case
discriminative-subset pass@1 (`results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/passrate_table.csv`).

## Main Table

| mode (display) | n (e2e) | p50 ms | p90 ms | avg cached | exact hit | F1 | n (acc) | pass@1 | delta vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | 100 | 3872 | 4135 | 1582 | 0.00 | 1.0000 | 28 | 3/28 (10.71%) | +0 |
| kvflow_style_prefix_baseline | 100 | 3878 | 4215 | 1582 | 0.00 | 0.4916 | 28 | 3/28 (10.71%) | +0 |
| kvflow_style_prefix_plus_hints | 100 | 3870 | 4264 | 1585 | 0.00 | 0.4295 | 28 | 3/28 (10.71%) | +0 |
| agenttemplatekv_exact_reuse | 100 | 3832 | 4187 | 2593 | 0.99 | 0.3461 | 28 | 2/28 (7.14%) | -1 |

## Tail Analysis (per mode)

| mode | p50 ms | p90 ms | p99 ms | max ms |
|---|---:|---:|---:|---:|
| stock_sglang_prefix_only | 3872 | 4135 | 6098 | 6218 |
| kvflow_style_prefix_baseline | 3878 | 4215 | 6099 | 6212 |
| kvflow_style_prefix_plus_hints | 3870 | 4264 | 6121 | 6215 |
| agenttemplatekv_exact_reuse | 3832 | 4187 | 6116 | 6221 |

## Statistical Significance (paired bootstrap, 10,000 resamples)

- **Latency**: stock SGLang − AgentTemplateKV = **+73 ms** (95% CI [+15, +132] ms), one-sided p = **0.0068** (n = 100).
- **Cached tokens**: AgentTemplateKV − stock SGLang = **+1011** (95% CI [+563, +1502]), one-sided p = **0.0000** (n = 100).

The latency improvement is significant at p < 0.05.
The cached-token gain is significant at p < 0.05.

## Pass@1 Detail

- **Cases**: 28 discriminative SWE-bench Verified instances (only this subset has local repo envs + gold tests; full 500-case pass@1 requires building more envs and is out of scope for this round).
- **Lossless** (stock SGLang) pass@1: 3/28.
- **AgentTemplateKV exact reuse** pass@1: 2/28.
- **Delta**: -1.
- Regression root-cause: `scikit-learn-10844` is a model-side JSON-edit extraction failure; KVCOMM gate fired correctly. See `results/passrate_28/regression_root_cause.md`.

## Files

- Speedup source: `results/coding_kvflow_prefetch/qwen2_5_7b_100/prefetch_table.csv`
- Pass@1 source: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/passrate_table.csv`
- Aggregated long-format: `merged_table.csv`
- Machine-readable: `summary.json`
