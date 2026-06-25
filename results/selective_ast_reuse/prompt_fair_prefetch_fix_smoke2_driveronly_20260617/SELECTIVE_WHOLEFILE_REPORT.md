# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | anchor match len | anchor match rate | prefetch hit | exact sig hit | exact output | token F1 vs lossless | predicted-d rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 2/2 | 1.00 | 2638.2 | 1053.6 | 1.00x | 6.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.00 | 1.00 | 1.0000 | 0 |
| `whole_file_reuse_all` | 2/2 | 1.00 | 2634.7 | 1049.1 | 1.00x | 6.0 | 3854.5 | 0.0 | 0.00 | 0.00 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_function_method_reuse` | 2/2 | 1.00 | 2635.4 | 1046.6 | 1.01x | 6.0 | 1669.0 | 0.0 | 0.00 | 0.00 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_extended_reuse` | 0/2 | 0.00 |  |  |  | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 0 |
| `selective_oracle_low_dnorm` | 0/2 | 0.00 |  |  |  | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 0 |
| `graph_aware_lossy` | 2/2 | 1.00 | 2629.1 | 1043.6 | 1.01x | 6.0 | 821.5 | 0.0 | 0.00 | 0.00 | 1.00 | 1.00 | 1.0000 | 0 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`est reused` is driver-side span volume; `anchor match len` and `prefetch hit` are server-side evidence that KV was actually copied/protected.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.

## Skipped rows

```json
{
  "selective_extended_reuse": {
    "fair_planner_per_mode_skipped:ValueError;target_skipped:ValueError": 2
  },
  "selective_oracle_low_dnorm": {
    "fair_planner_per_mode_skipped:ValueError;target_skipped:ValueError": 2
  }
}
```
