# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | est recomputed | exact sig hit | exact output | token F1 vs lossless | predicted-d rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 28/28 | 1.00 | 2430.6 | 855.2 | 1.00x | 4.9 | 0.0 | 3793.4 | 0.00 | 1.00 | 1.0000 | 0 |
| `whole_file_reuse_all` | 28/28 | 1.00 | 2422.1 | 846.1 | 1.01x | 0.0 | 2990.8 | 0.0 | 0.61 | 0.46 | 0.8752 | 0 |
| `selective_function_method_reuse` | 27/28 | 1.00 | 2417.2 | 840.0 | 1.02x | 5.1 | 1088.6 | 2645.4 | 0.63 | 1.00 | 1.0000 | 0 |
| `selective_extended_reuse` | 16/28 | 1.00 | 2368.3 | 795.0 | 1.08x | 5.2 | 1398.1 | 1257.8 | 0.94 | 1.00 | 1.0000 | 0 |
| `selective_oracle_low_dnorm` | 16/28 | 1.00 | 2367.9 | 794.2 | 1.08x | 5.2 | 1398.1 | 1257.8 | 0.94 | 1.00 | 1.0000 | 0 |
| `graph_aware_lossy` | 19/28 | 1.00 | 2447.1 | 869.4 | 0.98x | 4.4 | 386.2 | 2645.5 | 0.89 | 1.00 | 1.0000 | 0 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.

## Skipped rows

```json
{
  "selective_function_method_reuse": {
    "fair_planner_per_mode_skipped:ValueError;target_skipped:ValueError": 1
  },
  "selective_extended_reuse": {
    "fair_planner_per_mode;target_skipped:ValueError": 9,
    "fair_planner_per_mode_skipped:ValueError;target_skipped:ValueError": 3
  },
  "selective_oracle_low_dnorm": {
    "fair_planner_per_mode;target_skipped:ValueError": 9,
    "fair_planner_per_mode_skipped:ValueError;target_skipped:ValueError": 3
  },
  "graph_aware_lossy": {
    "fair_planner_per_mode;target_skipped:ValueError": 9
  }
}
```
