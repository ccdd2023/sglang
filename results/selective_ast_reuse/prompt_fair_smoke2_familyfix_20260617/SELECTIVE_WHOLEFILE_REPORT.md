# Selective whole-file AST reuse

- Warmup protocol: `fair_planner_per_mode`
- Protocol meaning: Prompt-fair mechanism protocol: for each mode, flush, run the same Planner warmup, then measure the same target prompt; only runtime reuse anchors differ.

| mode | n_ok/n | prompt fair | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | est recomputed | exact sig hit | exact output | token F1 vs lossless | predicted-d rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 2/2 | 1.00 | 2317.8 | 749.7 | 1.00x | 6.0 | 0.0 | 3788.5 | 0.00 | 1.00 | 1.0000 | 0 |
| `whole_file_reuse_all` | 2/2 | 1.00 | 2306.6 | 737.9 | 1.02x | 0.0 | 3270.0 | 0.0 | 0.50 | 0.50 | 0.9409 | 0 |
| `selective_function_method_reuse` | 2/2 | 1.00 | 2312.6 | 743.4 | 1.01x | 6.0 | 883.0 | 2905.5 | 0.50 | 1.00 | 1.0000 | 0 |
| `selective_extended_reuse` | 1/2 | 1.00 | 2587.1 | 1004.7 | 0.75x | 6.0 | 1794.0 | 0.0 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_oracle_low_dnorm` | 1/2 | 1.00 | 2582.1 | 999.7 | 0.75x | 6.0 | 1794.0 | 0.0 | 1.00 | 1.00 | 1.0000 | 0 |
| `graph_aware_lossy` | 1/2 | 1.00 | 2047.1 | 493.3 | 1.52x | 6.0 | 584.0 | 1486.0 | 1.00 | 1.00 | 1.0000 | 0 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the prompt-fair protocol: each mode starts from a fresh cache, runs the same Planner warmup, and then measures the same target prompt. Rows from prompt-unfair cases are excluded from mode aggregates.

## Skipped rows

```json
{
  "selective_extended_reuse": {
    "fair_planner_per_mode;target_skipped:ValueError": 1
  },
  "selective_oracle_low_dnorm": {
    "fair_planner_per_mode;target_skipped:ValueError": 1
  },
  "graph_aware_lossy": {
    "fair_planner_per_mode;target_skipped:ValueError": 1
  }
}
```
