# Selective whole-file AST reuse

- Warmup protocol: `natural_planner`
- Protocol meaning: Realistic agent protocol: measure a cold lossless reference, flush, run one Planner-style warmup, then measure reuse target modes against shared cache.

| mode | n_ok/n | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | est recomputed | exact sig hit | exact output | token F1 vs lossless | predicted-d rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 28/28 | 2316.7 | 700.2 | 1.00x | 0.0 | 0.0 | 5948.0 | 0.00 | 1.00 | 1.0000 | 0 |
| `whole_file_reuse_all` | 28/28 | 2313.4 | 695.8 | 1.01x | 0.0 | 2472.6 | 0.0 | 0.29 | 1.00 | 1.0000 | 0 |
| `selective_function_method_reuse` | 26/28 | 1848.0 | 228.3 | 3.07x | 5069.5 | 2252.5 | 3747.3 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_extended_reuse` | 8/28 | 1701.6 | 99.7 | 7.02x | 3418.1 | 1505.6 | 2591.6 | 0.88 | 1.00 | 1.0000 | 0 |
| `selective_oracle_low_dnorm` | 8/28 | 1642.2 | 40.2 | 17.43x | 4115.4 | 1505.6 | 2591.6 | 1.00 | 1.00 | 1.0000 | 0 |
| `graph_aware_lossy` | 24/28 | 1676.5 | 66.8 | 10.49x | 5073.7 | 428.8 | 1613.2 | 0.96 | 1.00 | 1.0000 | 0 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the realistic protocol: lossless is measured cold as the reference, then one Planner-style warmup is shared by later reuse target modes; target order is recorded in `summary.json`.

## Skipped rows

```json
{
  "selective_function_method_reuse": {
    "natural_planner_skipped:ValueError;target_skipped:ValueError": 2
  },
  "selective_extended_reuse": {
    "natural_planner;target_skipped:ValueError": 16,
    "natural_planner_skipped:ValueError;target_skipped:ValueError": 4
  },
  "selective_oracle_low_dnorm": {
    "natural_planner;target_skipped:ValueError": 16,
    "natural_planner_skipped:ValueError;target_skipped:ValueError": 4
  },
  "graph_aware_lossy": {
    "natural_planner_skipped:ValueError;target_skipped:ValueError": 3,
    "natural_planner;target_skipped:ValueError": 1
  }
}
```
