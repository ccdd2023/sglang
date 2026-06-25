# Selective whole-file AST reuse

- Warmup protocol: `natural_planner`
- Protocol meaning: Realistic agent protocol: measure a cold lossless reference, flush, run one Planner-style warmup, then measure reuse target modes against shared cache.

| mode | n_ok/n | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | est recomputed | exact sig hit | exact output | token F1 vs lossless | predicted-d rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 2/2 | 1021.5 | 496.8 | 1.00x | 0.0 | 0.0 | 5654.0 | 0.00 | 1.00 | 1.0000 | 0 |
| `whole_file_reuse_all` | 2/2 | 1024.8 | 499.3 | 1.00x | 0.0 | 2164.0 | 0.0 | 0.00 | 1.00 | 1.0000 | 0 |
| `selective_function_method_reuse` | 2/2 | 561.4 | 33.9 | 14.67x | 4967.0 | 1769.5 | 3884.5 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_extended_reuse` | 0/2 |  |  |  | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.0000 | 0 |
| `selective_oracle_low_dnorm` | 0/2 |  |  |  | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.0000 | 0 |
| `graph_aware_lossy` | 2/2 | 561.6 | 35.5 | 14.01x | 4967.0 | 1314.0 | 850.0 | 1.00 | 1.00 | 1.0000 | 0 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is the realistic protocol: lossless is measured cold as the reference, then one Planner-style warmup is shared by later reuse target modes; target order is recorded in `summary.json`.

## Skipped rows

```json
{
  "selective_extended_reuse": {
    "natural_planner;target_skipped:ValueError": 2
  },
  "selective_oracle_low_dnorm": {
    "natural_planner;target_skipped:ValueError": 2
  }
}
```
