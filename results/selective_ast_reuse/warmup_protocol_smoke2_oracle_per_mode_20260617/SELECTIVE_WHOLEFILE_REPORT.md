# Selective whole-file AST reuse

- Warmup protocol: `oracle_per_mode`
- Protocol meaning: Controlled upper bound: flush before each mode, run that mode's own warmup, then measure target.

| mode | n_ok/n | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | est recomputed | exact sig hit | exact output | token F1 vs lossless | predicted-d rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 2/2 | 1040.0 | 515.5 | 1.00x | 0.0 | 0.0 | 5654.0 | 0.00 | 1.00 | 1.0000 | 0 |
| `whole_file_reuse_all` | 2/2 | 560.7 | 35.1 | 14.70x | 4967.0 | 2164.0 | 0.0 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_function_method_reuse` | 2/2 | 563.5 | 36.4 | 14.17x | 4967.0 | 1769.5 | 3884.5 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_extended_reuse` | 0/2 |  |  |  | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.0000 | 0 |
| `selective_oracle_low_dnorm` | 0/2 |  |  |  | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.0000 | 0 |
| `graph_aware_lossy` | 2/2 | 568.0 | 42.4 | 12.16x | 4967.0 | 1314.0 | 850.0 | 1.00 | 1.00 | 1.0000 | 0 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is a controlled mechanism upper bound: each mode gets its own isolated warmup before target measurement.

## Skipped rows

```json
{
  "selective_extended_reuse": {
    "oracle_per_mode;target_skipped:ValueError": 2
  },
  "selective_oracle_low_dnorm": {
    "oracle_per_mode;target_skipped:ValueError": 2
  }
}
```
