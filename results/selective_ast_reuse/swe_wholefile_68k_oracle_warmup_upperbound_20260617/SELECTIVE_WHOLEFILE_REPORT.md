# Selective whole-file AST reuse

- Warmup protocol: `oracle_per_mode`
- Protocol meaning: Controlled upper bound: flush before each mode, run that mode's own warmup, then measure target.

| mode | n_ok/n | avg elapsed ms | avg TTFT ms | speedup vs lossless | avg cached | est reused | est recomputed | exact sig hit | exact output | token F1 vs lossless | predicted-d rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 28/28 | 2314.6 | 699.7 | 1.00x | 0.0 | 0.0 | 5948.0 | 0.00 | 1.00 | 1.0000 | 0 |
| `whole_file_reuse_all` | 28/28 | 1660.7 | 44.4 | 15.75x | 6802.2 | 2472.6 | 0.0 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_function_method_reuse` | 26/28 | 1661.1 | 42.6 | 16.42x | 6840.5 | 2252.5 | 3747.3 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_extended_reuse` | 8/28 | 1639.0 | 38.7 | 18.09x | 4115.4 | 1505.6 | 2591.6 | 1.00 | 1.00 | 1.0000 | 0 |
| `selective_oracle_low_dnorm` | 8/28 | 1638.8 | 38.5 | 18.18x | 4115.4 | 1505.6 | 2591.6 | 1.00 | 1.00 | 1.0000 | 0 |
| `graph_aware_lossy` | 24/28 | 1646.9 | 39.4 | 17.75x | 5366.7 | 428.8 | 1613.2 | 1.00 | 1.00 | 1.0000 | 0 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
This is a controlled mechanism upper bound: each mode gets its own isolated warmup before target measurement.

## Skipped rows

```json
{
  "selective_function_method_reuse": {
    "oracle_per_mode;target_skipped:ValueError": 2
  },
  "selective_extended_reuse": {
    "oracle_per_mode;target_skipped:ValueError": 20
  },
  "selective_oracle_low_dnorm": {
    "oracle_per_mode;target_skipped:ValueError": 20
  },
  "graph_aware_lossy": {
    "oracle_per_mode;target_skipped:ValueError": 4
  }
}
```
