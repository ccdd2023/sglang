# Selective whole-file AST reuse

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 2 | 1816.0 | 1816.0 | 17356.5 | 0.0 | 14433.5 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 2 | 1755.1 | 1755.1 | 17356.5 | 7441.0 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 2 | 3600.5 | 3600.5 | 0.0 | 5355.5 | 9078.0 | 0.00 | 0.9274 |
| `selective_oracle_low_dnorm` | 2 | 2601.0 | 2601.0 | 9291.0 | 5355.5 | 9078.0 | 0.50 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
