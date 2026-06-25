# Selective whole-file AST reuse smoke

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 3 | 587.5 | 584.6 | 9401.7 | 0.0 | 8231.7 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 3 | 590.0 | 577.5 | 9401.7 | 3558.0 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 3 | 1519.9 | 1324.4 | 0.0 | 3319.3 | 4912.3 | 0.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 3 | 1520.5 | 1315.5 | 0.0 | 3319.3 | 4912.3 | 0.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
