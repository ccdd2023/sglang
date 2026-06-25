# Selective whole-file AST reuse smoke

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 3 | 1515.2 | 1322.6 | 0.0 | 0.0 | 8231.7 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 3 | 584.9 | 573.0 | 9401.7 | 3558.0 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 3 | 604.3 | 604.1 | 9401.7 | 3319.3 | 4912.3 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 3 | 601.9 | 593.8 | 9401.7 | 3319.3 | 4912.3 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
