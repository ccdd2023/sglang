# Selective whole-file AST reuse

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 3 | 1679.7 | 1683.3 | 6282.0 | 0.0 | 6934.7 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 3 | 1680.0 | 1680.9 | 6282.0 | 2743.0 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 3 | 1676.5 | 1686.4 | 6282.0 | 2450.0 | 4484.7 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 3 | 1677.9 | 1684.7 | 6282.0 | 2450.0 | 4484.7 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
