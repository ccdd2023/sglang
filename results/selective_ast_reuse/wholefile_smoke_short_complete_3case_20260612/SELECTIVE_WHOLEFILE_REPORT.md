# Selective whole-file AST reuse smoke

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 3 | 589.0 | 585.7 | 9399.7 | 0.0 | 8231.7 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 3 | 1512.6 | 1312.3 | 0.0 | 745.0 | 7486.7 | 1.00 | 0.9811 |
| `selective_function_method_reuse` | 3 | 590.7 | 587.4 | 9399.7 | 3319.3 | 4912.3 | 1.00 | 0.9043 |
| `selective_oracle_low_dnorm` | 3 | 593.4 | 579.8 | 9400.7 | 3319.3 | 4912.3 | 1.00 | 0.9535 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
