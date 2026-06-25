# Selective whole-file AST reuse

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 28 | 799.5 | 667.9 | 0.0 | 0.0 | 1746.5 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 28 | 801.1 | 665.9 | 0.0 | 1914.9 | 0.0 | 0.50 | 1.0000 |
| `selective_function_method_reuse` | 28 | 800.5 | 667.3 | 0.0 | 433.6 | 1312.9 | 0.14 | 1.0000 |
| `selective_extended_reuse` | 28 | 16775.0 | 1617.7 | 0.0 | 822.9 | 534.2 | 0.25 | 1.0000 |
| `selective_oracle_low_dnorm` | 28 | 767.5 | 1.2 | 0.0 | 822.9 | 534.2 | 0.43 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
