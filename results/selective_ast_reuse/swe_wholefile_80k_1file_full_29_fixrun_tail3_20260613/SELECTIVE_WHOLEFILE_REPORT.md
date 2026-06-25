# Selective whole-file AST reuse

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 1 | 1287.8 | 1287.8 | 15616.0 | 0.0 | 13838.0 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 1 | 1274.1 | 1274.1 | 15616.0 | 6196.0 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 1 | 1301.4 | 1301.4 | 15616.0 | 5633.0 | 8205.0 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 1 | 1293.4 | 1293.4 | 15616.0 | 5633.0 | 8205.0 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
