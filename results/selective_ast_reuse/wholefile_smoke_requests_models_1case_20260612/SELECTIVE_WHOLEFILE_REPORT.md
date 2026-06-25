# Selective whole-file AST reuse smoke

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 1 | 570.5 | 570.5 | 4621.0 | 0.0 | 5783.0 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 1 | 1002.3 | 1002.3 | 0.0 | 642.0 | 5141.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 1 | 572.6 | 572.6 | 4621.0 | 1766.0 | 4017.0 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 1 | 579.0 | 579.0 | 4622.0 | 1766.0 | 4017.0 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
