# Selective whole-file AST reuse

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 27 | 1665.9 | 1725.0 | 12205.0 | 0.0 | 9330.9 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 27 | 1669.3 | 1722.2 | 12205.0 | 4023.6 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 27 | 2917.1 | 3122.2 | 0.0 | 3807.4 | 5523.5 | 0.00 | 0.8943 |
| `selective_oracle_low_dnorm` | 27 | 2306.1 | 2169.3 | 5933.7 | 3807.4 | 5523.5 | 0.44 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
