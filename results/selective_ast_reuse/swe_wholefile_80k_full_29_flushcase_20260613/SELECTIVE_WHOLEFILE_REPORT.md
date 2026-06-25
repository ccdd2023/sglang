# Selective whole-file AST reuse

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 9 | 1721.6 | 1726.6 | 12329.1 | 0.0 | 13894.8 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 9 | 1721.8 | 1732.6 | 12329.1 | 5671.8 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 9 | 2972.8 | 2994.6 | 0.0 | 4862.2 | 9032.6 | 0.00 | 0.9590 |
| `selective_oracle_low_dnorm` | 9 | 2838.4 | 2848.0 | 1435.2 | 4862.2 | 9032.6 | 0.11 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
