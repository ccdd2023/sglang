# Selective whole-file AST reuse smoke

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 1 | 574.5 | 574.5 | 5066.0 | 0.0 | 767.0 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 1 | 752.6 | 752.6 | 5066.0 | 767.0 | 0.0 | 1.00 | 0.8837 |
| `selective_function_method_reuse` | 1 | 573.5 | 573.5 | 5066.0 | 0.0 | 767.0 | 0.00 | 0.6000 |
| `selective_oracle_low_dnorm` | 1 | 565.2 | 565.2 | 5067.0 | 0.0 | 767.0 | 0.00 | 0.8000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
