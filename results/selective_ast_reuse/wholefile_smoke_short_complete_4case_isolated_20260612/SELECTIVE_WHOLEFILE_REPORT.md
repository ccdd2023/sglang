# Selective whole-file AST reuse smoke

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 4 | 1790.5 | 1773.4 | 0.0 | 0.0 | 9849.0 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 4 | 590.5 | 592.3 | 11696.8 | 4796.8 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 4 | 605.3 | 609.2 | 11696.8 | 3727.8 | 6121.2 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 4 | 596.4 | 594.6 | 11696.8 | 3727.8 | 6121.2 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
