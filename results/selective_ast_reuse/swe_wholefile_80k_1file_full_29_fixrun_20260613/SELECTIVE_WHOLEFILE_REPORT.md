# Selective whole-file AST reuse

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 26 | 1700.4 | 1746.0 | 12073.8 | 0.0 | 9157.5 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 26 | 1702.3 | 1748.0 | 12073.8 | 3940.0 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 26 | 1704.5 | 1746.0 | 12073.8 | 3737.2 | 5420.4 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 26 | 1706.8 | 1751.9 | 12073.8 | 3737.2 | 5420.4 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
