# Selective whole-file AST reuse full 29-case merged

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 29 | 1676.2 | 1726.1 | 12560.3 | 0.0 | 9682.8 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 29 | 1675.2 | 1729.0 | 12560.3 | 4259.2 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 29 | 2964.2 | 3175.4 | 0.0 | 3914.1 | 5768.7 | 0.00 | 0.8966 |
| `selective_oracle_low_dnorm` | 29 | 2326.4 | 2169.3 | 6165.2 | 3914.1 | 5768.7 | 0.45 | 1.0000 |

Interpretation: this is the complete 29-case SWE/codebase selective whole-file dataset under the 80KB, one-complete-file-per-case rule. It is merged from one 27-case run and one 2-case tail shard because the long-running SGLang server disconnected after 27 cases; every selected case and every mode is represented exactly once.
