# Selective whole-file AST reuse

| mode | n_ok/n | avg elapsed ms | avg TTFT ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 28/28 | 2190.4 | 580.6 | 0.0 | 0.0 | 1746.5 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 28/28 | 1652.4 | 41.5 | 5775.6 | 1914.9 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 28/28 | 2057.5 | 446.3 | 1427.6 | 433.6 | 1312.9 | 0.25 | 1.0000 |
| `selective_extended_reuse` | 25/28 | 1656.2 | 45.0 | 5824.7 | 822.9 | 534.2 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 25/28 | 1657.8 | 47.0 | 5824.7 | 822.9 | 534.2 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
By default this benchmark flushes between modes, then runs that mode's warmup before the target request. This avoids order contamination where a later mode reuses KV inserted by an earlier mode.
