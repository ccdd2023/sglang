# Selective whole-file AST reuse

| mode | n | avg elapsed ms | avg TTFT ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 10 | 2126.4 | 520.1 | 0.0 | 0.0 | 2299.9 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 10 | 1652.6 | 45.1 | 5238.8 | 2171.0 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 10 | 1987.2 | 378.1 | 1526.1 | 537.1 | 1762.8 | 0.30 | 1.0000 |
| `selective_extended_reuse` | 10 | 1653.0 | 45.1 | 5307.2 | 1001.0 | 911.9 | 0.90 | 0.9000 |
| `selective_oracle_low_dnorm` | 10 | 1660.1 | 42.9 | 5307.2 | 1001.0 | 911.9 | 0.90 | 0.9000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
By default this benchmark flushes between modes, then runs that mode's warmup before the target request. This avoids order contamination where a later mode reuses KV inserted by an earlier mode.
