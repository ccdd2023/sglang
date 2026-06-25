# Selective whole-file AST reuse

| mode | n | avg elapsed ms | avg TTFT ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 5 | 2076.1 | 519.4 | 0.0 | 0.0 | 1747.8 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 5 | 1618.0 | 60.2 | 5148.8 | 2192.6 | 0.0 | 1.00 | 0.8879 |
| `selective_function_method_reuse` | 5 | 1987.6 | 429.9 | 924.6 | 353.2 | 1394.6 | 0.20 | 1.0000 |
| `selective_extended_reuse` | 5 | 1619.8 | 61.7 | 5280.2 | 739.0 | 0.0 | 0.80 | 0.6879 |
| `selective_oracle_low_dnorm` | 5 | 1677.6 | 59.4 | 4012.5 | 739.0 | 0.0 | 0.60 | 0.6259 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
By default this benchmark flushes between modes, then runs that mode's warmup before the target request. This avoids order contamination where a later mode reuses KV inserted by an earlier mode.
