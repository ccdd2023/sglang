# Selective whole-file AST reuse

| mode | n | avg elapsed ms | avg TTFT ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 6 | 2122.6 | 515.6 | 0.0 | 0.0 | 1572.5 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 6 | 2122.2 | 512.5 | 0.0 | 2195.7 | 0.0 | 0.00 | 1.0000 |
| `selective_function_method_reuse` | 6 | 2116.6 | 508.8 | 0.0 | 294.3 | 1278.2 | 0.00 | 1.0000 |
| `selective_extended_reuse` | 6 | 2123.1 | 514.7 | 0.0 | 730.4 | 0.0 | 0.00 | 0.8333 |
| `selective_oracle_low_dnorm` | 6 | 2127.7 | 519.2 | 0.0 | 730.4 | 0.0 | 0.00 | 0.8333 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
By default this benchmark flushes between modes, then runs that mode's warmup before the target request. This avoids order contamination where a later mode reuses KV inserted by an earlier mode.
