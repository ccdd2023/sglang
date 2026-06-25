# Selective whole-file AST reuse

| mode | n | avg elapsed ms | avg TTFT ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 6 | 2116.0 | 509.2 | 0.0 | 0.0 | 1572.5 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 6 | 1649.4 | 41.4 | 5157.0 | 2195.7 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 6 | 2043.0 | 436.0 | 770.5 | 294.3 | 1278.2 | 0.17 | 1.0000 |
| `selective_extended_reuse` | 6 | 1655.6 | 47.4 | 5263.8 | 730.4 | 0.0 | 0.83 | 0.8333 |
| `selective_oracle_low_dnorm` | 6 | 1654.6 | 46.5 | 5263.8 | 730.4 | 0.0 | 0.83 | 0.8333 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
By default this benchmark flushes between modes, then runs that mode's warmup before the target request. This avoids order contamination where a later mode reuses KV inserted by an earlier mode.
