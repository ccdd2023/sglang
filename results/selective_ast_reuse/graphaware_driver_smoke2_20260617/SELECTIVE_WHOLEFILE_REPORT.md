# Selective whole-file AST reuse

| mode | n_ok/n | avg elapsed ms | avg TTFT ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 2/2 | 1568.8 | 504.3 | 0.0 | 0.0 | 5654.0 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 2/2 | 1107.9 | 41.7 | 4967.0 | 2164.0 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 2/2 | 1109.4 | 42.5 | 4967.0 | 1769.5 | 3884.5 | 1.00 | 1.0000 |
| `selective_extended_reuse` | 0/2 |  |  | 0.0 | 0.0 | 0.0 | 0.00 | 0.0000 |
| `selective_oracle_low_dnorm` | 0/2 |  |  | 0.0 | 0.0 | 0.0 | 0.00 | 0.0000 |
| `graph_aware_lossy` | 2/2 | 1112.9 | 47.5 | 4967.0 | 1314.0 | 850.0 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
By default this benchmark flushes between modes, then runs that mode's warmup before the target request. This avoids order contamination where a later mode reuses KV inserted by an earlier mode.
