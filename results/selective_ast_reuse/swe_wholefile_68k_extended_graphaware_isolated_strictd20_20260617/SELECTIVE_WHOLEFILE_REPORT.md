# Selective whole-file AST reuse

| mode | n_ok/n | avg elapsed ms | avg TTFT ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 28/28 | 2315.4 | 699.7 | 0.0 | 0.0 | 5948.0 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 28/28 | 1658.8 | 42.2 | 6802.2 | 2472.6 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 28/28 | 1661.1 | 42.2 | 6802.2 | 2221.0 | 3726.9 | 1.00 | 1.0000 |
| `selective_extended_reuse` | 8/28 | 1636.4 | 35.8 | 4115.4 | 1505.6 | 2591.6 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 8/28 | 1637.4 | 36.9 | 4115.4 | 1505.6 | 2591.6 | 1.00 | 1.0000 |
| `graph_aware_lossy` | 24/28 | 1646.7 | 38.9 | 5366.7 | 428.8 | 1613.2 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file code_base prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`graph_aware_lossy` keeps the same whole-file prompt, then maps relation-selected graph bundles back to exact AST spans already present in that prompt.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.
By default this benchmark flushes between modes, then runs that mode's warmup before the target request. This avoids order contamination where a later mode reuses KV inserted by an earlier mode.
