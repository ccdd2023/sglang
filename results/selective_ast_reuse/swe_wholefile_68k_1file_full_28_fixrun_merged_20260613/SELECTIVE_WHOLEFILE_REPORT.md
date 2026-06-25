# Selective whole-file AST reuse

Dataset: `swe_selective_wholefile_68k_1file` (complete deterministic set; one full Python file per case, `max_file_chars=68000`).

| mode | n | avg elapsed ms | p50 elapsed ms | avg cached | est reused | est recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless_full_prefill` | 28 | 1687.8 | 1746.0 | 12309.5 | 0.0 | 9366.8 | 0.00 | 1.0000 |
| `whole_file_reuse_all` | 28 | 1688.8 | 1748.0 | 12309.5 | 4047.2 | 0.0 | 1.00 | 1.0000 |
| `selective_function_method_reuse` | 28 | 1692.9 | 1746.0 | 12309.5 | 3833.6 | 5533.2 | 1.00 | 1.0000 |
| `selective_oracle_low_dnorm` | 28 | 1693.8 | 1751.9 | 12309.5 | 3833.6 | 5533.2 | 1.00 | 1.0000 |

Interpretation: agents receive whole-file `code_base` prompts in every mode; only the internal AST spans exposed to the exact-content KV reuse gate differ.
`whole_file_reuse_all` is diagnostic only. The main method is `selective_function_method_reuse`.

Hardware boundary: the 80k exploratory manifest exposed two larger complete-file cases that repeatedly disconnected the single 24GB run before producing measurements; they are not part of this 68k complete deterministic dataset.
