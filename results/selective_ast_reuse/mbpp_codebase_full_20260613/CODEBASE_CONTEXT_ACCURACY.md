# mbpp codebase-context selective accuracy

| mode | pass result | avg cached | avg reused toks | avg recomputed toks |
|---|---|---:|---:|---:|
| `lossless_full_prefill` | `{'pass_rate': 0.6381322957198443, 'passed': 164, 'n': 257}` | 302.0 | 0.0 | 131.0 |
| `whole_file_reuse_all` | `{'pass_rate': 0.6381322957198443, 'passed': 164, 'n': 257}` | 301.0 | 85.0 | 46.0 |
| `selective_function_method_reuse` | `{'pass_rate': 0.6381322957198443, 'passed': 164, 'n': 257}` | 302.0 | 30.0 | 101.0 |

HumanEval/MBPP are accuracy sanity checks; do not pool them with SWE/codebase pass@1.
