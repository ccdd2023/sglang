# mbpp codebase-context selective accuracy

| mode | pass result | avg cached | avg reused toks | avg recomputed toks |
|---|---|---:|---:|---:|
| `lossless_full_prefill` | `{'pass_rate': 1.0, 'passed': 3, 'n': 3}` | 297.0 | 0.0 | 132.0 |
| `whole_file_reuse_all` | `{'pass_rate': 1.0, 'passed': 3, 'n': 3}` | 212.3 | 86.0 | 46.0 |
| `selective_function_method_reuse` | `{'pass_rate': 1.0, 'passed': 3, 'n': 3}` | 297.0 | 30.0 | 102.0 |

HumanEval/MBPP are accuracy sanity checks; do not pool them with SWE/codebase pass@1.
