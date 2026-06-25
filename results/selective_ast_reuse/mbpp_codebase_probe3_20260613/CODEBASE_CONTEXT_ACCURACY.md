# mbpp codebase-context selective accuracy

| mode | pass result | avg cached | avg reused toks | avg recomputed toks |
|---|---|---:|---:|---:|
| `lossless_full_prefill` | `{'pass_rate': 0.0, 'passed': 0, 'n': 3}` | 193.7 | 0.0 | 102.7 |
| `whole_file_reuse_all` | `{'pass_rate': 0.0, 'passed': 0, 'n': 3}` | 128.7 | 56.7 | 46.0 |
| `selective_function_method_reuse` | `{'pass_rate': 0.0, 'passed': 0, 'n': 3}` | 193.7 | 30.0 | 72.7 |

HumanEval/MBPP are accuracy sanity checks; do not pool them with SWE/codebase pass@1.
