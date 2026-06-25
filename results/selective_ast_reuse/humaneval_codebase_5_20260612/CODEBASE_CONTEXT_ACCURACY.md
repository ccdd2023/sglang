# humaneval codebase-context selective accuracy

| mode | pass result | avg cached | avg reused toks | avg recomputed toks |
|---|---|---:|---:|---:|
| `lossless_full_prefill` | `{'pass@1': np.float64(0.8)}` | 267.0 | 0.0 | 199.4 |
| `whole_file_reuse_all` | `{'pass@1': np.float64(0.8)}` | 211.2 | 96.8 | 102.6 |
| `selective_function_method_reuse` | `{'pass@1': np.float64(0.8)}` | 267.0 | 86.6 | 112.8 |

HumanEval/MBPP are accuracy sanity checks; do not pool them with SWE/codebase pass@1.
