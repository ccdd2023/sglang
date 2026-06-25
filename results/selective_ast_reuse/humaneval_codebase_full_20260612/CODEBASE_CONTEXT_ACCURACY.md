# humaneval codebase-context selective accuracy

| mode | pass result | avg cached | avg reused toks | avg recomputed toks |
|---|---|---:|---:|---:|
| `lossless_full_prefill` | `{'pass@1': np.float64(0.7073170731707317)}` | 294.2 | 0.0 | 219.8 |
| `whole_file_reuse_all` | `{'pass@1': np.float64(0.7073170731707317)}` | 292.5 | 105.7 | 114.1 |
| `selective_function_method_reuse` | `{'pass@1': np.float64(0.7073170731707317)}` | 294.2 | 98.1 | 121.7 |

HumanEval/MBPP are accuracy sanity checks; do not pool them with SWE/codebase pass@1.
