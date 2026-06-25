# humaneval codebase-context selective accuracy

| mode | pass result | avg cached | avg reused toks | avg recomputed toks |
|---|---|---:|---:|---:|
| `lossless_full_prefill` | `{'pass@1': np.float64(0.75)}` | 256.4 | 0.0 | 184.2 |
| `whole_file_reuse_all` | `{'pass@1': np.float64(0.75)}` | 242.5 | 89.1 | 95.2 |
| `selective_function_method_reuse` | `{'pass@1': np.float64(0.75)}` | 256.4 | 79.2 | 105.1 |

HumanEval/MBPP are accuracy sanity checks; do not pool them with SWE/codebase pass@1.
