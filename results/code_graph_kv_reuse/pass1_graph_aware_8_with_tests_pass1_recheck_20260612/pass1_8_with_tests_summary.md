# 8-case graph-aware pass@1 with candidate tests summary

rows=32


| mode | n | apply_ok | synthesis_ok | search_not_found | json_parse_failed | candidate_test_runs | candidate_test_pass | mean_cached_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_aware_lossy | 8 | 6/8 | 6/8 | 2/8 | 0/8 | 6 | 1/6 | 2310 |
| lossless | 8 | 6/8 | 6/8 | 2/8 | 0/8 | 6 | 0/6 | 7534 |
| lossy | 8 | 6/8 | 6/8 | 2/8 | 0/8 | 6 | 0/6 | 9513 |
| lossy_prefetch | 8 | 6/8 | 6/8 | 2/8 | 0/8 | 6 | 0/6 | 9514 |

## Failure classes

| mode | failure_class_counts |
|---|---|
| graph_aware_lossy | `{"not_run": 2, "pass": 1, "real_pytest_failure": 5}` |
| lossless | `{"candidate_patch_syntax_failure": 1, "candidate_patch_syntax_or_install_failure": 1, "not_run": 2, "real_pytest_failure": 4}` |
| lossy | `{"candidate_patch_syntax_failure": 1, "candidate_patch_syntax_or_install_failure": 1, "not_run": 2, "real_pytest_failure": 4}` |
| lossy_prefetch | `{"candidate_patch_syntax_failure": 1, "not_run": 2, "real_pytest_failure": 5}` |
