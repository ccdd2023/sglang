# 8-case graph-aware pass@1 with candidate tests summary

rows=32


| mode | n | apply_ok | synthesis_ok | search_not_found | json_parse_failed | candidate_test_runs | candidate_test_pass | mean_cached_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_aware_lossy | 8 | 8/8 | 8/8 | 0/8 | 0/8 | 8 | 0/8 | 962 |
| lossless | 8 | 3/8 | 3/8 | 3/8 | 1/8 | 3 | 0/3 | 8217 |
| lossy | 8 | 2/8 | 2/8 | 5/8 | 0/8 | 2 | 0/2 | 8310 |
| lossy_prefetch | 8 | 2/8 | 2/8 | 5/8 | 0/8 | 2 | 0/2 | 9683 |

## Failure classes

| mode | failure_class_counts |
|---|---|
| graph_aware_lossy | `{"candidate_patch_syntax_failure": 1, "candidate_patch_syntax_or_install_failure": 1, "real_pytest_failure": 6}` |
| lossless | `{"not_run": 5, "real_pytest_failure": 3}` |
| lossy | `{"not_run": 6, "real_pytest_failure": 2}` |
| lossy_prefetch | `{"not_run": 6, "real_pytest_failure": 2}` |
