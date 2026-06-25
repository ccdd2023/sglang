# Pass@1 Repair Sweep Summary

- Git commit: `3d709f3ce`
- Input runs: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_28_repair2_shard3`

| run | mode | n | diff extracted | apply ok | pass@1 | JSON parse fail | search not found | repaired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shard3 | lossless | 7 | 5 | 5 | 2 (0.286) | 0 | 2 | 2 |
| shard3 | lossy | 7 | 5 | 5 | 2 (0.286) | 0 | 2 | 2 |
| shard3 | lossy_prefetch | 7 | 5 | 5 | 1 (0.143) | 0 | 2 | 2 |

## Per-Run Failure Cases

- `shard3 / lossless`: pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10356:test_failed;scikit-learn__scikit-learn-10297:search_not_found;scikit-learn__scikit-learn-10908:test_failed;sphinx-doc__sphinx-10323:search_not_found
- `shard3 / lossy`: pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10356:test_failed;scikit-learn__scikit-learn-10297:search_not_found;scikit-learn__scikit-learn-10908:test_failed;sphinx-doc__sphinx-10323:search_not_found
- `shard3 / lossy_prefetch`: pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10081:test_failed;pytest-dev__pytest-10356:test_failed;scikit-learn__scikit-learn-10297:search_not_found;scikit-learn__scikit-learn-10908:test_failed;sphinx-doc__sphinx-10323:search_not_found
