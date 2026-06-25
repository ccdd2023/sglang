# Pass@1 Repair Sweep Summary

- Git commit: `3d709f3ce`
- Input runs: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_28_repair2_shard0`

| run | mode | n | diff extracted | apply ok | pass@1 | JSON parse fail | search not found | repaired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shard0 | lossless | 7 | 6 | 6 | 0 (0.000) | 0 | 1 | 1 |
| shard0 | lossy | 7 | 6 | 6 | 0 (0.000) | 0 | 1 | 1 |
| shard0 | lossy_prefetch | 7 | 6 | 6 | 0 (0.000) | 0 | 1 | 1 |

## Per-Run Failure Cases

- `shard0 / lossless`: astropy__astropy-12907:test_failed;astropy__astropy-13033:test_failed;astropy__astropy-13236:search_not_found;django__django-10097:test_failed;django__django-10554:test_failed;django__django-10880:test_failed;matplotlib__matplotlib-13989:test_failed
- `shard0 / lossy`: astropy__astropy-12907:test_failed;astropy__astropy-13033:test_failed;astropy__astropy-13236:search_not_found;django__django-10097:test_failed;django__django-10554:test_failed;django__django-10880:test_failed;matplotlib__matplotlib-13989:test_failed
- `shard0 / lossy_prefetch`: astropy__astropy-12907:test_failed;astropy__astropy-13033:test_failed;astropy__astropy-13236:search_not_found;django__django-10097:test_failed;django__django-10554:test_failed;django__django-10880:test_failed;matplotlib__matplotlib-13989:test_failed
