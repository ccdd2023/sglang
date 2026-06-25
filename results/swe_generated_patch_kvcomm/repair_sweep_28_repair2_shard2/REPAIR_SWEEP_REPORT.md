# Pass@1 Repair Sweep Summary

- Git commit: `3d709f3ce`
- Input runs: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_28_repair2_shard2`

| run | mode | n | diff extracted | apply ok | pass@1 | JSON parse fail | search not found | repaired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shard2 | lossless | 7 | 5 | 5 | 0 (0.000) | 0 | 2 | 3 |
| shard2 | lossy | 7 | 4 | 4 | 0 (0.000) | 1 | 2 | 3 |
| shard2 | lossy_prefetch | 7 | 4 | 4 | 0 (0.000) | 1 | 2 | 3 |

## Per-Run Failure Cases

- `shard2 / lossless`: psf__requests-1766:test_failed;pydata__xarray-2905:test_failed;pydata__xarray-3095:test_failed;pydata__xarray-3151:search_not_found;pylint-dev__pylint-4551:search_not_found;pylint-dev__pylint-4604:test_failed;pylint-dev__pylint-4661:test_failed
- `shard2 / lossy`: psf__requests-1766:json_parse_failed;pydata__xarray-2905:test_failed;pydata__xarray-3095:test_failed;pydata__xarray-3151:search_not_found;pylint-dev__pylint-4551:search_not_found;pylint-dev__pylint-4604:test_failed;pylint-dev__pylint-4661:test_failed
- `shard2 / lossy_prefetch`: psf__requests-1766:json_parse_failed;pydata__xarray-2905:test_failed;pydata__xarray-3095:test_failed;pydata__xarray-3151:search_not_found;pylint-dev__pylint-4551:search_not_found;pylint-dev__pylint-4604:test_failed;pylint-dev__pylint-4661:test_failed
