# Pass@1 Repair Sweep Summary

- Git commit: `3d709f3ce`
- Input runs: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_28_repair2_shard1`

| run | mode | n | diff extracted | apply ok | pass@1 | JSON parse fail | search not found | repaired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shard1 | lossless | 7 | 2 | 2 | 0 (0.000) | 0 | 4 | 4 |
| shard1 | lossy | 7 | 2 | 2 | 0 (0.000) | 0 | 4 | 4 |
| shard1 | lossy_prefetch | 7 | 2 | 2 | 0 (0.000) | 0 | 4 | 4 |

## Per-Run Failure Cases

- `shard1 / lossless`: matplotlib__matplotlib-14623:test_failed;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:search_not_found;psf__requests-1142:generation_error;psf__requests-1724:search_not_found
- `shard1 / lossy`: matplotlib__matplotlib-14623:test_failed;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:search_not_found;psf__requests-1142:generation_error;psf__requests-1724:search_not_found
- `shard1 / lossy_prefetch`: matplotlib__matplotlib-14623:test_failed;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:search_not_found;psf__requests-1142:generation_error;psf__requests-1724:search_not_found
