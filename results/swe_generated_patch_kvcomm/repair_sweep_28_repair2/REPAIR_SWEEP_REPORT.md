# Pass@1 Repair Sweep Summary

- Git commit: `3d709f3ce`
- Input runs: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30, results/swe_generated_patch_kvcomm/qwen2_5_7b_json_28_repair2_shard0, results/swe_generated_patch_kvcomm/qwen2_5_7b_json_28_repair2_shard1, results/swe_generated_patch_kvcomm/qwen2_5_7b_json_28_repair2_shard2, results/swe_generated_patch_kvcomm/qwen2_5_7b_json_28_repair2_shard3`

| run | mode | n | diff extracted | apply ok | pass@1 | JSON parse fail | search not found | repaired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| paired28 | lossless | 28 | 14 | 14 | 3 (0.107) | 4 | 10 | 14 |
| paired28 | lossy | 28 | 12 | 12 | 2 (0.071) | 4 | 9 | 16 |
| repair2_s0 | lossless | 7 | 6 | 6 | 0 (0.000) | 0 | 1 | 1 |
| repair2_s0 | lossy | 7 | 6 | 6 | 0 (0.000) | 0 | 1 | 1 |
| repair2_s0 | lossy_prefetch | 7 | 6 | 6 | 0 (0.000) | 0 | 1 | 1 |
| repair2_s1 | lossless | 7 | 2 | 2 | 0 (0.000) | 0 | 4 | 4 |
| repair2_s1 | lossy | 7 | 2 | 2 | 0 (0.000) | 0 | 4 | 4 |
| repair2_s1 | lossy_prefetch | 7 | 2 | 2 | 0 (0.000) | 0 | 4 | 4 |
| repair2_s2 | lossless | 7 | 5 | 5 | 0 (0.000) | 0 | 2 | 3 |
| repair2_s2 | lossy | 7 | 4 | 4 | 0 (0.000) | 1 | 2 | 3 |
| repair2_s2 | lossy_prefetch | 7 | 4 | 4 | 0 (0.000) | 1 | 2 | 3 |
| repair2_s3 | lossless | 7 | 5 | 5 | 2 (0.286) | 0 | 2 | 2 |
| repair2_s3 | lossy | 7 | 5 | 5 | 2 (0.286) | 0 | 2 | 2 |
| repair2_s3 | lossy_prefetch | 7 | 5 | 5 | 1 (0.143) | 0 | 2 | 2 |

## Per-Run Failure Cases

- `paired28 / lossless`: astropy__astropy-12907:test_failed;astropy__astropy-13033:search_not_found;astropy__astropy-13236:search_not_found;django__django-10554:test_failed;django__django-10880:search_not_found;matplotlib__matplotlib-13989:test_failed;matplotlib__matplotlib-14623:search_not_found;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:test_failed;psf__requests-1724:search_not_found;psf__requests-1766:json_parse_failed;pydata__xarray-2905:test_failed;pydata__xarray-3095:search_not_found;pydata__xarray-3151:json_parse_failed;pylint-dev__pylint-4551:json_parse_failed;pylint-dev__pylint-4604:search_not_found;pylint-dev__pylint-4661:test_failed;pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10356:json_parse_failed;scikit-learn__scikit-learn-10297:test_failed;scikit-learn__scikit-learn-10908:search_not_found;sphinx-doc__sphinx-10323:test_failed;sphinx-doc__sphinx-10449:test_failed
- `paired28 / lossy`: astropy__astropy-12907:search_not_found;astropy__astropy-13033:json_parse_failed;astropy__astropy-13236:search_not_found;django__django-10554:test_failed;django__django-10880:search_not_found;matplotlib__matplotlib-13989:test_failed;matplotlib__matplotlib-14623:test_failed;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:test_failed;psf__requests-1724:search_not_found;psf__requests-1766:json_parse_failed;pydata__xarray-2905:other_synthesis_error;pydata__xarray-3095:search_not_found;pydata__xarray-3151:json_parse_failed;pylint-dev__pylint-4551:no_json_object;pylint-dev__pylint-4604:search_not_found;pylint-dev__pylint-4661:test_failed;pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10356:json_parse_failed;scikit-learn__scikit-learn-10297:test_failed;scikit-learn__scikit-learn-10844:file_not_found;scikit-learn__scikit-learn-10908:search_not_found;sphinx-doc__sphinx-10323:test_failed;sphinx-doc__sphinx-10449:test_failed
- `repair2_s0 / lossless`: astropy__astropy-12907:test_failed;astropy__astropy-13033:test_failed;astropy__astropy-13236:search_not_found;django__django-10097:test_failed;django__django-10554:test_failed;django__django-10880:test_failed;matplotlib__matplotlib-13989:test_failed
- `repair2_s0 / lossy`: astropy__astropy-12907:test_failed;astropy__astropy-13033:test_failed;astropy__astropy-13236:search_not_found;django__django-10097:test_failed;django__django-10554:test_failed;django__django-10880:test_failed;matplotlib__matplotlib-13989:test_failed
- `repair2_s0 / lossy_prefetch`: astropy__astropy-12907:test_failed;astropy__astropy-13033:test_failed;astropy__astropy-13236:search_not_found;django__django-10097:test_failed;django__django-10554:test_failed;django__django-10880:test_failed;matplotlib__matplotlib-13989:test_failed
- `repair2_s1 / lossless`: matplotlib__matplotlib-14623:test_failed;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:search_not_found;psf__requests-1142:generation_error;psf__requests-1724:search_not_found
- `repair2_s1 / lossy`: matplotlib__matplotlib-14623:test_failed;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:search_not_found;psf__requests-1142:generation_error;psf__requests-1724:search_not_found
- `repair2_s1 / lossy_prefetch`: matplotlib__matplotlib-14623:test_failed;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:search_not_found;psf__requests-1142:generation_error;psf__requests-1724:search_not_found
- `repair2_s2 / lossless`: psf__requests-1766:test_failed;pydata__xarray-2905:test_failed;pydata__xarray-3095:test_failed;pydata__xarray-3151:search_not_found;pylint-dev__pylint-4551:search_not_found;pylint-dev__pylint-4604:test_failed;pylint-dev__pylint-4661:test_failed
- `repair2_s2 / lossy`: psf__requests-1766:json_parse_failed;pydata__xarray-2905:test_failed;pydata__xarray-3095:test_failed;pydata__xarray-3151:search_not_found;pylint-dev__pylint-4551:search_not_found;pylint-dev__pylint-4604:test_failed;pylint-dev__pylint-4661:test_failed
- `repair2_s2 / lossy_prefetch`: psf__requests-1766:json_parse_failed;pydata__xarray-2905:test_failed;pydata__xarray-3095:test_failed;pydata__xarray-3151:search_not_found;pylint-dev__pylint-4551:search_not_found;pylint-dev__pylint-4604:test_failed;pylint-dev__pylint-4661:test_failed
- `repair2_s3 / lossless`: pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10356:test_failed;scikit-learn__scikit-learn-10297:search_not_found;scikit-learn__scikit-learn-10908:test_failed;sphinx-doc__sphinx-10323:search_not_found
- `repair2_s3 / lossy`: pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10356:test_failed;scikit-learn__scikit-learn-10297:search_not_found;scikit-learn__scikit-learn-10908:test_failed;sphinx-doc__sphinx-10323:search_not_found
- `repair2_s3 / lossy_prefetch`: pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10081:test_failed;pytest-dev__pytest-10356:test_failed;scikit-learn__scikit-learn-10297:search_not_found;scikit-learn__scikit-learn-10908:test_failed;sphinx-doc__sphinx-10323:search_not_found
