# Pass@1 Repair Sweep Summary

- Git commit: `3d709f3ce`
- Input runs: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30, results/swe_generated_patch_kvcomm/qwen2_5_7b_json_8_forceevict_reretest`

| run | mode | n | diff extracted | apply ok | pass@1 | JSON parse fail | search not found | repaired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| paired28 | lossless | 28 | 14 | 14 | 3 (0.107) | 4 | 10 | 14 |
| paired28 | lossy | 28 | 12 | 12 | 2 (0.071) | 4 | 9 | 16 |
| harder8 | lossless | 8 | 6 | 6 | 0 (0.000) | 0 | 0 | 2 |
| harder8 | lossy | 8 | 6 | 6 | 0 (0.000) | 0 | 0 | 2 |

## Per-Run Failure Cases

- `paired28 / lossless`: astropy__astropy-12907:test_failed;astropy__astropy-13033:search_not_found;astropy__astropy-13236:search_not_found;django__django-10554:test_failed;django__django-10880:search_not_found;matplotlib__matplotlib-13989:test_failed;matplotlib__matplotlib-14623:search_not_found;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:test_failed;psf__requests-1724:search_not_found;psf__requests-1766:json_parse_failed;pydata__xarray-2905:test_failed;pydata__xarray-3095:search_not_found;pydata__xarray-3151:json_parse_failed;pylint-dev__pylint-4551:json_parse_failed;pylint-dev__pylint-4604:search_not_found;pylint-dev__pylint-4661:test_failed;pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10356:json_parse_failed;scikit-learn__scikit-learn-10297:test_failed;scikit-learn__scikit-learn-10908:search_not_found;sphinx-doc__sphinx-10323:test_failed;sphinx-doc__sphinx-10449:test_failed
- `paired28 / lossy`: astropy__astropy-12907:search_not_found;astropy__astropy-13033:json_parse_failed;astropy__astropy-13236:search_not_found;django__django-10554:test_failed;django__django-10880:search_not_found;matplotlib__matplotlib-13989:test_failed;matplotlib__matplotlib-14623:test_failed;matplotlib__matplotlib-20488:search_not_found;mwaskom__seaborn-3069:search_not_found;mwaskom__seaborn-3187:test_failed;pallets__flask-5014:test_failed;psf__requests-1724:search_not_found;psf__requests-1766:json_parse_failed;pydata__xarray-2905:other_synthesis_error;pydata__xarray-3095:search_not_found;pydata__xarray-3151:json_parse_failed;pylint-dev__pylint-4551:no_json_object;pylint-dev__pylint-4604:search_not_found;pylint-dev__pylint-4661:test_failed;pytest-dev__pytest-10051:test_failed;pytest-dev__pytest-10356:json_parse_failed;scikit-learn__scikit-learn-10297:test_failed;scikit-learn__scikit-learn-10844:file_not_found;scikit-learn__scikit-learn-10908:search_not_found;sphinx-doc__sphinx-10323:test_failed;sphinx-doc__sphinx-10449:test_failed
- `harder8 / lossless`: django__django-11138:other_synthesis_error;django__django-11149:test_failed;matplotlib__matplotlib-20676:other_synthesis_error;matplotlib__matplotlib-20859:test_failed;matplotlib__matplotlib-21568:test_failed;psf__requests-5414:test_failed;psf__requests-6028:test_failed;pylint-dev__pylint-8898:test_failed
- `harder8 / lossy`: django__django-11138:other_synthesis_error;django__django-11149:test_failed;matplotlib__matplotlib-20676:other_synthesis_error;matplotlib__matplotlib-20859:test_failed;matplotlib__matplotlib-21568:test_failed;psf__requests-5414:test_failed;psf__requests-6028:test_failed;pylint-dev__pylint-8898:test_failed
