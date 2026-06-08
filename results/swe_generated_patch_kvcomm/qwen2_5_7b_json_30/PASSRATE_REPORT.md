# Qwen2.5-7B JSON-Edit 30-Case Pass@1 Report

## Summary

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/swebench_local_envs/expanded_30_discriminative_instances.json`
- Cases: 28 discriminative cases (gold pass and base nonzero from expanded_30 smoke)
- Lossless pass@1: 3/28
- KVCOMM lossy pass@1: 2/28
- Pass@1 delta: -1 cases
- Lossy exact-content hit rate: 28/28
- Avg cached tokens: lossless 1253.3, lossy 2190.2
- Avg generation latency: lossless 2052.1 ms, lossy 1729.4 ms
- Generation speedup by avg latency: 1.19x

## Main Table

| mode | diff extracted | clean apply | pass@1 | avg cached tokens | avg latency ms |
|---|---:|---:|---:|---:|---:|
| lossless | 14/28 | 14/28 | 3/28 | 1253.3 | 2052.1 |
| lossy | 12/28 | 12/28 | 2/28 | 2190.2 | 1729.4 |

## Figures

![Passrate pipeline](/home/gfy/CodeMAS_Project/sglang-kvflow/results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/fig_passrate_pipeline.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/fig_cached_tokens.png)

![Latency](/home/gfy/CodeMAS_Project/sglang-kvflow/results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/fig_latency.png)

## Per-Case Results

| instance_id | repo | lossless apply/pass | lossy apply/pass | lossy match reason | lossy cached |
|---|---|---|---|---|---:|
| astropy__astropy-12907 | astropy/astropy | True/False | False/False | exact_code_content_signature | 3515 |
| astropy__astropy-13033 | astropy/astropy | False/False | False/False | exact_code_content_signature | 1925 |
| astropy__astropy-13236 | astropy/astropy | False/False | False/False | exact_code_content_signature | 2055 |
| django__django-10554 | django/django | True/False | True/False | exact_code_content_signature | 1706 |
| django__django-10880 | django/django | False/False | False/False | exact_code_content_signature | 601 |
| matplotlib__matplotlib-13989 | matplotlib/matplotlib | True/False | True/False | exact_code_content_signature | 939 |
| matplotlib__matplotlib-14623 | matplotlib/matplotlib | False/False | True/False | exact_code_content_signature | 832 |
| matplotlib__matplotlib-20488 | matplotlib/matplotlib | False/False | False/False | exact_code_content_signature | 1294 |
| mwaskom__seaborn-3069 | mwaskom/seaborn | False/False | False/False | exact_code_content_signature | 934 |
| mwaskom__seaborn-3187 | mwaskom/seaborn | True/False | True/False | exact_code_content_signature | 1102 |
| pallets__flask-5014 | pallets/flask | True/False | True/False | exact_code_content_signature | 472 |
| psf__requests-1142 | psf/requests | True/True | True/True | exact_code_content_signature | 539 |
| psf__requests-1724 | psf/requests | False/False | False/False | exact_code_content_signature | 2679 |
| psf__requests-1766 | psf/requests | False/False | False/False | exact_code_content_signature | 748 |
| pydata__xarray-2905 | pydata/xarray | True/False | False/False | exact_code_content_signature | 7756 |
| pydata__xarray-3095 | pydata/xarray | False/False | False/False | exact_code_content_signature | 1047 |
| pydata__xarray-3151 | pydata/xarray | False/False | False/False | exact_code_content_signature | 1403 |
| pylint-dev__pylint-4551 | pylint-dev/pylint | False/False | False/False | exact_code_content_signature | 3586 |
| pylint-dev__pylint-4604 | pylint-dev/pylint | False/False | False/False | exact_code_content_signature | 8435 |
| pylint-dev__pylint-4661 | pylint-dev/pylint | True/False | True/False | exact_code_content_signature | 708 |
| pytest-dev__pytest-10051 | pytest-dev/pytest | True/False | True/False | exact_code_content_signature | 1125 |
| pytest-dev__pytest-10081 | pytest-dev/pytest | True/True | True/True | exact_code_content_signature | 1730 |
| pytest-dev__pytest-10356 | pytest-dev/pytest | False/False | False/False | exact_code_content_signature | 1757 |
| scikit-learn__scikit-learn-10297 | scikit-learn/scikit-learn | True/False | True/False | exact_code_content_signature | 1460 |
| scikit-learn__scikit-learn-10844 | scikit-learn/scikit-learn | True/True | False/False | exact_code_content_signature | 8863 |
| scikit-learn__scikit-learn-10908 | scikit-learn/scikit-learn | False/False | False/False | exact_code_content_signature | 1236 |
| sphinx-doc__sphinx-10323 | sphinx-doc/sphinx | True/False | True/False | exact_code_content_signature | 1352 |
| sphinx-doc__sphinx-10449 | sphinx-doc/sphinx | True/False | True/False | exact_code_content_signature | 1526 |

## Interpretation

All lossy reuse hits that were admitted by KVCOMM used `exact_code_content_signature`, consistent with the safety rule that AST/anchor only locates candidate code and exact content gates reuse. Absolute pass@1 remains limited by patch synthesis quality; the main contribution-3 accuracy claim should focus on lossless-vs-lossy delta under the same model and schema.
