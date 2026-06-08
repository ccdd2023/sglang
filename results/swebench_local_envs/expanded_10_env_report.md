# Expanded SWE-bench Repo Dataset and Local Env Smoke Report

Generated on 2026-06-02. This report records the expanded real repo-level dataset manifest and local non-Docker SWE-bench smoke tests for KVCOMM contribution-3 evaluation.

## Artifacts

- Repo-level large-code manifest: `results/repo_level_datasets/manifest_10.json`
- SWE-bench instance subset: `results/repo_level_datasets/swe_verified_10_instances.json`
- Gold smoke JSON: `results/swebench_local_envs/expanded_10_gold_smoke.json`
- Base smoke JSON: `results/swebench_local_envs/expanded_10_base_smoke.json`

## Dataset Manifest

- Source code snapshots: `ScalingIntelligence/swe-bench-verified-codebase-content`
- SWE-bench metadata: `princeton-nlp/SWE-bench_Verified`
- Cases: 10
- Files per case: up to 3 large Python files

| # | Instance | Repo | Version | Large files | Total lines | Total chars |
|---:|---|---|---|---:|---:|---:|
| 1 | `astropy__astropy-12907` | `astropy/astropy` | `4.3` | 3 | 12326 | 458107 |
| 2 | `django__django-10097` | `django/django` | `2.2` | 3 | 6726 | 274600 |
| 3 | `matplotlib__matplotlib-13989` | `matplotlib/matplotlib` | `3.0` | 3 | 17185 | 619066 |
| 4 | `mwaskom__seaborn-3069` | `mwaskom/seaborn` | `0.12` | 3 | 8488 | 304404 |
| 5 | `pallets__flask-5014` | `pallets/flask` | `2.3` | 3 | 4203 | 156986 |
| 6 | `psf__requests-1142` | `psf/requests` | `1.1` | 3 | 2178 | 149361 |
| 7 | `pydata__xarray-2905` | `pydata/xarray` | `0.12` | 3 | 14154 | 523417 |
| 8 | `pylint-dev__pylint-4551` | `pylint-dev/pylint` | `2.9` | 3 | 7078 | 272492 |
| 9 | `pytest-dev__pytest-10051` | `pytest-dev/pytest` | `7.2` | 3 | 8260 | 272935 |
| 10 | `scikit-learn__scikit-learn-10297` | `scikit-learn/scikit-learn` | `0.20` | 3 | 7480 | 278684 |

## Environment Builder Changes

- Added a 10-case manifest builder: `benchmark/multi_workflow/prepare_swebench_verified_expanded.py`.
- Added a batch local environment runner: `benchmark/multi_workflow/run_swebench_local_env_batch.py`.
- Updated local SWE-bench setup to install missing pytest runners when specs assume pytest is already present.
- Pinned old scikit-learn builds to `Cython<3`, which fixes Cython 3 incompatibility for `scikit-learn==0.20`.
- Skipped Docker/root-only pre-install commands such as `apt-get`, `/etc/locale.gen`, `locale-gen`, and Matplotlib QHULL source-build snippets in the local fallback; Matplotlib native deps are supplied through conda.

## Smoke Test Summary

- Gold smoke: 10/10 passed.
- Base smoke: 1/10 passed, 9/10 nonzero.
- Discriminative smoke cases (base nonzero, gold pass): 9/10.
- Django `django__django-10097` is environment-valid but non-discriminative under `--max-fail-tests 1`; its first selected FAIL_TO_PASS test passes on base too. Use full FAIL_TO_PASS or replace the selected target before treating it as an accuracy case.
- Pylint base returns pytest collection error after applying the test patch; gold passes. Count it as base-nonzero smoke evidence, but inspect before using as a clean pass@1 case.

| # | Instance | Repo | Test target | Base | Gold | Discriminative | Base sec | Gold sec |
|---:|---|---|---|---|---|---|---:|---:|
| 1 | `astropy__astropy-12907` | `astropy/astropy` | `astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]` | fail (1) | pass | yes | 74.65 | 76.66 |
| 2 | `django__django-10097` | `django/django` | `test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)` | pass | pass | no | 14.22 | 13.62 |
| 3 | `matplotlib__matplotlib-13989` | `matplotlib/matplotlib` | `lib/matplotlib/tests/test_axes.py::test_hist_range_and_density` | fail (1) | pass | yes | 59.57 | 62.27 |
| 4 | `mwaskom__seaborn-3069` | `mwaskom/seaborn` | `tests/_core/test_plot.py::TestScaling::test_nominal_x_axis_tweaks` | fail (1) | pass | yes | 55.2 | 14.43 |
| 5 | `pallets__flask-5014` | `pallets/flask` | `tests/test_blueprints.py::test_empty_name_not_allowed` | fail (1) | pass | yes | 27.78 | 14.57 |
| 6 | `psf__requests-1142` | `psf/requests` | `test_requests.py::RequestsTestCase::test_no_content_length` | fail (1) | pass | yes | 20.79 | 10.04 |
| 7 | `pydata__xarray-2905` | `pydata/xarray` | `xarray/tests/test_variable.py::TestAsCompatibleData::test_unsupported_type` | fail (1) | pass | yes | 40.17 | 20.1 |
| 8 | `pylint-dev__pylint-4551` | `pylint-dev/pylint` | `tests/unittest_pyreverse_writer.py::test_dot_files[packages_No_Name.dot]` | fail (4) | pass | yes | 28.88 | 14.39 |
| 9 | `pytest-dev__pytest-10051` | `pytest-dev/pytest` | `testing/logging/test_fixture.py::test_clear_for_call_stage` | fail (1) | pass | yes | 29.83 | 16.13 |
| 10 | `scikit-learn__scikit-learn-10297` | `scikit-learn/scikit-learn` | `sklearn/linear_model/tests/test_ridge.py::test_ridge_classifier_cv_store_cv_values` | fail (1) | pass | yes | 173.22 | 155.34 |

## Recommended Use for KVCOMM Experiments

- Use all 10 cases for repo-level exact code reuse and large code-base prompt construction, because all gold environments build and run.
- Use the 9 discriminative cases for first-pass SWE-bench smoke accuracy, excluding Django unless full FAIL_TO_PASS is rerun.
- Prefer the clean assertion-failure cases for a first paper table: Astropy, Matplotlib, Seaborn, Flask, Requests, Xarray, Pytest, scikit-learn.
- Treat Pylint as secondary until the base collection behavior is normalized.

## Next Commands

```bash
# Rebuild the 10-case manifest
/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/prepare_swebench_verified_expanded.py --max-cases 10 --max-files 3 --label 10

# Re-run gold smoke
/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/run_swebench_local_env_batch.py --dataset results/repo_level_datasets/swe_verified_10_instances.json --out results/swebench_local_envs/expanded_10_gold_smoke.json --mode gold --max-cases 10 --max-fail-tests 1 --timeout 1200 --skip-existing-pass

# Re-run base smoke
/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/run_swebench_local_env_batch.py --dataset results/repo_level_datasets/swe_verified_10_instances.json --out results/swebench_local_envs/expanded_10_base_smoke.json --mode base --max-cases 10 --max-fail-tests 1 --timeout 1200 --skip-existing-pass
```
