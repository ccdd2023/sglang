# Expanded 30 SWE-bench Local Env Report

## Summary

- Dataset: `results/repo_level_datasets/swe_verified_30_instances.json`
- Manifest: `results/repo_level_datasets/manifest_30.json`
- Gold smoke: 29/30 pass
- Base smoke: 1/30 pass, 29/30 nonzero
- Discriminative cases: 28/30 (gold pass and base nonzero)
- Filtered dataset: `results/swebench_local_envs/expanded_30_discriminative_instances.json`
- CSV table: `results/swebench_local_envs/expanded_30_env_table.csv`

## Repo Distribution

| repo | manifest cases | discriminative cases |
|---|---:|---:|
| astropy/astropy | 3 | 3 |
| django/django | 3 | 2 |
| matplotlib/matplotlib | 3 | 3 |
| mwaskom/seaborn | 2 | 2 |
| pallets/flask | 1 | 1 |
| psf/requests | 3 | 3 |
| pydata/xarray | 3 | 3 |
| pylint-dev/pylint | 3 | 3 |
| pytest-dev/pytest | 3 | 3 |
| scikit-learn/scikit-learn | 3 | 3 |
| sphinx-doc/sphinx | 3 | 2 |

## Failure Mode Summary

| base failure mode | count |
|---|---:|
| collection/import | 3 |
| pass | 1 |
| test assertion | 26 |

## Per-Case Table

| instance_id | repo | gold | base | discriminative | base failure mode |
|---|---|---:|---:|---|---|
| astropy__astropy-12907 | astropy/astropy | 0 | 1 | true | test assertion |
| astropy__astropy-13033 | astropy/astropy | 0 | 1 | true | test assertion |
| astropy__astropy-13236 | astropy/astropy | 0 | 1 | true | test assertion |
| django__django-10097 | django/django | 0 | 0 | false | pass |
| django__django-10554 | django/django | 0 | 1 | true | test assertion |
| django__django-10880 | django/django | 0 | 1 | true | test assertion |
| matplotlib__matplotlib-13989 | matplotlib/matplotlib | 0 | 1 | true | test assertion |
| matplotlib__matplotlib-14623 | matplotlib/matplotlib | 0 | 1 | true | test assertion |
| matplotlib__matplotlib-20488 | matplotlib/matplotlib | 0 | 1 | true | test assertion |
| mwaskom__seaborn-3069 | mwaskom/seaborn | 0 | 1 | true | test assertion |
| mwaskom__seaborn-3187 | mwaskom/seaborn | 0 | 1 | true | test assertion |
| pallets__flask-5014 | pallets/flask | 0 | 1 | true | test assertion |
| psf__requests-1142 | psf/requests | 0 | 1 | true | test assertion |
| psf__requests-1724 | psf/requests | 0 | 1 | true | test assertion |
| psf__requests-1766 | psf/requests | 0 | 1 | true | test assertion |
| pydata__xarray-2905 | pydata/xarray | 0 | 1 | true | test assertion |
| pydata__xarray-3095 | pydata/xarray | 0 | 1 | true | test assertion |
| pydata__xarray-3151 | pydata/xarray | 0 | 1 | true | test assertion |
| pylint-dev__pylint-4551 | pylint-dev/pylint | 0 | 4 | true | collection/import |
| pylint-dev__pylint-4604 | pylint-dev/pylint | 0 | 4 | true | collection/import |
| pylint-dev__pylint-4661 | pylint-dev/pylint | 0 | 4 | true | collection/import |
| pytest-dev__pytest-10051 | pytest-dev/pytest | 0 | 1 | true | test assertion |
| pytest-dev__pytest-10081 | pytest-dev/pytest | 0 | 1 | true | test assertion |
| pytest-dev__pytest-10356 | pytest-dev/pytest | 0 | 1 | true | test assertion |
| scikit-learn__scikit-learn-10297 | scikit-learn/scikit-learn | 0 | 1 | true | test assertion |
| scikit-learn__scikit-learn-10844 | scikit-learn/scikit-learn | 0 | 1 | true | test assertion |
| scikit-learn__scikit-learn-10908 | scikit-learn/scikit-learn | 0 | 1 | true | test assertion |
| sphinx-doc__sphinx-10323 | sphinx-doc/sphinx | 0 | 1 | true | test assertion |
| sphinx-doc__sphinx-10435 | sphinx-doc/sphinx | 1 | 1 | false | test assertion |
| sphinx-doc__sphinx-10449 | sphinx-doc/sphinx | 0 | 1 | true | test assertion |

## Notes

- `sphinx-doc__sphinx-10435` is excluded from pass@1 main comparison because gold smoke failed locally.
- `django__django-10097` is excluded because base also passed, so it is not discriminative.
- Pylint base failures are collection/import style nonzero outcomes after applying the SWE-bench test patch; they are kept in the environment table but should be inspected separately if used for pass@1.
