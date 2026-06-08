# Repo-Level 100 Reuse/Scalability Report

## Summary

- Manifest cases: 100
- Repos: 10
- Code segments/files: 300
- Total lines: 1,019,831
- Total chars: 37,926,017
- Approx reusable tokens: 9,481,493
- Median file tokens: 26,412
- P90 file tokens: 46,450
- Exact-content gate false accepts by construction: 0
- Duplicate content signatures across cases: 49 signatures / 123 case references
- CSV: `100_gate_anchor_stats.csv`

## Figures

![Repo distribution](/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/100_repo_distribution.png)

![Token length histogram](/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/100_token_length_hist.png)

![Gate acceptance](/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/100_gate_acceptance.png)

## Repo Distribution

| repo | cases | segments |
|---|---:|---:|
| astropy/astropy | 15 | 45 |
| django/django | 15 | 45 |
| matplotlib/matplotlib | 15 | 45 |
| mwaskom/seaborn | 2 | 6 |
| pallets/flask | 1 | 3 |
| psf/requests | 8 | 24 |
| pydata/xarray | 15 | 45 |
| pylint-dev/pylint | 10 | 30 |
| pytest-dev/pytest | 15 | 45 |
| scikit-learn/scikit-learn | 4 | 12 |

## Gate Interpretation

AST/anchor is treated only as a locator. The reusable segment is admitted only when the full code content signature is identical, so near matches, same AST shape, same file path, or same function name do not pass the reuse gate.
