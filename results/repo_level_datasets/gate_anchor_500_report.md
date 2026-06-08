# Gate/Anchor 500 Scalability Report

## Summary

- Manifest cases: 500
- Repos: 12
- Code segments/files: 1500
- Total lines: 5,973,340
- Total chars: 235,090,023
- Approx reusable tokens: 58,772,478
- Median file tokens: 27,416
- P90 file tokens: 75,581
- Exact-content gate false accepts by construction: 0
- Duplicate content signatures across cases: 254 signatures / 1073 case references
- CSV: `500_gate_anchor_stats.csv`

## Figures

![Repo distribution](/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/500_repo_distribution.png)

![Token length histogram](/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/500_token_length_hist.png)

![Gate acceptance](/home/gfy/CodeMAS_Project/sglang-kvflow/results/repo_level_datasets/500_gate_acceptance.png)

## Repo Distribution

| repo | cases | segments |
|---|---:|---:|
| astropy/astropy | 22 | 66 |
| django/django | 231 | 693 |
| matplotlib/matplotlib | 34 | 102 |
| mwaskom/seaborn | 2 | 6 |
| pallets/flask | 1 | 3 |
| psf/requests | 8 | 24 |
| pydata/xarray | 22 | 66 |
| pylint-dev/pylint | 10 | 30 |
| pytest-dev/pytest | 19 | 57 |
| scikit-learn/scikit-learn | 32 | 96 |
| sphinx-doc/sphinx | 44 | 132 |
| sympy/sympy | 75 | 225 |

## Gate Interpretation

AST/anchor is treated only as a locator. The reusable segment is admitted only when the full code content signature is identical, so near matches, same AST shape, same file path, or same function name do not pass the reuse gate.
