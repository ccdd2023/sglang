# Phase 2.1 per-case pass@1 summary

baseline: `results/swe_percase_baseline_20260624T085604Z`
v44:      `results/swe_percase_v44_20260624T085604Z`

Format: <bytes>B[ext,syn,app✓/✗,sim=X.XXX,copy=method]

## Per-case × per-mode table

| case_id | repo | lossless_base | lossless_v44 | eq | lossy_base | lossy_v44 | eq | lossy_prefetch_base | lossy_prefetch_v44 | eq | placeholder_knn_lossy_base | placeholder_knn_lossy_v44 | eq |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| astropy__astropy-12907 |  | 3868B[ext,syn,app✗] | - | - | 2239B[ext,syn,app✗] | - | - | 2239B[ext,syn,app✗] | - | - | - | - | - |
| django__django-10097 |  | 2433B[ext,syn,app✗] | - | - | 1823B[ext,syn,app✗] | - | - | 1823B[ext,syn,app✗] | - | - | - | - | - |
| matplotlib__matplotlib-13989 |  | 4330B[ext,syn,app✗] | - | - | 1241B[ext,syn] | - | - | 1241B[ext,syn] | - | - | - | - | - |
| mwaskom__seaborn-3069 |  | 1061B[ext,syn,app✗] | - | - | 3690B[ext,syn,app✗] | - | - | 3690B[ext,syn,app✗] | - | - | - | - | - |
| pallets__flask-5014 |  | 893B[ext,syn,app✗] | - | - | 904B[ext,syn,app✗] | - | - | 904B[ext,syn,app✗] | - | - | - | - | - |
| psf__requests-1142 |  | 438B[ext,syn] | - | - | 431B[ext,syn,app✓] | - | - | 431B[ext,syn,app✓] | - | - | - | - | - |
| pydata__xarray-2905 |  | 697B[ext,syn,app✗] | - | - | 779B[ext,syn] | - | - | 779B[ext,syn] | - | - | - | - | - |
| pylint-dev__pylint-4551 |  | 3248B[ext,syn,app✗] | - | - | 3937B[ext,syn,app✗] | - | - | 3937B[ext,syn,app✗] | - | - | - | - | - |
| pytest-dev__pytest-10051 |  | 694B[ext,syn,app✗] | - | - | 694B[ext,syn,app✗] | - | - | 694B[ext,syn,app✗] | - | - | - | - | - |
| scikit-learn__scikit-learn-10297 |  | 3365B[ext,syn,app✗] | - | - | 2767B[ext,syn] | - | - | 2767B[ext,syn] | - | - | - | - | - |

## Aggregate counts

| mode | source | extracted | synth_ok | apply_pass | candidate_pass |
|---|---|---|---|---|---|
| lossless | baseline | 10/10 | 10/10 | 0/10 | 0/10 |
| lossless | v44 | 0/0 | 0/0 | 0/0 | 0/0 |
| lossy | baseline | 10/10 | 10/10 | 1/10 | 0/10 |
| lossy | v44 | 0/0 | 0/0 | 0/0 | 0/0 |
| lossy_prefetch | baseline | 10/10 | 10/10 | 1/10 | 0/10 |
| lossy_prefetch | v44 | 0/0 | 0/0 | 0/0 | 0/0 |
| placeholder_knn_lossy | baseline | 0/0 | 0/0 | 0/0 | 0/0 |
| placeholder_knn_lossy | v44 | 0/0 | 0/0 | 0/0 | 0/0 |

## Byte-equality: baseline patch vs v44 patch

| case_id | mode | baseline | v44 | equal |
|---|---|---|---|---|