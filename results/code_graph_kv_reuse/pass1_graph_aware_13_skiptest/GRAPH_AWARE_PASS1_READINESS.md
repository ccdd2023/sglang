# Graph-Aware Lossy Pass@1 Readiness Diagnostic
- Result dir: `results/code_graph_kv_reuse/pass1_graph_aware_13_skiptest`
- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Candidate tests: skipped; this run validates generation, JSON edit synthesis, git apply-check, and reuse metadata.

| mode | n | diff extracted | apply ok | generation errors | search-not-found | exact signature match | mean cached tokens | mean elapsed ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless` | 13 | 4 | 4 | 1 | 6 | 0 | 13687.5 | 3177.6 |
| `lossy` | 13 | 2 | 2 | 1 | 8 | 12 | 9681.1 | 3733.1 |
| `lossy_prefetch` | 13 | 3 | 3 | 1 | 8 | 12 | 9902.0 | 3530.8 |
| `graph_aware_lossy` | 13 | 8 | 8 | 0 | 2 | 13 | 1444.9 | 2822.5 |

## Graph-Aware Case Outcomes

| instance | diff | apply | synthesis error | exact match | cached tokens | graph segments |
|---|---:|---:|---|---:|---:|---:|
| `astropy__astropy-12907` | True | True |  | True | 1843 | 1 |
| `astropy__astropy-13033` | False | False | search not found in astropy/timeseries/core.py | True | 2459 | 3 |
| `astropy__astropy-13236` | True | True |  | True | 0 | 1 |
| `django__django-10554` | True | True |  | True | 8223 | 3 |
| `django__django-10880` | True | True |  | True | 2559 | 2 |
| `matplotlib__matplotlib-13989` | True | True |  | True | 0 | 2 |
| `matplotlib__matplotlib-14623` | True | True |  | True | 0 | 3 |
| `matplotlib__matplotlib-20488` | True | True |  | True | 0 | 2 |
| `mwaskom__seaborn-3069` | False | False | json parse failed: Expecting ',' delimiter: line 1 column 65 (char 64) | True | 0 | 2 |
| `mwaskom__seaborn-3187` | False | False | json parse failed: Invalid \escape: line 1 column 559 (char 558) | True | 0 | 3 |
| `pallets__flask-5014` | False | False | json parse failed: Expecting ',' delimiter: line 1 column 59 (char 58) | True | 0 | 2 |
| `psf__requests-1142` | True | True |  | True | 3700 | 2 |
| `psf__requests-1724` | False | False | search not found in requests/sessions.py | True | 0 | 2 |

## Immediate Reading

- `graph_aware_lossy` reaches exact-content-signature reuse on most cases, so the mechanism is connected to the live patch harness.
- Apply-check success is still not strong enough to claim pass@1 non-degradation; failures are mostly `search not found`, meaning the generated edit refers to context not present exactly in the checked repo file or changed by repair.
- Next pass@1 should be restricted to graph-aware cases with `apply_ok=True`, and the paper should keep code-graph lossy as a precision/failure-mode diagnostic until candidate tests pass.
