# Code Graph Lossy-Reuse Precision KV Diagnostic

## 1. What Was Run

- Model: `/home/gfy/models/Qwen2.5-3B-Instruct`
- Selected layers: `(-1, -2, -3, -4)`
- Sampled exact bundle groups: 8
- Records: 16 = sampled groups × coder/reviewer comparisons
- Canonical comparison: same exact bundle under `coder`/`reviewer` prompt vs `planner` prompt
- Scope tokens are recorded only as covariates, not as the optimization target.

## 2. By Bundle Type

| bundle | n | mean d_norm | p50 | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|---:|
| `ast_function_only` | 4 | 0.188 | 0.194 | 0.200 | 0.00 |
| `call_neighborhood_1hop` | 4 | 0.178 | 0.179 | 0.183 | 0.00 |
| `import_dependency_bundle` | 4 | 0.185 | 0.187 | 0.199 | 0.00 |
| `test_target_bundle` | 4 | 0.186 | 0.192 | 0.202 | 0.00 |

## 3. By Code Task Family

| task family | n | mean d_norm | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|
| `model_or_query_logic` | 6 | 0.190 | 0.200 | 0.00 |
| `test_aligned` | 4 | 0.186 | 0.202 | 0.00 |
| `validation` | 6 | 0.178 | 0.182 | 0.00 |

## 4. Worst Precision-Risk Cases

| case | task | bundle | role | d_norm | symbol |
|---|---|---|---|---:|---|
| `astropy__astropy-12907` | `test_aligned` | `test_target_bundle` | `reviewer` | 0.202 | `_cstack` |
| `astropy__astropy-12907` | `model_or_query_logic` | `ast_function_only` | `reviewer` | 0.200 | `_cstack` |
| `astropy__astropy-12907` | `model_or_query_logic` | `import_dependency_bundle` | `reviewer` | 0.199 | `_cstack` |
| `astropy__astropy-12907` | `model_or_query_logic` | `ast_function_only` | `coder` | 0.194 | `_cstack` |
| `astropy__astropy-12907` | `test_aligned` | `test_target_bundle` | `coder` | 0.192 | `_cstack` |
| `astropy__astropy-12907` | `model_or_query_logic` | `import_dependency_bundle` | `coder` | 0.187 | `_cstack` |
| `astropy__astropy-12907` | `model_or_query_logic` | `call_neighborhood_1hop` | `coder` | 0.183 | `_cstack` |
| `django__django-10097` | `validation` | `import_dependency_bundle` | `reviewer` | 0.182 | `URLValidator` |

## 5. Interpretation Boundary

This is a KV precision diagnostic, not a TTFT or pass@1 result. A lower cross-role `d_norm` suggests the exact code bundle is more stable under lossy reuse across agent prompts. The next confirmation step is output drift and paired pass@1 non-degradation on the same sampled groups.
