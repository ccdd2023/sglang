# Code Graph Lossy-Reuse Precision KV Diagnostic

## 1. What Was Run

- Model: `/home/gfy/models/Qwen2.5-3B-Instruct`
- Selected layers: `(-1, -2, -3, -4)`
- Sampled exact bundle groups: 48
- Records: 96 = sampled groups × coder/reviewer comparisons
- Canonical comparison: same exact bundle under `coder`/`reviewer` prompt vs `planner` prompt
- Scope tokens are recorded only as covariates, not as the optimization target.

## 2. By Bundle Type

| bundle | n | mean d_norm | p50 | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|---:|
| `ast_function_only` | 24 | 0.182 | 0.189 | 0.201 | 0.00 |
| `call_neighborhood_1hop` | 24 | 0.173 | 0.174 | 0.201 | 0.00 |
| `import_dependency_bundle` | 24 | 0.173 | 0.172 | 0.202 | 0.00 |
| `test_target_bundle` | 24 | 0.177 | 0.173 | 0.211 | 0.00 |

## 3. By Code Task Family

| task family | n | mean d_norm | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|
| `general_library_logic` | 12 | 0.186 | 0.201 | 0.00 |
| `model_or_query_logic` | 30 | 0.182 | 0.235 | 0.00 |
| `plotting_rendering` | 18 | 0.154 | 0.166 | 0.00 |
| `test_aligned` | 24 | 0.177 | 0.211 | 0.00 |
| `validation` | 6 | 0.178 | 0.182 | 0.00 |
| `web_http` | 6 | 0.191 | 0.202 | 0.00 |

## 4. Worst Precision-Risk Cases

| case | task | bundle | role | d_norm | symbol |
|---|---|---|---|---:|---|
| `psf__requests-1142` | `model_or_query_logic` | `call_neighborhood_1hop` | `reviewer` | 0.279 | `PreparedRequest.prepare_content_length` |
| `psf__requests-1142` | `model_or_query_logic` | `ast_function_only` | `coder` | 0.273 | `PreparedRequest.prepare_content_length` |
| `psf__requests-1142` | `model_or_query_logic` | `ast_function_only` | `reviewer` | 0.235 | `PreparedRequest.prepare_content_length` |
| `psf__requests-1142` | `test_aligned` | `test_target_bundle` | `reviewer` | 0.221 | `PreparedRequest.prepare_content_length` |
| `psf__requests-1142` | `model_or_query_logic` | `call_neighborhood_1hop` | `coder` | 0.219 | `PreparedRequest.prepare_content_length` |
| `psf__requests-1142` | `test_aligned` | `test_target_bundle` | `coder` | 0.212 | `PreparedRequest.prepare_content_length` |
| `astropy__astropy-13033` | `test_aligned` | `test_target_bundle` | `reviewer` | 0.211 | `BaseTimeSeries._delay_required_column_checks` |
| `astropy__astropy-13033` | `test_aligned` | `test_target_bundle` | `coder` | 0.204 | `BaseTimeSeries._delay_required_column_checks` |

## 5. Interpretation Boundary

This is a KV precision diagnostic, not a TTFT or pass@1 result. A lower cross-role `d_norm` suggests the exact code bundle is more stable under lossy reuse across agent prompts. The next confirmation step is output drift and paired pass@1 non-degradation on the same sampled groups.
