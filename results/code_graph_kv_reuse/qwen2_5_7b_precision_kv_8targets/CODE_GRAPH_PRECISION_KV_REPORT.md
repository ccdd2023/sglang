# Code Graph Lossy-Reuse Precision KV Diagnostic

## 1. What Was Run

- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Selected layers: `(-1, -2, -3, -4)`
- Sampled exact bundle groups: 31
- Records: 62 = sampled groups × coder/reviewer comparisons
- Canonical comparison: same exact bundle under `coder`/`reviewer` prompt vs `planner` prompt
- Scope tokens are recorded only as covariates, not as the optimization target.

## 2. By Bundle Type

| bundle | n | mean d_norm | p50 | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|---:|
| `ast_function_only` | 16 | 0.283 | 0.279 | 0.356 | 0.00 |
| `call_neighborhood_1hop` | 14 | 0.270 | 0.268 | 0.369 | 0.00 |
| `import_dependency_bundle` | 16 | 0.270 | 0.278 | 0.320 | 0.00 |
| `test_target_bundle` | 16 | 0.289 | 0.273 | 0.366 | 0.00 |

## 3. By Code Task Family

| task family | n | mean d_norm | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|
| `general_library_logic` | 6 | 0.292 | 0.324 | 0.00 |
| `model_or_query_logic` | 18 | 0.283 | 0.369 | 0.00 |
| `plotting_rendering` | 12 | 0.238 | 0.256 | 0.00 |
| `test_aligned` | 16 | 0.289 | 0.366 | 0.00 |
| `validation` | 4 | 0.256 | 0.273 | 0.00 |
| `web_http` | 6 | 0.319 | 0.356 | 0.00 |

## 4. Worst Precision-Risk Cases

| case | task | bundle | role | d_norm | symbol |
|---|---|---|---|---:|---|
| `astropy__astropy-12907` | `test_aligned` | `test_target_bundle` | `coder` | 0.394 | `_cstack` |
| `psf__requests-1142` | `model_or_query_logic` | `call_neighborhood_1hop` | `coder` | 0.377 | `PreparedRequest.prepare_content_length` |
| `psf__requests-1142` | `model_or_query_logic` | `ast_function_only` | `reviewer` | 0.369 | `PreparedRequest.prepare_content_length` |
| `psf__requests-1142` | `model_or_query_logic` | `call_neighborhood_1hop` | `reviewer` | 0.369 | `PreparedRequest.prepare_content_length` |
| `astropy__astropy-12907` | `test_aligned` | `test_target_bundle` | `reviewer` | 0.366 | `_cstack` |
| `psf__requests-1142` | `test_aligned` | `test_target_bundle` | `reviewer` | 0.357 | `PreparedRequest.prepare_content_length` |
| `pallets__flask-5014` | `web_http` | `ast_function_only` | `reviewer` | 0.356 | `Blueprint.__init__` |
| `psf__requests-1142` | `test_aligned` | `test_target_bundle` | `coder` | 0.354 | `PreparedRequest.prepare_content_length` |

## 5. Interpretation Boundary

This is a KV precision diagnostic, not a TTFT or pass@1 result. A lower cross-role `d_norm` suggests the exact code bundle is more stable under lossy reuse across agent prompts. The next confirmation step is output drift and paired pass@1 non-degradation on the same sampled groups.
