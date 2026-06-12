# Code Graph Output Drift Diagnostic

## 1. Setup

- Model: `/home/gfy/models/Qwen2.5-3B-Instruct`
- Baseline: `ast_function_only` output for the same target and role
- Candidates: `import_dependency_bundle`, `call_neighborhood_1hop`, `test_target_bundle`
- Targets: 12
- Pairs: 108
- Generation: deterministic, max_new_tokens=96

## 2. By Candidate Bundle

| candidate bundle | n | mean token F1 | JSON valid | reuse-risk match | high-risk drift |
|---|---:|---:|---:|---:|---:|
| `call_neighborhood_1hop` | 36 | 0.786 | 0.89 | 0.86 | 0.83 |
| `import_dependency_bundle` | 36 | 0.745 | 0.81 | 0.61 | 0.94 |
| `test_target_bundle` | 36 | 0.766 | 0.94 | 0.69 | 0.86 |

## 3. Failure Modes

| failure mode | n |
|---|---:|
| `format_failure` | 13 |
| `missing_context_drift` | 9 |
| `ok` | 6 |
| `reuse_risk_drift` | 17 |
| `wrong_or_missing_symbol` | 63 |

## 4. Interpretation

This diagnostic measures whether graph-aware exact bundles change the model's JSON risk judgment relative to the minimal exact target-span baseline. It is not a runtime KV-reuse pass@1 result. Use it to choose which bundle policies deserve paired SWE pass@1 evaluation.
