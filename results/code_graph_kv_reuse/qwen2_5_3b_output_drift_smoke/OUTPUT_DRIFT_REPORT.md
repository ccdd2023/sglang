# Code Graph Output Drift Diagnostic

## 1. Setup

- Model: `/home/gfy/models/Qwen2.5-3B-Instruct`
- Baseline: `ast_function_only` output for the same target and role
- Candidates: `import_dependency_bundle`, `call_neighborhood_1hop`, `test_target_bundle`
- Targets: 2
- Pairs: 15
- Generation: deterministic, max_new_tokens=80

## 2. By Candidate Bundle

| candidate bundle | n | mean token F1 | JSON valid | reuse-risk match | high-risk drift |
|---|---:|---:|---:|---:|---:|
| `call_neighborhood_1hop` | 3 | 0.638 | 1.00 | 0.33 | 1.00 |
| `import_dependency_bundle` | 6 | 0.673 | 0.67 | 0.17 | 0.83 |
| `test_target_bundle` | 6 | 0.733 | 0.50 | 0.33 | 0.67 |

## 3. Failure Modes

| failure mode | n |
|---|---:|
| `format_failure` | 5 |
| `missing_context_drift` | 2 |
| `ok` | 1 |
| `reuse_risk_drift` | 6 |
| `wrong_or_missing_symbol` | 1 |

## 4. Interpretation

This diagnostic measures whether graph-aware exact bundles change the model's JSON risk judgment relative to the minimal exact target-span baseline. It is not a runtime KV-reuse pass@1 result. Use it to choose which bundle policies deserve paired SWE pass@1 evaluation.
