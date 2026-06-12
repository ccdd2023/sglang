# Code Graph Output Drift Diagnostic

## 1. Setup

- Model: `/home/gfy/models/Qwen2.5-3B-Instruct`
- Baseline: `ast_function_only` output for the same target and role
- Candidates: `import_dependency_bundle`, `call_neighborhood_1hop`, `test_target_bundle`
- Targets: 2
- Pairs: 15
- Generation: deterministic, max_new_tokens=96

## 2. By Candidate Bundle

| candidate bundle | n | mean token F1 | JSON valid | reuse-risk match | high-risk drift |
|---|---:|---:|---:|---:|---:|
| `call_neighborhood_1hop` | 3 | 0.729 | 1.00 | 1.00 | 1.00 |
| `import_dependency_bundle` | 6 | 0.849 | 1.00 | 0.83 | 1.00 |
| `test_target_bundle` | 6 | 0.832 | 1.00 | 0.67 | 0.50 |

## 3. Failure Modes

| failure mode | n |
|---|---:|
| `missing_context_drift` | 3 |
| `ok` | 2 |
| `reuse_risk_drift` | 3 |
| `wrong_or_missing_symbol` | 7 |

## 4. Interpretation

This diagnostic measures whether graph-aware exact bundles change the model's JSON risk judgment relative to the minimal exact target-span baseline. It is not a runtime KV-reuse pass@1 result. Use it to choose which bundle policies deserve paired SWE pass@1 evaluation.
