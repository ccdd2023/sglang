# Code Graph Lossy Reuse Precision Report

## 1. Evidence Stack

这份报告把 code-specific lossy reuse 的精度证据分成三层：

1. **KV stability**：同一个 exact code bundle 在 planner/coder/reviewer prompt 下的 KV 表示是否稳定。
2. **Output drift**：graph-aware bundle 是否改变模型的 JSON risk judgment。
3. **Pass@1 readiness**：已有 pass@1 case 中有多少能映射到 code graph bundle，下一步如何跑 graph-aware lossy。

重要边界：这里仍然只主张 non-degradation readiness，不主张 accuracy improvement。

## 2. KV Stability

### 3B cross-task diagnostic

- Result: `results/code_graph_kv_reuse/qwen2_5_3b_precision_kv_12targets/`
- Records: 96; sampled bundle groups: 48
- Overall mean/p90/max d_norm: 0.176 / 0.202 / 0.279
- Tail `d_norm>0.5`: 0.00

| bundle | n | mean d_norm | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|
| `ast_function_only` | 24 | 0.182 | 0.201 | 0.00 |
| `call_neighborhood_1hop` | 24 | 0.173 | 0.201 | 0.00 |
| `import_dependency_bundle` | 24 | 0.173 | 0.202 | 0.00 |
| `test_target_bundle` | 24 | 0.177 | 0.211 | 0.00 |

### 7B robustness sanity

- Result: `results/code_graph_kv_reuse/qwen2_5_7b_precision_kv_8targets/`
- Records: 62; sampled bundle groups: 31
- Overall mean/p90/max d_norm: 0.278 / 0.356 / 0.394
- Tail `d_norm>0.5`: 0.00

| bundle | n | mean d_norm | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|
| `ast_function_only` | 16 | 0.283 | 0.356 | 0.00 |
| `call_neighborhood_1hop` | 14 | 0.270 | 0.369 | 0.00 |
| `import_dependency_bundle` | 16 | 0.270 | 0.320 | 0.00 |
| `test_target_bundle` | 16 | 0.289 | 0.366 | 0.00 |

Interpretation: KV 层面上，`import_dependency_bundle` 和 `call_neighborhood_1hop` 在 3B/7B 上都没有出现高 tail risk，是值得进入输出和 pass@1 验证的候选。

## 3. Output Drift

- Result: `results/code_graph_kv_reuse/qwen2_5_3b_output_drift_12targets/`
- Model: `/home/gfy/models/Qwen2.5-3B-Instruct`
- Baseline: same target/role 的 `ast_function_only` deterministic JSON output
- Candidates: graph-aware bundles
- Pairs: 108
- Overall mean token F1: 0.765
- JSON valid rate: 0.88
- Reuse-risk match rate: 0.72

| candidate bundle | n | mean token F1 | JSON valid | reuse-risk match | high-risk drift |
|---|---:|---:|---:|---:|---:|
| `call_neighborhood_1hop` | 36 | 0.786 | 0.89 | 0.86 | 0.83 |
| `import_dependency_bundle` | 36 | 0.745 | 0.81 | 0.61 | 0.94 |
| `test_target_bundle` | 36 | 0.766 | 0.94 | 0.69 | 0.86 |

Failure breakdown:

| failure mode | n |
|---|---:|
| `format_failure` | 13 |
| `missing_context_drift` | 9 |
| `ok` | 6 |
| `reuse_risk_drift` | 17 |
| `wrong_or_missing_symbol` | 63 |

Interpretation: 输出层比 KV 层敏感得多。`call_neighborhood_1hop` 的 reuse-risk match 最好，但 relevant-symbol/missing-context 字段仍然经常漂移。因此 graph-aware lossy 进入 pass@1 前必须加 output-level gate，不能只凭 KV distance 放行。

## 4. Pass@1 Readiness Audit

- Existing paired pass@1 cases: 28
- Code graph census cases: 49
- Overlap available for graph-aware pass@1: 13
- Overlap cases: `astropy__astropy-12907`, `astropy__astropy-13033`, `astropy__astropy-13236`, `django__django-10554`, `django__django-10880`, `matplotlib__matplotlib-13989`, `matplotlib__matplotlib-14623`, `matplotlib__matplotlib-20488`, `mwaskom__seaborn-3069`, `mwaskom__seaborn-3187`, `pallets__flask-5014`, `psf__requests-1142`, `psf__requests-1724`
- Current paired pass@1 baseline: lossless 3/28, current lossy 2/28

Current lossy regression(s):

| case | lossy fail step | lossy cached | lossless cached |
|---|---|---:|---:|
| `scikit-learn__scikit-learn-10844` | `json_edit_extract` | 8863 | 1306 |

Interpretation: P3 可以先在这 13 个 overlap cases 上跑，而不是重新扩 100/500 cases。成功标准是 `graph_aware_lossy` regression count 不超过 current lossy，并解释所有 failure mode。

### Live patch-harness readiness, 13 overlap cases

- Result: `results/code_graph_kv_reuse/pass1_graph_aware_13_skiptest/`
- Candidate tests were skipped in this run; it checks generation, JSON-edit synthesis, `git apply --check`, and reuse metadata.

| mode | n | diff extracted | apply ok | generation errors | search-not-found | exact signature match | mean cached tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lossless` | 13 | 4 | 4 | 1 | 6 | 0 | 13687.5 |
| `lossy` | 13 | 2 | 2 | 1 | 8 | 12 | 9681.1 |
| `lossy_prefetch` | 13 | 3 | 3 | 1 | 8 | 12 | 9902.0 |
| `graph_aware_lossy` | 13 | 8 | 8 | 0 | 2 | 13 | 1444.9 |

Interpretation: `graph_aware_lossy` reached exact signature match on 13/13 cases and produced git-applyable JSON edits on 8/13 cases. This is a readiness signal, not pass@1. The dominant remaining failure is `search not found`, so the next pass@1 run should either evaluate only applyable patches first or tighten the JSON-edit prompt to force search strings from the graph bundle.

## 5. Policy Implication

- 默认候选：`call_neighborhood_1hop`，因为 KV 稳定且 output reuse-risk match 最高。
- 保守候选：`import_dependency_bundle`，KV 稳定但 output risk drift 需要 gate。
- 任务诊断：`test_target_bundle`，只用于 SWE-style failure analysis，不作为默认 runtime policy。
- 拒绝条件：JSON invalid、baseline symbol coverage 低、reuse-risk label 改变、或 KV `d_norm>0.5`。

## 6. Next Required Experiment

Run `graph_aware_lossy` paired pass@1 on the 13 overlap cases using the policy above. This is the only missing step before writing a paper-level non-degradation claim for code graph-aware lossy reuse.
