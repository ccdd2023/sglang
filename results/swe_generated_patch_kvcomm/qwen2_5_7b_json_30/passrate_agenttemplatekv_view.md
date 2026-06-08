# AgentTemplateKV 28-Case Pass@1 View (Qwen2.5-7B, JSON-Edit)

Re-framed from the original `lossless` vs `lossy` rows of
`passrate_table.csv` to the **AgentTemplateKV** terminology used
elsewhere in the paper (KVFlow/KVCOMM stay as reference baselines
in the prose; `lossy` = position-transformed reuse =
`agenttemplatekv_exact_reuse`).

## Main Table

- Cases: 28 discriminative SWE-bench Verified instances
- Dataset: `results/swebench_local_envs/expanded_30_discriminative_instances.json`
- Output schema: `json-edit`

| mode (AgentTemplateKV) | mode (legacy) | n | diff extracted | clean apply | pass@1 | avg cached tokens | avg elapsed ms |
|---|---|---:|---:|---:|---:|---:|---:|
| stock_sglang_prefix_only | lossless | 28 | 14/28 | 14/28 | 3/28 | 1253.3 | 2052.1 |
| agenttemplatekv_exact_reuse | lossy | 28 | 12/28 | 12/28 | 2/28 | 2190.2 | 1729.4 |

## Headline

- **Pass@1**: 3/28 (stock SGLang) → 2/28 (AgentTemplateKV exact reuse) = delta -1.
- **Avg cached tokens**: 1253.3 → 2190.2 = 1.75×.
- **Avg generation latency**: 2052.1 ms → 1729.4 ms = 1.19×.
- **Exact-content reuse**: 28/28 cases hit `exact_code_content_signature`.

## Regression Detail

- **`scikit-learn__scikit-learn-10844`**: lossless = pass, AgentTemplateKV = `json_edit_extract` (match reason = `exact_code_content_signature`, candidates = 24, cached = 8863). Root-cause: model-side JSON-edit extraction failure (path `superviseded` vs `supervised.py`); KVCOMM gate fired correctly. See `results/passrate_28/regression_root_cause.md`.

## Device-First Protected-Anchor Telemetry (sidecar)

From the 3-case device-prefetch smoke run (
`results/agenttemplatekv_device_prefetch_smoke_3`):

| field | total |
|---|---:|
| `agenttemplatekv_prefetch_hit_count` | 6 |
| `codebase_prefetch_device_hit_count` | 6 |
| `agenttemplatekv_prefetch_protected_tokens` | 55795 |
| `agenttemplatekv_prefetch_newly_protected_tokens` | 28127 |
| `agenttemplatekv_prefetch_expired_tokens` | 0 |
| `agenttemplatekv_rejected_large_gap_count` | 0 |

These counters show that the AgentTemplateKV device-first protected-anchor path is exercised end-to-end; the 28-case pass@1 run uses a different harness (no hint serialization) and is not directly comparable on these metrics.

## Cross-Reference

- Original lossless-vs-lossy report: `PASSRATE_REPORT.md` in this directory.
- Regression root-cause: `results/passrate_28/regression_root_cause.md`
- Per-case trace: `results/passrate_28/per_case_trace.jsonl`
- Source CSV: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/passrate_table.csv`
