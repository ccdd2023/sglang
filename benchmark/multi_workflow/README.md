# Coding-aware experiment surface

This directory intentionally retains frozen experiment drivers: artifact
registrations and manifests reference their exact code, so deleting an old
runner would reduce reproducibility rather than clean the repository.

Use this index instead of treating all `run_v*.py` files as active.

## Active implementation and latest campaign

- `coding_reuse_policy.py`: policy definitions; V40 candidate classification.
- `bridge_reuse_litellm_model.py`: rolling-history source/target adapter.
- `motivate_v40_grounded_observation_island.py`: V40 motivation analysis.
- `run_v44_dense_sensitive_v40_campaign.py`: latest frozen campaign.
- `summarize_v44_schema_compat.py`: narrowly scoped post-treatment summary
  repair; it does not modify raw rows or official evaluations.
- `audit_v43_call_budget_collapse.py`: records why V43 is protocol-invalid.

Focused tests:

```text
test_coding_reuse_policy.py
test_bridge_reuse_litellm_model.py
test_motivate_v40_grounded_observation_island.py
test_v43_new_verified_v40_campaign.py
test_audit_v43_call_budget_collapse.py
test_v44_dense_sensitive_v40_campaign.py
test_summarize_v44_schema_compat.py
```

## Historical but reproducible

- V11–V12: `sessiongraph_*`, `probehead_v12*`,
  `measure_*`, `analyze_*`, and their tests.
- V13–V39: versioned `motivate_`, `probe_`, `run_`, `audit_`, `summarize_`,
  and matching test files.
- Dataset/container preparation: `prepare_*`, `freeze_*`,
  `generate_swebench_verified_predictions.py`,
  `run_swebench_verified_containers.py`, and the frozen JSON/YAML inputs.

These are not recommended entry points, but they are evidence for failed or
superseded methods and must stay until their artifacts are moved to a
separately versioned research archive.

## Repository hygiene rule

Do not commit run outputs, logs, downloaded repositories, containers,
`__pycache__`, or `.pyc` files here. Outputs belong under
`/home/gfy/CodeMAS_Project/kvflow-artifacts/`; presentation reports belong
under `/home/gfy/CodeMAS_Project/kvflow-reports/`.

When a new version becomes active, update only the first section. Delete a
historical driver only after an artifact manifest proves that the source has
been archived with a content hash and a reproducible entry command.
