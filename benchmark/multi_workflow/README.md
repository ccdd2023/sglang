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

## Active three-method coding comparison

`register_three_method_coding_benchmark.py` freezes the narrowed comparison:
V40, KVCOMM, and CacheBlend only.  Every native engine retains a matched Dense
normalization arm.  The headline development cohort is selected from
SWE-bench Verified using outcome-free reuse-opportunity measurements from
already-frozen Dense trajectories; a salted-hash RepoBench-P sample is the
static repository-completion control.

`run_v40_repobench_control.py` implements the V40 static projection for that
control.  It treats repository chunks as successful read-only observations,
copies one largest unique 128--4096-token middle island, and recomputes
everything else.  Because RepoBench-P has neither an agent trajectory nor file
mutation, this is mechanism/control evidence and not a substitute for the
SWE-bench result.

`run_swebench_with_limit_patch_capture.py` preserves a tracked source diff
when mini-SWE-agent reaches its fixed call limit; it adds no model request and
prevents a real edit from being mislabeled as an empty submission.
`summarize_three_method_coding_benchmark.py` combines the frozen RepoBench-P
native-engine controls with the same-engine Qwen3 SWE-bench Dense/V40 result.
Its output deliberately does not impute unrun KVCOMM or CacheBlend SWE-bench
accuracy, and it does not rank cross-engine absolute latency.
The collaborator-facing intent and merge boundary are summarized in
`docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md`.

QCFuse, FUSE-RAG, ProphetKV, tail repair, generic contiguous reuse, and
prefetch are outside this campaign.  The reuse-rich SWE-bench cohort is
explicitly a mechanism/development cohort rather than a population estimate.

## CacheBlend static-control adapter

`cacheblend_coding_matrix.py` and `run_cacheblend_coding_matrix.sh` add a
native CacheBlend lane for the same retained LongBench LCC and RepoBench-P
texts used by the 2026-07-28 QCFuse comparison.  CacheBlend stays in its
officially derived vLLM 0.4.1 engine and is normalized to a matching native
Dense arm.

The public CacheBlend runtime cannot load the Qwen3-8B snapshot used by the
QCFuse common-stack lane.  Therefore cross-engine absolute accuracy and
milliseconds are descriptive only.  The comparable quantities are accuracy
change versus native Dense and TTFT speedup versus native Dense.  This is an
additive post-hoc comparison lane and does not modify the project's frozen
pre-registration gates.

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
