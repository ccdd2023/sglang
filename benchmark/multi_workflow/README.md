# Coding-aware experiment surface

**Current ASPLOS headline** (7B SWE-bench file-module replay, job 137185)
is **not** the V46 RepoBench drivers below. Start at
[`../../IMPACTKV.md`](../../IMPACTKV.md). Run on your own GPU: unpack
`offcluster/impactkv-claim-pack.tar.gz` then `run_impactkv_headline.sh`.

SWE-bench exact-prompt runners on this branch:

- `prepare_7b_swebench_file_modules_plan.py` / `run_swebench_prerotated_file_modules.py`
- `run_swebench_7b_prefix_on.py` (job 139839)
- `run_swebench_7b_sota_copiers.py` (job 137400, same-engine clones)
- `run_swebench_template_prefetch.py` (appendix; not the 1.492× method)
- `slurm/swebench_*.sbatch`

This directory also retains frozen older experiment drivers: artifact
registrations and manifests reference their exact code, so deleting an old
runner would reduce reproducibility rather than clean the repository.

Use this index instead of treating all `run_v*.py` files as active.

## Active implementation and latest campaign

- `coding_reuse_policy.py`: V40 source classification, V45 target-time
  file-version validation, and V46 online-observed path provenance.
- `bridge_reuse_litellm_model.py`: rolling-history source/target adapter;
  `coding_observed_path_pool_v46` is the active development arm. It keeps at
  most three persistent grounded tool-observation sources and selects at most
  three non-overlapping shifted islands per target. Sources referenced by the
  current target are protected from same-request eviction.
- `motivate_v45_versioned_evidence.py`: answer-blind audit of V45's two
  proposed mechanisms on frozen V40 trajectories. The audit found a real
  cross-request invalidation gap but no symbol-disjoint reuse opportunity, so
  the active arm keeps the guard and does not enable symbol relaxation.
- `audit_v45_selected_target_guard.py`: production-tokenizer replay through
  the real V40/V45 planners. The strict rerun verified identical prompts and
  shared token segments, 203 V40 targets versus 183 V45 targets, and eight
  runtime-eligible V40 targets removed after a newly visible same-file write.
- `audit_v451_multi_observation_pool.py`: tests whether multiple independent
  grounded observations create useful extra islands without using answers.
- `audit_v453_observed_path_pool.py`: adds literal repository paths observed
  in the current tool output; repository-wide search observations invalidate
  after any later repository write.
- `audit_v46_runtime_parity.py`: replays the production V46 bridge and checks
  prompt identity, coverage, pool/island bounds, and the source-lifetime
  invariant before GPU execution.
- `run_v46_repobench_control.py`: three-island static RepoBench-P mechanism
  control with physical copy/fallback telemetry and source-build accounting.
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

V46's lifecycle-safe offline replay covers 236/331 requests and 28.51% of
prompt tokens with zero prompt mismatch or target/source-release conflict.
The completed 50-case RepoBench-P mechanism control makes 150/150 physical
island copies with zero fallback and measures 1.326x cache-ready speedup
(1.050x at N=4 including source build). Exact next-line accuracy is Dense 5/50
and V46 4/50. On three prior Dense/V40 SWE-bench passes, V46 preserves 2/3;
one of the two tasks with active copied KV fails official evaluation. The
full-12 campaign was therefore not restarted after its combined canary. These
results establish speed opportunity but do not support an accuracy or SOTA
claim. See `docs/kvflow/CODING_AWARE_V46_DEVELOPMENT_20260803.md`.

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
