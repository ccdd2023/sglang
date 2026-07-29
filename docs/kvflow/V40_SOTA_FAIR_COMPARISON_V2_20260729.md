# V40 SOTA fair comparison v2

## Purpose

This experiment compares V40 only with CacheBlend and KVCOMM. Prefetch,
general reuse, tail repair, QCFuse and FUSE-RAG are outside the headline
matrix.

The comparison has two enforced layers:

- **controlled**: model input token hashes must be identical before methods
  may be ranked directly;
- **native**: each reuse method is paired with Dense from the same upstream
  engine, and absolute cross-engine TTFT is not ranked.

## Frozen experiment

- Model: Qwen2.5-Coder-3B-Instruct snapshot
  `488639f1ff808d1d3d0ba301aef8c11461451ec5`
- Development: the existing 12-task reuse-rich SWE-bench Verified cohort.
- Holdout: 20 disjoint SWE-bench Verified tasks selected by salted hash,
  capped at two tasks per repository and selected without method outputs.
- Static controls: all 200 RepoBench-P and all 200 LCC cases.
- V40 copy caps: 2048, 4096 and 8192.
- CacheBlend recompute ratios: 0.25, 0.50 and 0.75.
- KVCOMM thresholds: 0.3, 0.5 and 0.7; anchor count 20; window 5.
- Temperature 0, concurrency 1, no prefetch.

The immutable registration and generated commands are under:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_v40_sota_fair_v2_20260729/
    COMPARISON_REGISTRATION.json
    CANARY_COMMAND_PLAN.json
    STATIC_COMMAND_PLAN.json
```

## Canary result

The RepoBench-P three-case protocol canary passed:

- V40, CacheBlend and corrected KVCOMM each produced physical reuse on 3/3
  targets with no fallback.
- V40 and CacheBlend target token hashes matched on 3/3 cases.
- KVCOMM token hashes differed on 3/3 because its native three-agent graph
  rewrites the prompt; it remains native-only.
- The canary is a mechanism check, not an accuracy claim.

The canary exposed two adapter bugs in the previously instrumented KVCOMM
runner: canary mode skipped source materialization and every graph agent
ignored the frozen 64-token output limit in favor of the native 512-token
default. The isolated KVCOMM comparison worktree fixes both without changing
the upstream anchor/reuse algorithm. Failed pre-fix ledgers are preserved and
explicitly excluded in `CANARY_AUDIT.json`.

Current three-case cache-ready/N=4 speedups are:

| Method | Cache-ready | N=4 including build |
|---|---:|---:|
| V40 | 1.089x | 0.911x |
| CacheBlend | 1.306x | 0.593x |
| KVCOMM native graph | 14.234x | 7.987x |

KVCOMM's absolute timing is not comparable with the controlled single-request
methods; only its paired native Dense speedup is meaningful.

## Verification and execution

Run focused tests:

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/fair-comparison-v2
PYTHONPATH=.:python /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  benchmark/multi_workflow/test_fair_sota_comparison_v2.py \
  benchmark/multi_workflow/test_prepare_fair_sota_comparison_v2.py \
  benchmark/multi_workflow/test_execute_fair_sota_canary_v2.py \
  benchmark/multi_workflow/test_summarize_fair_sota_canary_v2.py
```

Execute the frozen static matrix sequentially:

```bash
PYTHONPATH=.:python /home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/execute_fair_sota_canary_v2.py \
  --plan /home/gfy/CodeMAS_Project/kvflow-artifacts/\
impactkv_v40_sota_fair_v2_20260729/STATIC_COMMAND_PLAN.json \
  --all
```

The executor refuses to append to any existing ledger or overwrite a status,
stdout or stderr log. A failed command stops the matrix and remains available
for diagnosis.
