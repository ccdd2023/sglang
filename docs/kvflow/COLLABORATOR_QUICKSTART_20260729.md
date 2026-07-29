# Coding-aware × Prefetch collaborator quickstart

This page is the shortest current description of what the two owners are
trying to compose. It supplements, rather than replaces, `KVFLOW.md`,
`docs/kvflow/ARCHITECTURE.md`, and
`docs/kvflow/PREFETCH_HANDOFF_20260722.md`.

## One-sentence intent

Coding-aware chooses **which exact historical KV span may be reused**;
prefetch decides **when and where that already-identified span becomes
resident**. Prefetch must not change the selected span, and coding-aware must
not obtain hidden warming in coding-only experiments.

## Branches and owners

| Branch | Owner responsibility |
|---|---|
| `kvflow/shared-core` | Segment identity, leases, transfer and fail-closed validation |
| `research/coding-aware-lossy` | Coding evidence, invalidation and `KVReusePlan` construction |
| `research/prefetch-p8-async-20260722` | `KVPrefetchHint`, queueing, deadlines and residency |
| `integration/coding-aware-prefetch-v2` | Thin composition adapters and four-mode tests |

Do not cherry-pick research commits between the coding and prefetch branches.
Merge both owners only in the integration branch.

## Current coding-aware method

V40 reuses one successful, read-only repository observation from recent agent
history. It rejects reasoning text, failed commands, mutable or subsequently
modified files, duplicate text, short spans and prompt prefix/suffix spans.
The target copies V, rotates K by the RoPE position delta, and densely
recomputes everything outside the selected island.

The current evidence is intentionally modest:

- RepoBench-P static control: 1.089x cache-ready TTFT, exact-line 10% to 8%.
- Twelve-task SWE-bench development cohort: Dense 6/12, V40 4/12; median
  TTFT 295.5 ms to 258.3 ms.
- The two observed Dense-pass/V40-fail tasks did not reproduce as stable
  regressions: both Dense repeats failed. Treat the accuracy difference as a
  single-run point estimate, not a causal loss estimate.
- Native KVCOMM uses a different multi-agent prompt topology. Cross-system
  absolute accuracy is not a prompt-controlled comparison.

Machine-readable and human-readable evidence lives outside Git under:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_three_method_coding_benchmark_20260728/
    THREE_METHOD_AUDIT.json
    THREE_METHOD_AUDIT.md
```

## Composition contract

The coding owner provides:

1. an immutable `KVSegmentKey`;
2. the exact source and target token spans;
3. RoPE delta and dense/copy partition in a `KVReusePlan`;
4. policy label, file-version evidence and fail-closed reasons.

The prefetch owner may:

1. consume a `KVPrefetchHint`;
2. prioritize or cancel residency work;
3. return a leased device-resident handle for the same segment;
4. expose queue/load/wait telemetry.

The prefetch owner must not:

- widen, shrink or replace the coding-selected span;
- turn an invalid or missing source into a different reusable source;
- own source-file validity or coding-risk policy;
- make coding-only experiments cache-ready in the background.

If residency is late or missing, the combined path falls back to the same
dense tokens that the coding plan already declared.

## Four required modes

| Mode | Core | Coding | Prefetch | Purpose |
|---|---:|---:|---:|---|
| Dense | 1 | 0 | 0 | accuracy and latency reference |
| Coding-only | 1 | 1 | 0 | measure pure lossy reuse |
| Prefetch-only | 1 | 0 | 1 | measure residency/scheduling |
| Combined | 1 | 1 | 1 | verify composition and total TTFT |

The corresponding gates are:

```bash
export SGLANG_KVCOMM_CORE=1
export SGLANG_CODING_AWARE_LOSSY=0
export SGLANG_KV_PREFETCH=0
```

Change only the final two values for the four modes. Every result must record
the three values explicitly.

## Start here

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/integration-v2
git status --short
git log --graph --oneline --decorate -12
```

The integration branch already contains the collaborator's async prefetch
branch and the earlier V40 merge rehearsal. After the coding owner commits a
new batch, merge that commit into integration; do not merge integration back
into either research owner branch.

Run the shared and composition suites:

```bash
PYTHONPATH=python /home/gfy/.conda/envs/sglang-kvflow/bin/python -m pytest -q \
  python/sglang/srt/mem_cache/kvcomm/test_core.py \
  python/sglang/srt/mem_cache/kvcomm_prefetch/test_coordinator.py \
  python/sglang/srt/mem_cache/kvcomm_prefetch/test_middle_kv.py \
  python/sglang/srt/mem_cache/kvcomm_prefetch/test_scheduler.py \
  python/sglang/srt/mem_cache/kvflow_integration/test_composition_v2.py
```

Then run one canary per mode with the same model, task, prompt, generation
limits and order. Report:

- official task accuracy;
- cache-ready and transfer-inclusive TTFT;
- selected, copied, recomputed and fallback tokens;
- prefetch queue, load and wait time;
- source residency and lease cleanup.

## Merge stop conditions

Stop the combined merge if any of the following occurs:

- coding-only behavior changes when prefetch is disabled;
- combined mode changes the coding-selected token span;
- source generation, model identity, token hash or file version is bypassed;
- a missing prefetch ticket does not fall back to Dense;
- a lease, source slot, worker or CUDA event survives request cleanup;
- the four modes cannot be distinguished from telemetry.

Paper files, old preregistration gates and large experiment artifacts are not
part of this merge.
