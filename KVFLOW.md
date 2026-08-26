# KVFlow research fork

> **ASPLOS 2027 ImpactKV (SWE-bench file-island) is not this page.**
> Collaborator entry: [`IMPACTKV.md`](IMPACTKV.md). Paper + checker:
> [`docs/kvflow/paper/`](docs/kvflow/paper/). This file is the older
> V40–V46 / prefetch composition note (2026-07-28).

This file is the authoritative repository entry point as of **2026-07-28**.
Frozen experiment outputs for the current headline live under
`IMPACTKV_ARTIFACTS` (cluster default `/home/gfy/CodeMAS_Project/kvflow-artifacts/`).

KVFlow studies two independent ways to reduce time-to-first-token for coding
workloads:

1. **Coding-aware lossy reuse** chooses a token-identical middle region whose
   old K/V may be copied, while recomputing the rest of the current request.
2. **KV prefetch** predicts future KV demand and moves an already-created
   segment to the device before it is needed.

The first answers **what may be reused**. The second answers **when and where
an existing segment should become resident**. They share only the
policy-neutral **KVCOMM data plane** and must remain independently testable.

## Branches

| Branch | Owner and allowed behavior |
|---|---|
| `kvflow/shared-core` | Segment identity, residency, lease lifecycle, safe transfer plan execution |
| `research/coding-aware-lossy` | Coding signals and lossy recompute/copy plans; no prefetch or eviction |
| `research/prefetch-p8-async-20260722` | Prefix/middle-KV prefetch, priority and residency; no coding policy |
| `integration/coding-aware-prefetch-v2` | Current composition tests and thin adapters only |

The legacy `feature/context-aware-kv-reuse` and
`fix/placeholder-pool-activation` branches are preserved as read-only
research history. Do not use them as a shared development base.

## Current coding-aware method

The active research arm is
`coding_grounded_observation_island_v40` (“V40”). It is a **pure KV-reuse**
method; it does not prefetch.

For every next agent request it:

1. keeps a rolling six-interaction history and reasons over the five groups
   that will remain after the next roll;
2. considers only successful, substantial, read-only tool output produced by
   commands such as `rg`, `grep`, `find`, `sed`, `cat`, `head`, or `tail`;
3. rejects execution output, state-changing commands, failed reads, short
   observations, assistant reasoning, and a read whose repository path was
   later mutated;
4. tokenizes eligible tool outputs, requires one exact occurrence in the
   target prompt and a strict middle position, then selects one largest island
   (at least 128 and at most 4,096 copied tokens; newest wins ties);
5. materializes full-layer K/V from a request that naturally computes that
   history, and allows one exact-token reuse in the next request;
6. copies V unchanged, rotates every copied K by the source-to-target RoPE
   position delta, and densely computes the prefix and suffix.

The coding signal is therefore concrete but deliberately narrow:
**successful repository reads + file-version invalidation**. V40 does not yet
use AST structure, symbol dependencies, test relevance, per-layer sensitivity,
separate K/V budgets, or post-copy repair.

The reuse remains lossy because a token-identical tool observation was
originally encoded under an older left context. RoPE correction fixes position
coordinates, not that contextual-state difference.

Implementation:

- `benchmark/multi_workflow/coding_reuse_policy.py`
- `benchmark/multi_workflow/bridge_reuse_litellm_model.py`
- `python/sglang/srt/mem_cache/kvcomm/radix_backend.py`

The active/reproducible benchmark surface is indexed in
`benchmark/multi_workflow/README.md`.

## Latest truthful result: V44

Frozen artifact:

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_v44_dense_sensitive_v40_20260728/V44_RESULT.json
```

V44 is a **12-task SWE-bench Verified development experiment**, not a
population or SOTA result.

| Metric | Dense | General contiguous reuse | V40 |
|---|---:|---:|---:|
| Official resolved | 3/12 | 3/12 | **4/12** |
| Wilson 95% interval | 8.9–53.2% | 8.9–53.2% | 13.8–60.9% |
| Damage among 3 Dense-pass tasks | — | 1/3 | **0/3** |
| Rescue among 9 Dense-fail tasks | — | 1/9 | 1/9 |
| Copied tokens | 0 | 487,144 | **171,139** |
| Fixed-order host-resident median TTFT | 357.6 ms | 335.7 ms | 327.5 ms |

V40 copied **64.9% fewer tokens** than General while winning one task that
General lost (`scikit-learn__scikit-learn-10297`). However, the sample is too
small for a superiority claim: the V40 accuracy interval is wide, only three
tasks expose Dense-pass damage sensitivity, all sources were host-resident,
and fixed-order TTFT is only a diagnostic.

The preceding V43 run is not accuracy evidence. All six tasks exhausted the
20-call agent budget and produced empty submissions, so the run was audited as
a protocol failure and replaced by V44.

Historical native baselines on a different frozen 225-task protocol remain the
competitive reference:

- KVCOMM: 164/225, 8.55× cache-ready and 5.34× at N=4 including build;
- CacheBlend: 169/225, 4.77× cache-ready and 1.22× at N=4 including build.

These figures cannot be directly ranked against V44. A same-task, native,
four-arm evaluation is still required before claiming that V40 beats either
baseline.

## Merge readiness

Research and collaborator heads used by this audit:

```text
V44 result     144e80255
coding docs    current HEAD (this file)
prefetch      e44ce40dc
integration-v2 0ab4fc942
old integration d4a7ec132
shared core   c16bfbb8e
```

A three-way merge preview from the shared-core merge base found:

- coding-aware changed 174 paths after this documentation cleanup;
- prefetch changed 11 paths;
- **no overlapping code paths**;
- two modify/delete documentation conflicts:
  `docs/kvflow/HANDOFF.md` and `docs/kvflow/STATUS.md`.

The old integration head contains only an early July-17 coding snapshot.
Integration-v2 was therefore created from the current coding head and merged
with the latest prefetch head. The merge added the independent
`kvcomm_prefetch/` namespace and one composition test; it did not modify either
research branch.

The merged unit surface passed **113 tests**, including coding policy, KVCOMM,
prefetch coordinator/middle-KV/async scheduler, and
select → reside → validate → execute → release composition. This establishes
merge and lifecycle mechanics, not production GPU prefetch performance.

## Feature gates

All new behavior is opt-in:

```bash
export SGLANG_KVCOMM_CORE=1
export SGLANG_CODING_AWARE_LOSSY=0
export SGLANG_KV_PREFETCH=0
```

Enabling either client without `SGLANG_KVCOMM_CORE=1` is an error. Old
`SGLANG_LOSSY_ENABLED` behavior is available only when
`SGLANG_KVFLOW_LEGACY_FLAGS=1` is also set.

## Read next

- [Architecture, ownership, and merge contract](docs/kvflow/ARCHITECTURE.md)
- [Experiment script index](benchmark/multi_workflow/README.md)
