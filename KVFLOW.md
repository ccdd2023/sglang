# KVFlow fork

This fork studies two independent ways to reduce time-to-first-token for
coding workloads:

1. **Coding-aware lossy reuse** reuses more KV without relying on eviction. It
   decides which code or workflow regions must be recomputed for accuracy.
2. **KV prefetch** predicts future prefix and middle-of-request KV demand under
   concurrency, then moves those segments to the device before use.

The two control planes share only the policy-neutral **KVCOMM data plane**.
Neither research branch imports or configures the other.

## Branches

| Branch | Owner and allowed behavior |
|---|---|
| `kvflow/shared-core` | Segment identity, residency, lease lifecycle, safe transfer plan execution |
| `research/coding-aware-lossy` | Coding signals and lossy recompute/copy plans; no prefetch or eviction |
| `research/prefetch` | Prefix/middle-KV prefetch, priority and eviction; no coding policy |
| `integration/coding-aware-prefetch` | Composition tests and thin adapters only |

The legacy `feature/context-aware-kv-reuse` and
`fix/placeholder-pool-activation` branches are preserved as read-only
research history. Do not use them as a shared development base.

## Current status

The code layout is **interface complete but server-canary pending**:

- `kvcomm/` has identity, generation, lease/resource lifecycle, validated
  transfer plans and a Radix allocator adapter;
- the coding policy produces complete copy/dense plans without importing
  scheduler or prefetch code;
- the prefetch branch has a host/device middle-KV handoff contract;
- the integration branch has a reference composition test;
- no production request currently calls `KVCommManager.execute`, so none of
  these interface tests is an end-to-end SGLang speed result.

Research status:

- FileVersion SessionGraphKV V11 formal P0: **FALSIFIED**;
- ProbeHead StateSensitivityKV V12 development calibration:
  **FALSIFIED** (`4,784` observations, `4,639` configurations, `0` feasible);
- sequential composition, holdout, objective workflow accuracy and P1 TTFT
  remain closed.

Before a runtime-complete claim, the project still needs a real model-server
exact-transfer canary, production allocator/source-lifecycle integration,
target-slot and dense-fallback wiring, HiCache payload validation, stream
synchronization, the four-mode server matrix and sustained lifecycle tests.

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

- [Coding-aware session handoff (2026-07-17)](CODING_AWARE_HANDOFF_20260717.md)
- [Architecture and interface contract](docs/kvflow/ARCHITECTURE.md)
- [Weekly research and collaboration audit](docs/kvflow/WEEKLY_RESEARCH_AUDIT_20260718.md)
