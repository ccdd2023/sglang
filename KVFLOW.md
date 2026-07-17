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

- [Architecture and interface contract](docs/kvflow/ARCHITECTURE.md)
- [Verified status and known gaps](docs/kvflow/STATUS.md)
- [Current handoff and commands](docs/kvflow/HANDOFF.md)
- [Historical handoff index](_archive/handovers/README.md)
