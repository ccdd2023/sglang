# KVFlow architecture and ownership contract

## Data flow

```text
coding-aware policy                    prefetch policy
         │                                   │
         │ KVReusePlan                       │ KVPrefetchHint
         ▼                                   ▼
┌──────────────────── policy-neutral KVCOMM core ────────────────────┐
│ segment identity │ residency │ lease/GC │ validated transfer plan │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                    RadixCache / HiCache backend
```

Coding-aware policy answers **what may be copied and what must be
recomputed**. Prefetch answers **when and where an existing segment should be
loaded**. KVCOMM validates and executes both decisions but creates neither.

## Shared types

- `KVSegmentKey`: content hash, token hash, length, model/cache identity and
  prefix-or-middle kind.
- `KVSegmentHandle`: key, generation, source position, token IDs, residency
  tier and backend reference.
- `KVReusePlan`: non-overlapping dense ranges and copied spans over one target
  token sequence.
- `KVPrefetchHint`: segment key, destination tier, deadline and priority.
- `KVTransferStats`: actual copied, rotated and recomputed tokens plus visible
  fallback reasons.

The store rejects token-length/hash mismatches. Re-registering the same key
creates a new generation so stale handles cannot silently address replaced KV.

## Mechanical contract

Before copying, the executor checks:

1. the handle is current and device resident;
2. source and target token slices are identical;
3. source and target bounds are valid;
4. target ranges do not overlap or leave an unowned gap when full coverage is
   required.

A failed check recomputes the complete affected chunk. Every copied K token
must receive the position delta rotation; V is copied without rotation. A
backend reporting partial K/V copy or partial K rotation is a hard invariant
error.

## Ownership

### Shared Core

May modify `mem_cache/kvcomm`, the minimal `RadixCache` attachment and common
telemetry. It must not import experiment results, AST policy, scheduler
prefetch or eviction code.

### Coding-aware branch

May produce `KVReusePlan` from AST, dependency, task or session signals. It
must work with prefetch disabled and must not call `ensure_resident`.

### Prefetch branch

May consume `KVPrefetchHint`, choose deadlines/priorities and invoke
`ensure_resident`. It must work with coding-aware reuse disabled and must not
import AST or coding risk labels.

### Integration branch

May merge both branches and add thin adapters or composition tests. New
research logic must be fixed in its owning branch.

## Compatibility

No new OpenAI request field is required. Legacy request fields such as
`code_anchor_*`, `codebase_prefetch_hints` and `next_agent_prefix` should be
translated by branch-specific adapters. They are not part of the KVCOMM core
identity model.
