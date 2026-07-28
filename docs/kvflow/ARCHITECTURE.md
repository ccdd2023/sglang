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
may add the minimal scheduler/request adapter needed to execute that plan. It
must work with prefetch disabled, must not call `ensure_resident`, and must not
implement deadline, priority, eviction, or residency scheduling.

### Prefetch branch

May consume `KVPrefetchHint`, choose deadlines/priorities and invoke
`ensure_resident`. It must work with coding-aware reuse disabled and must not
import AST or coding risk labels.

### Integration branch

May merge both branches and add thin adapters or composition tests. New
research logic must be fixed in its owning branch.

## Composition contract

Merging code is not permission to blur the two mechanisms. The integration
branch must expose and record four modes:

| Mode | Coding reuse | Prefetch | Required attribution |
|---|---:|---:|---|
| feature-off | off | off | Dense reference |
| coding-only | on | off | selected/copy/dense/fallback tokens |
| prefetch-only | off | on | queue/load/residency/lease timing |
| combined | on | on | both namespaces plus shared transfer stats |

Rules:

1. `KVReusePlan` may name a segment but must not call `ensure_resident`.
2. `KVPrefetchHint` may move or pin a segment but must not change the reuse
   plan, its copied-token budget, or its accuracy label.
3. One `KVSegmentHandle` has one generation and one store owner. Prefetch must
   not create a second lifecycle for a segment already registered by reuse.
4. A `PrefetchTicket` lease is released after execution, cancellation, timeout,
   or fallback. Request/Radix ownership of source slots remains separate.
5. Position correction occurs exactly once in `KVCommManager.execute` through
   the transfer backend. Prefetch never pre-rotates K.
6. If a segment is missing, stale, mismatched, late, or non-resident, the
   affected span fails closed to Dense and records the reason.
7. Coding-only results must run with `SGLANG_KV_PREFETCH=0`; they cannot include
   hidden residency warming. Prefetch speed must be reported separately as
   cache-ready and build/transfer-inclusive.

## Integration sequence

The existing integration branch is stale relative to the current coding head.
Use a fresh integration branch; do not merge either research branch into the
other.

```bash
git switch -c integration/coding-aware-prefetch-v2 \
  research/coding-aware-lossy
git merge --no-ff research/prefetch-p8-async-20260722
```

Resolve the two documentation conflicts by keeping `KVFLOW.md` as the global
status and incorporating collaborator-specific operational notes without
restoring stale coding claims. No paper, report artifact, experiment result, or
preregistered threshold belongs in this merge.

## Integration acceptance gates

Before the combined branch is handed back:

- branch-scope check passes for integration;
- coding-only unit and V40 policy tests pass unchanged;
- prefetch coordinator, middle-KV, and async scheduler tests pass unchanged;
- one composition test proves select → host registration → async residency →
  reuse execution → lease release;
- the same test covers late ticket, cancelled ticket, stale generation, token
  mismatch, and Dense fallback;
- feature-off, coding-only, prefetch-only, and combined server smokes run from
  the same base configuration;
- repeated combined requests show no allocator, store, generation, ticket, or
  lease growth;
- telemetry makes copied-token savings distinguishable from queue/load overlap;
- no reported coding-only speed number depends on prefetch.

## Compatibility

No new OpenAI request field is required. Legacy request fields such as
`code_anchor_*`, `codebase_prefetch_hints` and `next_agent_prefix` should be
translated by branch-specific adapters. They are not part of the KVCOMM core
identity model.
