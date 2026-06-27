# Session Handoff 2026-06-23 — SUPERSEDED STUB

> ⚠️ **This document is superseded.**
> It was the session handoff on 2026-06-23. The default value of
> `SGLANG_PLACEHOLDER_KNN_MATCH` it documents is wrong — the actual
> current default is `0` (production-safe), not `1`.
>
> **For the current single source of truth, read
> [CANONICAL_TARGET.md](./CANONICAL_TARGET.md).**
>
> For current session state, see [HANDOFF.md](./HANDOFF.md).

---

## Why this was superseded

The L3 placeholder k-NN body (which this handoff promoted as the
default ON path) was formally deprecated 4 days later on 2026-06-27
(commit `8064ea450`). The default flipped from
`SGLANG_PLACEHOLDER_KNN_MATCH=1` to `=0`. L3 is now research-only.

Anyone following the instructions in this old handoff will get wrong
default behavior. **Do not run with `SGLANG_PLACEHOLDER_KNN_MATCH=1`
unless you explicitly want the deprecated L3 path for research.**

The current production path is L1 + L2 only (1.31× TTFT speedup).
The next research target is L1 + L2 + L4 chunk pool (~1.49×), enabled
via `SGLANG_CHUNKED_PLACEHOLDER_KNN=1 SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1`.

---

## Historical content preserved below

Original content preserved in `git log --follow -p -- SESSION_HANDOFF_2026-06-23.md`.

For paper evidence and migration story:

- L3 deprecation rationale: see memory entry `l3-placeholder-knn-deprecated`
- Direction #3 chunk pool migration: see memory entry `direction-3-phase-c-d`
- Consolidated v44 cycle evidence: see memory entry `v44-cycle-history`
