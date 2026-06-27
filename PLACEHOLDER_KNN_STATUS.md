# Placeholder k-NN KV Reuse — SUPERSEDED STUB

> ⚠️ **This document is superseded.**
> It was originally written on 2026-06-22 and claims the v44 placeholder
> k-NN body (L3) was "production-ready" or "ready for O5-real". This
> is no longer true.
>
> **The L3 placeholder k-NN body was formally DEPRECATED on 2026-06-27**
> (commit `8064ea450`). The driver default flipped from
> `SGLANG_PLACEHOLDER_KNN_MATCH=1` to `=0`; L3 is now research-only
> behind `--enable-research-l3`.
>
> **For the current single source of truth, read
> [CANONICAL_TARGET.md](./CANONICAL_TARGET.md).**

---

## Why this was superseded

The 3 critical bugs in the placeholder pool activation (commits
`d85ca7f45`, `b2920ba64`, and predecessors) that kept L3 from firing
across the entire v44 cycle were eventually diagnosed and fixed.
However, fixing the activation exposed the underlying byte-vs-semantic
mismatch: 8.2% of pool hits were non-byte-identical (variable renames,
comment edits), which MiniLM cos ≥ 0.85 cannot distinguish from
benign whitespace drift. The failure mode is silent: tests pass,
output reads correctly, but runtime behavior diverges.

The safe replacement is **Direction #3 AST chunk pool** (Phase A/B/C/D
landed), which preserves the byte-exact invariant at function/class
boundary chunks rather than relying on semantic similarity.

---

## Historical content preserved below

Original content preserved in `git log --follow -p -- PLACEHOLDER_KNN_STATUS.md`.

For paper evidence and detailed migration story:

- L3 deprecation rationale: see memory entry `l3-placeholder-knn-deprecated`
- Direction #3 chunk pool migration: see memory entry `direction-3-phase-c-d`
- Pool activation bugs that broke v44: see memory entry `sglang-kvflow-placeholder-pool-bugs`
- Consolidated v44 cycle evidence: see memory entry `v44-cycle-history`
