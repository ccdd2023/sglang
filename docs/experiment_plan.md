# KVFlow 实验设计文档 — SUPERSEDED STUB

> ⚠️ **This document is superseded.**
> It was the original KVFlow experiment plan written 2026-06-07. It
> predates the L3 deprecation (2026-06-27) and Direction #3 chunk pool
> (commits `7fb1a5bb2`, `8599afcfc`, `5197823bf`).
>
> **For the current single source of truth, read
> [../CANONICAL_TARGET.md](../CANONICAL_TARGET.md).**

---

## Why this was superseded

- L3 (placeholder k-NN body) was deprecated — the 3× speedup goal is
  no longer the production target.
- The branch pointers in this doc (`phase-2.7-prerot` etc.) are stale;
  active branch is `fix/placeholder-pool-activation`.
- HiCache host storage is no longer broken (this doc lists it as a
  known limitation); that was fixed earlier in the v44 cycle.

The active experiment targets are now:

1. Direction #3 chunk pool smoke on giant-codebase
   (`SGLANG_CHUNKED_PLACEHOLDER_KNN=1 SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1`)
2. (Optional) Phase E: whitespace-drift tolerance gated on telemetry

---

## Historical content preserved below

Original content preserved in `git log --follow -p -- docs/experiment_plan.md`.

For current direction: [../CANONICAL_TARGET.md](../CANONICAL_TARGET.md).
