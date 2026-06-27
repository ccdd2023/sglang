# AgentTemplateKV: Coding-MAS-Aware KV Cache Management for SGLang — SUPERSEDED STUB

> ⚠️ **This document is superseded.**
> It was originally written on 2026-06-05 referencing branch
> `feature/context-aware-kv-reuse`. Both the branch reference and the
> project state described here are no longer accurate.
>
> **For the current single source of truth, read
> [CANONICAL_TARGET.md](./CANONICAL_TARGET.md).**
>
> For current session state, see [HANDOFF.md](./HANDOFF.md).

---

## Why this was superseded

This overview predates:

- The Direction #3 AST chunk pool landing (commits `7fb1a5bb2`,
  `8599afcfc`, `5197823bf`, `fea64d4cc`, on branch
  `fix/placeholder-pool-activation`).
- The formal deprecation of L3 placeholder k-NN body
  (commit `8064ea450`).
- The activation of the placeholder pool (commits `d85ca7f45`, `b2920ba64`)
  which fixed the 3 silent bugs that kept L3 from firing in v44.

The "current branch" reference (`feature/context-aware-kv-reuse`) is
the Phase 1/2 era base. **Active branch is `fix/placeholder-pool-activation`**.

The v44 cycle (10 memory entries) is now consolidated — see
`v44-cycle-history` memory entry.

---

## Historical content preserved below

The original content of this document is preserved in
`git log --follow -p -- KVFLOW_OVERVIEW.md` and the previous commit. It
is intentionally NOT inlined here because the historical claims
contradict current project state and create confusion.

If you need to reference historical context for the paper (e.g. Related
Work, "Why we abandoned L3"):

- L3 deprecation rationale: see memory entry `l3-placeholder-knn-deprecated`
- Direction #3 chunk pool migration: see memory entry `direction-3-phase-c-d`
- Pool activation bugs that broke v44: see memory entry `sglang-kvflow-placeholder-pool-bugs`
- Consolidated v44 cycle evidence: see memory entry `v44-cycle-history`
