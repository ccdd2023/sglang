# Phase 2 Findings — SUPERSEDED STUB

> ⚠️ **This document is superseded.**
> It was originally written on 2026-06-20 documenting Phase 2 selective
> AST reuse workstream findings. That workstream is no longer active.
>
> **For the current single source of truth, read
> [CANONICAL_TARGET.md](./CANONICAL_TARGET.md).**

---

## Why this was superseded

Phase 2 selective AST reuse was the v44-era optimization layer that
exposed the L3 byte-vs-semantic mismatch problem. The findings here
were the basis for the formal L3 deprecation (commit `8064ea450`,
2026-06-27).

The current code path is **Direction #3 AST chunk pool** (Phase A/B/C/D
landed on branch `fix/placeholder-pool-activation`), which preserves
the byte-exact invariant at chunk granularity rather than relying on
MiniLM semantic similarity.

---

## Historical content preserved below

Original content preserved in `git log --follow -p -- PHASE2_FINDINGS.md`.

For paper evidence:

- L3 deprecation rationale: see memory entry `l3-placeholder-knn-deprecated`
- Direction #3 chunk pool migration: see memory entry `direction-3-phase-c-d`
- Consolidated v44 cycle evidence: see memory entry `v44-cycle-history`
