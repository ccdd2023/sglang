# Phase 2 Plan — SUPERSEDED STUB

> ⚠️ **This document is superseded.**
> It was originally written on 2026-06-16 as the Phase 2 selective AST
> reuse plan. That plan was executed and concluded with v44 results,
> which then triggered the L3 deprecation on 2026-06-27.
>
> **For the current single source of truth, read
> [CANONICAL_TARGET.md](./CANONICAL_TARGET.md).**

---

## Why this was superseded

The Phase 2 plan was completed during the v44 cycle (Jun 23–25, 2026).
The 28-case SWE-bench extended-policy results landed in
`results/swe_*_20260624T*/` and the 91/89/27 byte-equal summary in
`correctness_validation_report_20260624.md`. v44 evidence is preserved
for the paper but the production-facing path is now Direction #3.

Direction #3 Phase A/B/C/D all landed (commits `7fb1a5bb2`,
`8599afcfc`, `5197823bf`, `fea64d4cc`). Next step is the giant-codebase
smoke run with `SGLANG_CHUNKED_PLACEHOLDER_KNN=1
SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1`.

---

## Historical content preserved below

Original content preserved in `git log --follow -p -- PHASE2_PLAN.md`.

For paper evidence:

- Consolidated v44 cycle evidence: see memory entry `v44-cycle-history`
- Direction #3 chunk pool migration: see memory entry `direction-3-phase-c-d`
