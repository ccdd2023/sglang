# ~~KVFlow 最优场景测试方案~~ — SUPERSEDED STUB

> ⚠️ **This document is superseded (2026-05-19).**
> It was a 179-line Chinese-language benchmark design rationale that
> proposed increasing workflow/agent counts to demonstrate KVFlow
> priority benefits. The design assumptions in this document were
> superseded by:
> - Phase 2.4 whole-slot byte-exact reuse (L2), shipped 2026-06-14
> - L3 (MiniLM k-NN body) **deprecated for production** 2026-06-27
> - Direction #3 (L4 AST chunk pool), Phase A/B/C/D landed 2026-06-27
> - The giant-codebase benchmark (50 pandas tasks × 5 agents) at
>   `benchmark/multi_workflow/bench_giant_codebase_reuse.py`, which
>   is the canonical benchmark driver for current measurements
>
> **For current benchmark commands, see
> [../../HANDOFF.md](../../HANDOFF.md) §"Common commands".**
> **For current project state, see
> [../../CANONICAL_TARGET.md](../../CANONICAL_TARGET.md).**
>
> Original 179-line content is preserved in
> `git log --follow -p -- benchmark/multi_workflow/KVFLOW_OPTIMAL_DESIGN.md`.
