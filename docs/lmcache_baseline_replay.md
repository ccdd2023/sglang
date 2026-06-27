# LMCache Baseline Replay Runbook — HISTORICAL RUNBOOK

> ⚠️ **This runbook is dated 2026-06-14.**
> It is preserved as-is for paper reproducibility, but the
> **SGLang version and KVFlow implementation have evolved since**.
>
> For current directions: see [../CANONICAL_TARGET.md](../CANONICAL_TARGET.md).

---

## Status

This is a dated runbook describing how to run a same-workload LMCache
baseline comparison against AgentTemplateKV at the time of writing.
The LMCache integration in sglang has changed since then; if you need
to rerun this baseline, check the sglang-kvflow branch state first
(currently `fix/placeholder-pool-activation`).

For the current benchmark infra, see
`benchmark/multi_workflow/bench_giant_codebase_reuse.py` which
supersedes this runbook for the AgentTemplateKV-vs-baseline comparison.
