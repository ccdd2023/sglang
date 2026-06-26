# AST-Alignment Partial-Match Hit Rate — Measurement Report

**Date**: 2026-06-26  
**Plan**: `/home/gfy/.claude/plans/whimsical-stirring-thimble.md` (Direction #3 measurement)  
**Workload**: 60-case stratified sweep (manifest_500.json), 5 agents per task, segment_count=3, mode=`placeholder_knn_reuse`, Qwen2.5-3B-Instruct

## Headline

- **Requests sent**: 300 (60 cases × 5 agents)
- **Placeholder pool hits**: 0
- **Placeholder pool misses**: 0
- **Max pool size**: 0
- **Prefix-cache reuse ratio**: 0.0053 (10,472 / 1,972,750 tokens)
- **AST_ALIGN log rows**: 0

## Per-Agent Breakdown

| Agent | Requests | Pool Hits | Pool Misses | Max Pool Stored | Mean Cached Ratio | Mean TTFT (ms) |
|-------|---------:|----------:|------------:|----------------:|-------------------:|---------------:|
| `auditor` | 60 | 0 | 0 | 0 | 0.0053 | 498 |
| `debugger` | 60 | 0 | 0 | 0 | 0.0053 | 505 |
| `implementer` | 60 | 0 | 0 | 0 | 0.0053 | 505 |
| `reviewer` | 60 | 0 | 0 | 0 | 0.0053 | 492 |
| `verifier` | 60 | 0 | 0 | 0 | 0.0053 | 486 |

## Decision

**POOL INACTIVE — AST-aligned hit rate is UNDEFINED (0/0).** The placeholder anchor pool never accumulated a single entry across 300 requests (60 cases × 5 agents). The prerequisite for measuring AST-aligned partial-match hit rate — pool activation — is unmet. Direction #3 cannot be evaluated yet.

## Interpretation

The placeholder anchor pool never accumulated an entry across 300 requests. This reproduces the Gate 2 finding (`results/ttft_agenttemplatekv/giant_pandas_50_20260626/rows.csv`) on a different manifest (60-case stratified instead of 50 pandas).

**Root-cause hypothesis** (from Gate 2 debug):
1. The `placeholder_anchor_pool` requires the k-NN body to fire (`SGLANG_PLACEHOLDER_KNN_MATCH=1` and family — set correctly here).
2. The k-NN body short-circuits when the prefix cache fully satisfies the request (`cached_tokens ≈ prompt_tokens`); no `insert()` runs, so `_store_placeholder_anchor_kv` is never called.
3. `vary_code=True` would break the prefix hit, but the slot-text embedding diverges from the warm_planner's stored embedding (cos drops below 0.85) → no match anyway.
4. **Pool activation is gated on the k-NN body actually firing**, which in this configuration it never does.

**Decision gate from the plan:**

| AST-aligned hit rate | Decision |
|---|---|
| ≥ 30% | Direction #3 worth pursuing (8-12 weeks) |
| 10-30% | Marginal; combine with cache-ordering first (option B/D) |
| < 10% | Pivot to production hardening (option B) |
| **UNDEFINED (pool inactive)** | **Fix pool activation bug FIRST, then re-measure** |

**Recommended next step** (option B from the user's earlier menu): fix the placeholder pool activation bug before pursuing any new direction. Specifically:
1. Add server-side print at `radix_cache.py:1386` to log when the F1 check (`SGLANG_PLACEHOLDER_STORE_MIN_F1`) drops entries — confirm or refute the F1-fail hypothesis.
2. If F1 is the issue, lower `SGLANG_PLACEHOLDER_STORE_MIN_F1` from 0.60 to 0.0 (bypass) and re-run the 60-case sweep.
3. If F1 is not the issue, instrument `_try_placeholder_knn_lossy_match` at `radix_cache.py:2313` to log the gating decisions (env-var check, spans check, embedder load, cost guard).
4. Once the pool activates, this measurement driver can be re-run to produce a meaningful AST-aligned hit rate.

**Why not just implement Direction #3?**
Direction #3 (AST-boundary chunked prefill) builds on top of the placeholder k-NN body. Without a working pool, the AST-boundary chunker has no pool to look up against. Building it now would be premature — the measurement is the right next step *after* the pool is fixed.