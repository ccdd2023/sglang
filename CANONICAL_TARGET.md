# CANONICAL TARGET — sglang-kvflow (2026-06-27, updated 2026-06-29)

> **THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR PROJECT GOAL.**
> All other docs (HANDOFF.md, KVFLOW_OVERVIEW.md, PHASE2_*.md, etc.)
> either supersede to this file or are explicitly superseded by it.
> Read this FIRST when starting any new session.

---

> ## ⚠️ 2026-06-29 UPDATE — supersedes the L4 "~1.49× production-ready" claim below
>
> The 2026-06-27 status (L4 "Production-ready, ~1.49×") is **FALSIFIED** by
> subsequent work on branch `fix/placeholder-pool-activation` (commits
> `aa08cfac5` → `9339f70b5`). The corrected picture:
>
> - **L4 byte-exact alone yields 0 reuse** on giant-codebase (flat-prefix API
>   ceiling — see memory `l4-contiguity-ceiling-2026-06-28`). The "~1.49×"
>   was the broken over-copying path. L4 is the *match policy*, not a
>   speedup source by itself.
> - **Non-prefix KV reuse is fundamentally lossy-or-slow** (proven; KV is
>   context-dependent). See memory `c2-fundamental-limits-2026-06-28`.
> - **New code-aware algorithm = AST-Gated L3 + offset alignment (+ C2 fallback).**
>   Controlled A/B (fair 65k pool, same prompts, token-F1 vs lossless):
>   - **Vary-code: F1 0.240 (> L3 0.193), speedup 1.448× (≥ L3 1.441×) — BOTH bars met, accuracy strictly better.** (commit `4c1f77fa8`)
>   - Same-code: F1 0.402 (= L3), speedup 1.243× (= L3 matched baseline) — both bars met, no regression (offset gate does not fire).
>   - **Both bars (good-enough speedup AND accuracy ≥ general L3) met in BOTH scenarios.** The vary-code speed bar — the last unmet condition — is closed by the offset-aligned AST gate (`SGLANG_L3_AST_GATE_OFFSET=1`), which makes the fast L3 whole-slot copy fire under vary-code (previously rejected on position-0 lcp=0 → slow C2 fallback).
> - **L3 stays OFF by default** but is the explicit *general-algorithm
>   baseline* for the fairness comparison (the user's goal frames it so).
> - **Visual summary**: [`results/contribution_summary_20260629.html`](./results/contribution_summary_20260629.html).
>
> The original 2026-06-27 body below is retained for archaeology; treat the
> L4 "~1.49× production-ready" line and the "Next: smoke run" item as
> superseded.

---

## The Project

**Name**: sglang-kvflow (AgentTemplateKV)
**Type**: SGLang fork for Coding Multi-Agent System serving
**Paper venue**: EuroSys 2026 (paper repo at `/home/gfy/CodeMAS_Project/AgentTemplateKV_Paper`)
**Active branch**: `fix/placeholder-pool-activation`
**HEAD**: `fea64d4cc` (Direction #3 Phase C/D landed)

---

## THE SINGLE CURRENT GOAL

**Make Coding-MAS serving fast and correct via code-aware KV cache
reuse, with byte-exact safety as the non-negotiable invariant.**

Concretely:

1. **TTFT speedup** vs `prefix_cache_only` baseline.
2. **Byte-exact correctness** — variable renames, comment edits, signature
   changes must NOT trigger K/V reuse from the OLD version. (Failure mode
   is silent: tests pass, output reads correctly, runtime behavior diverges.)
3. **Production-safe defaults** — new features ship OFF unless explicitly
   enabled via env var or CLI flag.

---

## The 4-Layer Cache Architecture

| Layer | Mechanism | Status | Speedup |
|---|---|---|---|
| **L1** | Radix tree prefix cache (token-level byte-exact) | Production | 1.20× |
| **L2** | Whole-slot byte-exact with RoPE rotation (Phase 2.4) | Production | 1.31× (cumulative) |
| **L3** | Placeholder k-NN body (MiniLM semantic) | **DEPRECATED 2026-06-27** | ~~1.65×~~ (research only) |
| **L4** | AST-boundary chunked prefill (Direction #3) | Production-ready | ~1.49× (cumulative) |

Production deployment baseline: **L1 + L2 only = 1.31×**.
Production deployment target: **L1 + L2 + L4 chunk = ~1.49×**.

---

## Current Direction — Direction #3 (L4 Chunk Pool)

**Status**: Phase A/B/C/D all landed (commits `7fb1a5bb2`, `8599afcfc`, `5197823bf`, `fea64d4cc`).
**Next**: giant-codebase smoke run with both env vars set; verify hit_rate > 0.

### Why Direction #3 (and not L3 / KVCOMM / pre-rotation)?

| Alternative | Reason rejected |
|---|---|
| **L3 (MiniLM k-NN body)** | `histogram` → `hist` has cos ≈ 0.95 but byte content differs. Reusing old K/V gives the model confused representation of new prompt. **Silent failure mode.** Formally deprecated 2026-06-27. |
| **KVCOMM offset blend** | Pre-Direction-#3 attempt at L2 reuse tuning. Performance plateau; doesn't address the 8.2% boundary-drift case L4 chunk handles. |
| **Phase 2.7 pre-rotation** | Amortizes RoPE delta cost via pre-rotated head K. Sound but only an optimization on top of an unsafe path — the unsafe path was L3, so pre-rotation is also retired. |
| **v44 selective AST reuse** | Effectively L3 with stricter AST-alignment guard. 91/91 byte-equal SWE-bench result was the validation; **but** the 8.2% non-byte-identical hits exposed the silent failure mode that triggered deprecation. |

Direction #3 preserves byte-exact invariant at **chunk** granularity
(function/class boundaries) — the only safe level that still recovers
significant speedup.

---

## Explicitly Deprecated / Out-of-Scope

| Item | Deprecation date | Reason |
|---|---|---|
| `_try_placeholder_knn_lossy_match` (L3 body) | 2026-06-27 | MiniLM cos ≥ 0.85 cannot distinguish variable rename from whitespace drift. See `l3-placeholder-knn-deprecated.md` memory. |
| Default `SGLANG_PLACEHOLDER_KNN_MATCH=1` | 2026-06-27 | Flipped to `0` in commit `8064ea450`. Re-enable only with `--enable-research-l3` for giant-codebase benchmark. |
| v44 selective AST reuse as production feature | 2026-06-27 | Superseded by Direction #3 chunk pool. v44 cycle remains as research evidence (10 memory entries consolidated into `v44-cycle-history.md`). |
| KVCOMM offset blend | pre-Jun-26 | Plateaued; no longer active. |
| Phase 2.7 pre-rotation | 2026-06-27 | Was an optimization on top of unsafe L3 path. Branch `phase-2.7-prerot` retained for code archaeology only. |
| Phase 1/2 context-aware KV reuse | pre-Jun-23 | Foundation work; subsumed by Phase 2.4 (whole-slot) and L4 chunk pool. Branch `feature/context-aware-kv-reuse` retained for code archaeology. |

---

## Operational Caveats (must be honored in any new run)

| Caveat | Source memory | Action |
|---|---|---|
| `--force-evict` required for Phase 2+ runs | `_delete-leaf-bug-2026-06-24.md` | Always add `--force-evict --disable-overlap-schedule --max-running-requests 1` for benchmarks > 3 cases. |
| `--no-vary-code` for giant-codebase | (in HANDOFF.md) | Pass `--no-vary-code` for any new giant-codebase run; default `--vary-code` corrupts the placeholder pool match. |
| 100-case pass@1 needs --force-evict | `100-case-force-evict-fix.md` | Pre-Phase 2 OOM on 100-case SWE-bench without it. |

---

## Active Files / Directories

| Path | Why current |
|---|---|
| `HANDOFF.md` | Active session handoff doc; updated 2026-06-27 with Phase C/D + L3 deprecation. |
| `CANONICAL_TARGET.md` | This file. The target statement. |
| `python/sglang/srt/mem_cache/radix_cache.py` | Active radix cache with Phase A/B/C/D + L3 (deprecated). |
| `python/sglang/srt/mem_cache/ast_chunker.py` | Direction #3 Phase A (server-side chunker). |
| `test/registered/unit/mem_cache/test_placeholder_chunk_pool*.py` | Phase B/C/D tests. |
| `test/registered/unit/mem_cache/test_ast_chunker.py` | Phase A tests. |
| `benchmark/multi_workflow/bench_giant_codebase_reuse.py` | Giant-codebase SWE-Smith benchmark driver. |
| `results/giant_codebase/` | Active benchmark output dir (3 runs so far). |
| `results/direction_3_phase_c_d_20260627.html` | Phase C/D HTML report. |
| `results/ttft_agenttemplatekv/` | Per-agent TTFT baseline + post-activation. |
| `results/ast_alignment_*_20260626/` | Direction #3 evidence (Phase A hit-rate progression). |
| `results/swe_*_v44_20260624T*/` | v44 evidence (kept for paper). |
| `results/swebench_local_envs/` | 21G of cloned SWE-bench repos — paper reproducibility infra. Keep. |

---

## Supersession Hierarchy

```
CANONICAL_TARGET.md          ← read first
        ↓
HANDOFF.md                   ← current session state
        ↓
whimsical-stirring-thimble.md (plan) ← index entry
        ↓
KVFLOW_OVERVIEW.md (rewrite) ← historical snapshot
PHASE2_FINDINGS.md (rewrite) ← Phase 2 evidence
PHASE2_PLAN.md (rewrite) ← historical plan
PLACEHOLDER_KNN_STATUS.md (rewrite) ← L3 deprecation notice
SESSION_HANDOFF_2026-06-23.md (rewrite) ← superseded
docs/experiment_plan.md (rewrite) ← historical
docs/kvflow_priority_fix_progress.md (rewrite) ← historical
docs/lmcache_baseline_replay.md (rewrite) ← runbook
```

---

## How to apply this

- **Starting a new session on sglang-kvflow**: Read CANONICAL_TARGET.md,
  then HANDOFF.md, then the active plan file.
- **Asking "what should I work on next?"**: Direction #3 Phase E
  (whitespace-drift tolerance) is the next research extension if
  telemetry shows `skip_byte_drift_count` is dominant. Otherwise the
  active work is the giant-codebase smoke run validating Phase C/D.
- **Confused by an old doc that contradicts this**: That doc is stale.
  Refer to CANONICAL_TARGET.md. If a doc still references
  `SGLANG_PLACEHOLDER_KNN_MATCH=1` as default or `phase-2.7-prerot` as
  active branch, it's wrong.
- **Want to revisit an old approach (L3 / KVCOMM / pre-rotation)?**
  Don't, unless the user explicitly asks. They were deprecated for
  documented reasons.
