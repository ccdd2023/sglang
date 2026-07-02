# CANONICAL TARGET — sglang-kvflow (2026-07-01)

> **THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR PROJECT GOAL AND STATE.**
> Read this FIRST when starting any new session. For the full development
> timeline, results tables, and the proven fundamental limit, see
> [`results/CODE_AWARE_LOSSY_KV_PROGRESS.md`](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md).
> For current session state, see [`HANDOFF.md`](./HANDOFF.md).

---

## The Project

**Name**: sglang-kvflow (AgentTemplateKV)
**Type**: SGLang fork for Coding Multi-Agent System (MAS) serving
**Paper venue**: EuroSys 2026 (paper repo at `/home/gfy/CodeMAS_Project/AgentTemplateKV_Paper`)
**Active branch**: `fix/placeholder-pool-activation`

---

## THE SINGLE CURRENT GOAL

**Make Coding-MAS serving fast and correct via code-aware KV cache
reuse, with acceptable accuracy under the same prompts as a general
algorithm.** Speedup must come ONLY from more reuse — no KV-cache
scheduling tricks. The two bars:

1. **TTFT speedup** vs `prefix_cache_only` / lossless baseline.
2. **Acceptable accuracy** — under the same prompts, accuracy need not be
   worse than a general (non-code-aware) reuse algorithm. Measured as real
   token-F1 vs a lossless reference.

---

## Current State (2026-07-01) — honest

The code-aware lossy KV reuse path was iterated through five mechanisms
(L3 MiniLM → L4 AST chunk → cross-position slot_id fix → C2 CacheBlend
gap-prefill → MULTI_SLOT copy). The speed bar is **solved**; the accuracy
bar is **the open problem**, and its root cause is now proven.

| Mechanism | Reuse (7B, full-share) | p50 TTFT | F1 vs lossless | Status |
|---|---|---|---|---|
| lossless (no reuse) | 0 tok | 932 ms | 1.000 | reference |
| single-slot staged (1 slot ≈ 1400 tok) | ~1300 tok | ~820 ms | 0.461 | valid-but-different |
| **MULTI_SLOT (5 slots ≈ 7100 tok)** | ~7100 tok | **124 ms (7.5×)** | **0.000** | garbage |

- **Speed bar: MET.** MULTI_SLOT breaks the 1-slot reuse ceiling (97%
  utilization, 7.5× vs lossless for hitters). The speed problem is
  demonstrably solvable with code-aware reuse alone.
- **Accuracy bar: NOT MET for substantial reuse.** Reusing one slot
  (≈1400 tok) gives F1≈0.46 (valid-but-different outputs). Reusing five
  slots (≈7100 tok) gives F1=0.000 (empty/garbage). The loss scales with
  reuse volume.

### The proven fundamental limit: cross-context KV loss

Raw copy + RoPE of KV across a different prefix is **lossy**. KV at
layers > 0 encodes the preceding prefix; when segments are reused under a
new prefix (different agent role / rotation order / leading gap), the
copied KV is stale. This is confirmed with data, not theory:
single-slot (1400 tok) → F1 0.46; multi-slot (7100 tok) → F1 0.00. The
~52 zeroed inter-slot header tokens are NOT the cause — the 7100 tok of
stale slot KV is.

See memory `c2-cacheblend-lossy-not-safe`, `c2-fundamental-limits`,
`multi-slot-copy-2026-07-01`.

### The only remaining path

**True CacheBlend** — for each copied chunk, recompute attention under
the new context (instead of raw-copy + RoPE). This is the one mechanism
that can give speed AND accuracy. It is expensive and **not yet built**.
Direction awaits explicit user sign-off (it is a fresh algorithmic
change beyond "optimize the copy path").

---

## The 4-Layer Cache Architecture (honest)

| Layer | Mechanism | Status | Note |
|---|---|---|---|
| **L1** | Radix tree prefix cache (token-level byte-exact, same position) | Production | the only *safe* reuse; baseline |
| **L2** | Whole-slot byte-exact + RoPE (cross-position) | Implemented | lossy when substantial; single-slot F1≈0.46 |
| **L3** | Placeholder k-NN body (MiniLM semantic) | **DEPRECATED 2026-06-27** | research only; `SGLANG_PLACEHOLDER_KNN_MATCH=0` default |
| **L4** | AST-boundary chunked reuse (byte-exact per function/class) | Implemented | lossy; accuracy advantage in partial-sharing (F1 0.62 vs L2 0.51), not speed |
| **C2 / MULTI_SLOT** | CacheBlend gap-prefill + multi-slot batched copy | Implemented | 7.5× speed but F1=0.000 (lossy) |

**There is no current "production-ready" speedup claim.** Earlier claims
(L4 "~1.49× production-ready", AST-gated L3 "1.448× both bars met") are
RETRACTED — the first was the broken over-copying path, the second
conflated radix prefix with code-aware reuse. See
`fair-measurement-prefix-conflation-2026-06-30`.

---

## Supported regime

- **KVCOMM byte-exact cross-position copy + RoPE**: same prompt content
  at a different absolute position, copied with RoPE delta
  (`new_pos - old_pos`). This is the supported lossy regime.
- **AST chunking** is the coding-specific optimization ON TOP of byte-exact
  reuse (per-function/class granularity so partially-changed slots still
  reuse the unchanged functions).
- **MiniLM semantic k-NN (L3) is NOT the regime** — deprecated; code is
  too sensitive to surface changes for MiniLM to gate safely.

---

## Non-negotiable invariants

- **L3 (MiniLM k-NN body) is OFF by default** (`SGLANG_PLACEHOLDER_KNN_MATCH=0`).
  Do not re-enable for production. (Memory: `l3-placeholder-knn-deprecated`.)
- **Byte-exact match is the reuse gate.** L2/L4 reuse fires only on exact
  text match (AST anchors decide alignment boundaries, not fuzzy match).
  Do not propose drift tolerance / MiniLM fallback at the reuse layer.
- **Speedup ONLY from more reuse.** No KV-cache scheduling for speed.
- **New features ship OFF by default** (env var / CLI flag to enable).
- **Experiment output goes to `results/`**, not `/tmp`. (Memory: `output-path`.)

## Operational caveats (must honor in any run)

| Caveat | Action |
|---|---|
| `_delete_leaf` assertion crashes normal-evict under > 3 cases | Add `--force-evict --disable-overlap-schedule --max-running-requests 1` (>3 cases; `launch_server` auto-adds the latter two). (Memory: `_delete-leaf-bug-2026-06-24`.) |
| `--vary-code` mutates source across runs → unrepeatable | Use `--no-vary-code` for measurement. |
| MULTI_SLOT copied spans occasionally leak (not radix-evictable) | Run with `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` (warn-only) to survive; leak is bounded (pool exhaustion → 0 reuse, graceful). |
| 21G `swebench_local_envs/` | gitignored; do not re-track. |

---

## Active files / directories

| Path | Why current |
|---|---|
| `HANDOFF.md` | Active session handoff. |
| `results/CODE_AWARE_LOSSY_KV_PROGRESS.md` | Master progress + timeline + results + fundamental limit. |
| `results/kvcomm_ab/CROSS_POSITION_REPORT.md` | Cross-position fix + 7B + partial-share results. |
| `results/kvcomm_ab/KV_BREAKDOWN_REPORT.html` | Visual KV-breakdown + multi-slot results. |
| `python/sglang/srt/mem_cache/radix_cache.py` | L1/L2/L3(deprecated)/L4/C2/MULTI_SLOT implementation. |
| `python/sglang/srt/mem_cache/ast_chunker.py` | L4 server-side AST chunker. |
| `benchmark/multi_workflow/bench_giant_codebase_reuse.py` | giant-codebase benchmark driver (`--partial-share`, `--position-shift`). |
| `benchmark/multi_workflow/analyze_fair_ab.py` | Fair A/B analyzer (decomposed counters, source-exclusion, parity gate). |
| `results/kvcomm_ab/run_*.sh` | Reproducible launchers for every measured config. |
| `test/registered/unit/mem_cache/test_ast_chunker.py`, `test_placeholder_chunk_pool*.py` | L4 tests. |
| `results/swebench_local_envs/` | 21G SWE-bench reproducibility infra (gitignored). |

---

## How to apply this

- **Starting a new session**: read `CANONICAL_TARGET.md` (this file), then
  `HANDOFF.md`, then `results/CODE_AWARE_LOSSY_KV_PROGRESS.md`.
- **"What should I work on next?"**: the only identified path to BOTH
  bars is true CacheBlend (attention recompute). Requires user sign-off.
- **Confused by an old doc that contradicts this**: that doc is stale.
  Refer here. If a doc still references `SGLANG_PLACEHOLDER_KNN_MATCH=1`
  as default, "~1.49× production-ready", or "1.448× both bars met", it is
  wrong / retracted.
- **Want to revisit L3 / pre-rotation / KVCOMM offset blend?** Don't,
  unless the user explicitly asks — deprecated for documented reasons.
