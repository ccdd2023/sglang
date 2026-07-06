# CANONICAL TARGET — sglang-kvflow (2026-07-02)

> **THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR PROJECT GOAL AND STATE.**
> Read this FIRST when starting any new session. For the full development
> timeline, results tables, and the proven fundamental limit, see
> [`results/CODE_AWARE_LOSSY_KV_PROGRESS.md`](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md).
> For current session state, see [`HANDOFF.md`](./HANDOFF.md).
> For a paste-into-new-session prompt, see [`NEXT_SESSION_PROMPT.md`](./NEXT_SESSION_PROMPT.md).
> For the most recent precompute + Phase 7 cycle, see
> [`results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`](./results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md).

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

## Current State (2026-07-02) — honest

The code-aware lossy KV reuse path was iterated through six mechanisms
(L3 MiniLM → L4 AST chunk → cross-position slot_id fix → C2 CacheBlend
gap-prefill → MULTI_SLOT copy → PRECOMPUTE pipeline). The speed bar is
**solved by MULTI_SLOT**; the accuracy bar is **the open problem**, and
its root cause is proven. Precompute validates the end-to-end pipeline
as a building block but does not change the fundamental limit.

| Mechanism | Reuse (7B, full-share) | p50 TTFT | F1 vs lossless | Status |
|---|---|---|---|---|
| lossless (no reuse) | 0 tok | 932 ms | 1.000 | reference |
| single-slot staged (1 slot ≈ 1400 tok) | ~1300 tok | ~820 ms | 0.461 | valid-but-different |
| **MULTI_SLOT (5 slots ≈ 7100 tok)** | ~7100 tok | **124 ms (7.5×)** | **0.000** | garbage |
| precompute SYNC (host pool) | ~870 tok | ~923 ms | 0.374 | lossy, no speedup |
| precompute LAYERED (host pool) | ~870 tok | ~918 ms | 0.508 | lossy + load_stream side-effect |
| precompute DEVICE-RESIDENT (diagnostic) | ~830 tok | ~960 ms | 0.447 | lossy, slower than lossless |

- **Speed bar: MET.** MULTI_SLOT breaks the 1-slot reuse ceiling (97%
  utilization, 7.5× vs lossless for hitters). Precompute's diagnostic
  device-resident mode (zero H2D transfer) is **slower** than lossless —
  proving transfer is not the bottleneck; the read-path itself costs
  ~867-tok worth of overhead at this scale.
- **Accuracy bar: NOT MET for substantial reuse.** Reusing one slot
  (≈1400 tok) gives F1≈0.46 (valid-but-different outputs). Reusing five
  slots (≈7100 tok) gives F1=0.000 (empty/garbage). Precompute's
  F1=0.374 is consistent with the same fundamental limit; the LAYERED
  F1=0.508 is documented as a real but mechanism-specific load-stream
  side-effect (see Phase 7 below).

### Phase 7 verdict — LAYERED F1=0.508 is not a transferable correctness fix

7-way A/B (3 "fence differently on SYNC" experiments + agent_count=1
ablation + 1×1 byte-compare dump): all three "make SYNC behave like
LAYERED" hypotheses **FALSIFIED**. The gap is a `load_stream` +
`LayerDoneCounter` side-effect, not a default-stream race or per-layer
wait pattern. Even if closed (0.508 → ~0.55), accuracy bar is still not
met (1.000 lossless). **Decision: not pursuing further.** See
`results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`.

### The proven fundamental limit: cross-context KV loss

Raw copy + RoPE of KV across a different prefix is **lossy**. KV at
layers > 0 encodes the preceding prefix; when segments are reused under a
new prefix (different agent role / rotation order / leading gap), the
copied KV is stale. This is confirmed with data, not theory:
single-slot (1400 tok) → F1 0.46; multi-slot (7100 tok) → F1 0.00;
precompute SYNC (870 tok) → F1 0.374. The ~52 zeroed inter-slot header
tokens are NOT the cause — the 7100 tok of stale slot KV is.

See memory `c2-cacheblend-lossy-not-safe`, `c2-fundamental-limits`,
`multi-slot-copy-2026-07-01`, `precompute-kv-ab-2026-07-02`.

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
| **PRECOMPUTE** | Offline AST-aware KV precompute → CPU host pool → async CPU→GPU reuse | Implemented (default OFF) | end-to-end A/B: F1=0.374 lossy, no speedup; validated as building block |

**There is no current "production-ready" speedup claim.** Earlier claims
(L4 "~1.49× production-ready", AST-gated L3 "1.448× both bars met") are
RETRACTED — the first was the broken over-copying path, the second
conflated radix prefix with code-aware reuse. The LAYERED F1=0.508 from
the 2026-07-02 precompute cycle is also **not** a transferable correctness
fix — it is a load_stream + LayerDoneCounter side-effect (Phase 7
investigation, all fence-hypotheses falsified). See
`fair-measurement-prefix-conflation-2026-06-30` and
`results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`.

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
- **F1 measurement reads `outputs.jsonl`'s `output_text`.** Do not trust
  `rows.csv`'s `output` / `output_token_f1_vs_baseline` columns — they
  may be empty/placeholder. (Memory: `precompute-kv-ab-2026-07-02`.)
- **Precompute artifacts (`results/codebase_kv/`) stay gitignored** —
  1.2 GB / run, regenerated by `scripts/precompute_codebase_kv.py`.

## Operational caveats (must honor in any run)

| Caveat | Action |
|---|---|
| `_delete_leaf` assertion crashes normal-evict under > 3 cases | Add `--disable-overlap-schedule --max-running-requests 1` (>3 cases; `launch_server` auto-adds these). **`--force-evict` is NOT a real server flag** (older docs are wrong). (Memory: `_delete-leaf-bug-2026-06-24`.) |
| `--vary-code` mutates source across runs → unrepeatable | Use `--no-vary-code` for measurement. |
| MULTI_SLOT copied spans occasionally leak (not radix-evictable) | Run with `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` (warn-only) to survive; leak is bounded (pool exhaustion → 0 reuse, graceful). |
| Precompute host pool pinned-device leak alarm | Split `placeholder_chunk_pool_pinned_tokens` into device-only counter (commit `628aeab83`); otherwise `_check_radix_cache_memory` false-flags host-pool entries as device leaks. |
| 21G `swebench_local_envs/` | gitignored; do not re-track. |
| 1.2 GB/run `results/codebase_kv/` | gitignored; regenerate with `scripts/precompute_codebase_kv.py`. |

---

## Active files / directories

| Path | Why current |
|---|---|
| `HANDOFF.md` | Active session handoff (refreshed 2026-07-02). |
| `NEXT_SESSION_PROMPT.md` | Paste-into-new-session prompt. |
| `results/CODE_AWARE_LOSSY_KV_PROGRESS.md` | Master progress + timeline + results + fundamental limit. |
| `results/kvcomm_ab/CROSS_POSITION_REPORT.md` | Cross-position fix + 7B + partial-share results. |
| `results/kvcomm_ab/KV_BREAKDOWN_REPORT.html` | Visual KV-breakdown + multi-slot results. |
| `results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md` | Phase 7 LAYERED F1 investigation (most recent cycle). |
| `results/kvcomm_ab/precompute_ab_report/COMPARISON.txt` | 7-way precompute A/B table. |
| `scripts/precompute_codebase_kv.py` | Offline KV precompute extractor. |
| `python/sglang/srt/mem_cache/codebase_kv_loader.py` | Server-start disk→CPU loader + read-path residency branching. |
| `python/sglang/srt/m_cache/radix_cache.py` | L1/L2/L3(deprecated)/L4/C2/MULTI_SLOT/PRECOMPUTE implementation + Phase 7 hooks (default OFF). |
| `python/sglang/srt/mem_cache/ast_chunker.py` | L4 server-side AST chunker. |
| `benchmark/multi_workflow/bench_giant_codebase_reuse.py` | giant-codebase benchmark driver (`--partial-share`, `--position-shift`, `--precompute-*`). |
| `benchmark/multi_workflow/analyze_fair_ab.py` | Fair A/B analyzer (decomposed counters, source-exclusion, parity gate). |
| `results/kvcomm_ab/run_*.sh` | Reproducible launchers for every measured config. |
| `test/registered/unit/mem_cache/test_ast_chunker.py`, `test_placeholder_chunk_pool*.py`, `test_codebase_kv_loader.py` | L4 + precompute tests. |
| `results/swebench_local_envs/` | 21G SWE-bench reproducibility infra (gitignored). |
| `results/codebase_kv/` | 1.2 GB/run precomputed KV (gitignored; regenerate). |

---

## How to apply this

- **Starting a new session**: read `CANONICAL_TARGET.md` (this file), then
  `HANDOFF.md`, then `results/CODE_AWARE_LOSSY_KV_PROGRESS.md`. For a
  paste-into-prompt summary, use `NEXT_SESSION_PROMPT.md`.
- **"What should I work on next?"**: the only identified path to BOTH
  bars is true CacheBlend (attention recompute). Precompute is the
  prerequisite for it (now built). The concrete next speed lever
  is precompute async overlap via SGLang's HiCache `LayerDoneCounter`
  mechanism (see plan Phase 4A-REVISED). Both await user sign-off.
- **Confused by an old doc that contradicts this**: that doc is stale.
  Refer here. If a doc still references `SGLANG_PLACEHOLDER_KNN_MATCH=1`
  as default, "~1.49× production-ready", "1.448× both bars met", or
  "LAYERED F1=0.508 as a correctness fix", it is wrong / retracted.
- **Want to revisit L3 / pre-rotation / KVCOMM offset blend?** Don't,
  unless the user explicitly asks — deprecated for documented reasons.
