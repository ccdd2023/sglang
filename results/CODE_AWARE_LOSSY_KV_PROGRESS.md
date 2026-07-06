# Code-Aware Lossy KV Reuse — Progress, Timeline & Results (2026-07-02)

> **Master document** for the code-aware KV reuse workstream in sglang-kvflow.
> This is the single place for the development timeline, all measured results,
> and the proven fundamental limit. Companion to
> [`CANONICAL_TARGET.md`](../CANONICAL_TARGET.md) (goal/state) and
> [`HANDOFF.md`](../HANDOFF.md) (session state).
>
> Most recent cycle (precompute + Phase 7): see
> [`kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`](kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md).
> Visual companion: [`kvcomm_ab/KV_BREAKDOWN_REPORT.html`](kvcomm_ab/KV_BREAKDOWN_REPORT.html).

---

## 1. Goal & regime

**Goal.** Make Coding-MAS serving fast and correct via code-aware KV cache
reuse. Two bars, both required:

1. **Speed** — TTFT speedup vs a `prefix_cache_only` / lossless baseline.
2. **Accuracy** — under the same prompts, accuracy need not be worse than a
   general (non-code-aware) reuse algorithm. Measured as real token-F1 vs a
   lossless reference.

**Constraint.** Speedup must come **only from more reuse** — no KV-cache
scheduling tricks.

**Supported regime.** **KVCOMM byte-exact cross-position copy + RoPE**:
the same prompt content at a different absolute position, copied with a
RoPE delta (`new_pos - old_pos`). **AST chunking** is the coding-specific
optimization on top (per-function/class granularity so a partially-changed
slot still reuses the unchanged functions). MiniLM semantic k-NN (L3) is
**not** the regime — deprecated.

---

## 2. Development timeline

| Date | Step | Outcome |
|---|---|---|
| 2026-06 (Phase 1/2) | Context-aware KV reuse, L2 whole-slot byte-exact + RoPE | Foundation. L2 = the safe-ish whole-slot copy path. |
| 2026-06-22 → 25 (v44) | L3 placeholder MiniLM k-NN body; selective AST reuse; 91/89/27 byte-equal SWE-bench | **Deprecated 2026-06-27.** MiniLM cos≥0.85 cannot distinguish variable-rename from whitespace drift → 8.2% silent failure. Evidence kept for the paper. |
| 2026-06-26 | Placeholder pool activation bugs found+fixed; giant-codebase 50-task × 5-agent benchmark (pandas, SWE-Smith) | Pool finally fires; 1.31× vs prefix-only, 20% reuse. |
| 2026-06-27 | Direction #3 (L4 AST chunk pool) Phase A/B/C/D land | Byte-exact per-function reuse. |
| 2026-06-28 | L4 contiguity ceiling; C2 CacheBlend attempted | Correct L4 yields **~0 reuse** on giant-codebase (flat-prefix API ceiling). The "~1.49×" was the broken over-copying path. C2 raw-copy+RoPE proven lossy. |
| 2026-06-29 | AST-gated L3 + offset alignment | Claimed "1.448× both bars met" — **RETRACTED 2026-06-30** (cached_tokens conflated radix prefix + code-aware reuse; warmup gave source agent artificial reuse). |
| 2026-06-30 | Fair-measurement redesign (decomposed counters A1-A6, source-agent exclusion, warmup-parity gate, ephemeral copy) | Measurement now honest. Same regime: L2/L4/C2 all yield **0 reuse** — root cause = positional `slot_id`. |
| 2026-06-30 | **Cross-position `slot_id` fix** (`code_base:<file>` not positional idx) | **Unblocked** byte-exact KVCOMM reuse (0 → 7-13/16 reusers hit). |
| 2026-06-30 | C2-direct gap-ZEROING vs STAGED gap-prefill (3B) | Gap-zeroing: 2.5× speed but F1=0.008 (garbage — zeroes the instruction). Staged: 1.085×, F1=0.376 (valid-but-different). |
| 2026-06-30 → 07-01 | 7B-Coder scaling (larger model → bigger copy-vs-prefill delta) | L2-staged 1.144× / F1 0.461; L4-staged 1.044× / F1 0.399. Whole-slot (L2) beats AST (L4) in full-sharing. |
| 2026-07-01 | AST chunker gap-fill (`SGLANG_CHUNK_FILL_GAPS=1`) | L4 captures module-level code: 1.074× / F1 0.414. Still below L2 in full-sharing (granularity = per-chunk overhead). |
| 2026-07-01 | Partial-share + rotation scenario (AST's niche) | 12-task/42-case: AST F1 0.622 > L2 0.513 (accuracy bar MET), but AST 0.96× (speed bar NOT met — per-chunk overhead cancels reuse). |
| 2026-07-01 | **MULTI_SLOT copy** (`SGLANG_CACHEBLEND_MULTI_SLOT=1`) | **Breaks the 1-slot ceiling**: 97% utilization (5 slots ≈ 7100 tok), hitter p50 TTFT 124 ms = **7.5× vs lossless**. **BUT F1=0.000** (garbage) — definitively confirms cross-context KV loss. |
| 2026-07-02 | **Precompute pipeline** (Phases 1–3, 4B/4C, commit `628aeab83`) | Offline AST-aware KV precompute → CPU host pool → async CPU→GPU reuse. 5 case × 5 agent end-to-end A/B: reuse 867 tok/row, **real F1=0.374 (LOSSY)**, no speedup (TTFT ~923ms ≈ lossless 948ms). Precompute validated as a building block; raw-copy+RoPE still lossy. |
| 2026-07-02 | Phase 6 device-resident diagnostic | KV straight to GPU (zero H2D) → 960 ms, **slower than lossless 948 ms** and slower than host-pool SYNC 923 ms. **Transfer is NOT the bottleneck** — read-path overhead (move_kv_cache + RoPE + alloc) cancels the 867-tok reuse gain at this scale. F1=0.447 (same default-stream RoPE as SYNC; between SYNC and LAYERED). |
| 2026-07-02 | Phase 7 LAYERED F1=0.508 investigation | 7-way A/B + 3 "fence differently on SYNC" experiments + agent_count=1 ablation. **All three fence hypotheses FALSIFIED** (default-stream race / per-layer event-wait / record_stream collision). SYNC vs LAYERED on the same input produces F1=0.714 (material divergence, not noise). **Verdict**: LAYERED F1=0.508 is real, stable, mechanism-specific (load_stream + LayerDoneCounter side-effect), NOT a transferable correctness fix. Decision: not pursuing further. Full report [`ANOMALY_FINAL.md`](kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md). |

---

## 3. The 4-layer cache (honest)

| Layer | Mechanism | Reuse gate | Status |
|---|---|---|---|
| **L1** | Radix prefix cache (token-level byte-exact, same position) | exact prefix | Production — the only *safe* reuse; baseline |
| **L2** | Whole-slot byte-exact + RoPE (cross-position) | exact slot text + content-derived `slot_id` | Implemented; lossy when substantial (F1≈0.46 @ 1 slot) |
| **L3** | Placeholder k-NN body (MiniLM semantic) | cos≥0.85 | **DEPRECATED** — research only; `SGLANG_PLACEHOLDER_KNN_MATCH=0` |
| **L4** | AST-boundary chunk reuse (byte-exact per function/class) | exact chunk (signature + byte span + token ids) | Implemented; accuracy advantage in partial-sharing, not speed |
| **C2 / MULTI_SLOT** | CacheBlend gap-prefill + multi-slot batched copy | exact chunk; leading gap staged real, internal gaps zeroed | Implemented; 7.5× speed but F1=0.000 (lossy) |
| **PRECOMPUTE** | Offline AST-aware KV precompute → CPU host pool → async CPU→GPU reuse | exact chunk (location="host" / "device") | Implemented (default OFF); F1=0.374 lossy, no speedup; validated as building block |

> No current "production-ready" speedup claim. The L4 "~1.49×" and the
> AST-gated-L3 "1.448× both bars met" are both RETRACTED. The LAYERED
> F1=0.508 from the precompute cycle is documented as a real but
> mechanism-specific side-effect (Phase 7), NOT a transferable correctness
> improvement.

---

## 4. Results

All numbers are from the **fair A/B harness** (`analyze_fair_ab.py`):
decomposed counters (radix prefix separated from code-aware reuse),
source agent excluded from the speedup average, warmup-parity gate,
real token-F1 vs a lossless reference. Scenario: giant-codebase (pandas),
`--position-shift --no-vary-code` (same code, cyclically rotated slot
order per agent), MiniLM L3 + offset-gate OFF, distinct per-agent
`cache_salt`. Reusers only (agent 1 = source, excluded).

### 4a. Full-share position-shift — the multi-slot ceiling test (7B-Coder)

| config | reuse (tok) | p50 TTFT | speedup vs lossless | F1 vs lossless |
|---|---|---|---|---|
| lossless (reference) | 0 | 932 ms | 1.0× | 1.000 |
| single-slot staged (L2, 1 slot) | ~1300 | ~820 ms | ~1.14× | 0.461 |
| **MULTI_SLOT (5 slots)** | **~7100** | **124 ms** | **7.5×** | **0.000** |

The speed bar is met (7.5×). The accuracy bar is catastrophically failed
for substantial reuse (F1=0.000).

### 4b. Single-slot staged — 3B vs 7B (full-share, cross-position fix)

| model | config | reuse hit | avg codeaware | per-case speedup (avg / p50) | F1 |
|---|---|---|---|---|---|
| 3B | L2 whole-slot STAGED | 13/16 | 1252 | 1.085× / 1.052× | 0.376 |
| 3B | L4 AST STAGED | 11/16 | 642 | 0.998× / 0.958× | 0.388 |
| **7B** | **L2 whole-slot STAGED** | **13/16** | **1259** | **1.136× / 1.144×** | **0.461** |
| 7B | L4 AST STAGED | 16/16 | 1069 | 1.067× / 1.044× | 0.399 |
| 7B | L4 AST STAGED + FILL_GAPS | 14/16 | 1312 | 1.074× | 0.414 |

Larger model amplifies both speedup and F1 (7B prefill ~2× costlier →
reuse saves more; coder model more robust to cross-context loss). In
**full-sharing**, whole-slot (L2) beats AST (L4) on both axes — AST's
granularity is a disadvantage (N copies > 1 copy) and the chunker misses
module-level code without gap-fill.

### 4c. Partial-share + rotation — AST's niche (7B-Coder, 12-task/42-case)

| metric | L2 whole-slot | L4 AST+fill | verdict |
|---|---|---|---|
| p50 TTFT (aligned 42 cases) | 927 ms | 966 ms | L4/L2 = **0.96× (AST slower)** |
| per-case speedup vs lossless (p50) | 1.046× | 1.004× | both barely beat lossless |
| avg codeaware reused | 1399 | 516 | AST reuses less (only shared functions) |
| F1 vs lossless | 0.513 | **0.622** | **AST more accurate** |

In partial-sharing, AST meets the **accuracy** bar (copies only shared
functions → smaller cross-context copies → less loss) but NOT the speed
bar (per-chunk alloc+move+RoPE overhead cancels the reuse advantage at
this scale). The earlier 4-task "AST 1.205× faster" was noise.

### 4d. Gap-handling mechanism (3B, full-share, cross-position)

| config | reuse hit | avg codeaware | speedup (avg / p50) | F1 |
|---|---|---|---|---|
| L2 whole-slot, gap-ZERO (C2-direct) | 7/16 | 2751 | 2.533× / 1.015× | 0.106 (reuse cases 0.008) |
| L4 AST, gap-ZERO | 5/16 | 386 | 1.015× / 0.987× | 0.197 |
| **L2 whole-slot, STAGED** | **13/16** | **1252** | **1.085× / 1.052×** | **0.376** |
| L4 AST, STAGED | 11/16 | 642 | 0.998× / 0.958× | 0.388 |

Gap-zeroing (zeroing the task instruction) → garbage; staging (prefill the
gap with real KV) → valid-but-different. This established that the
**leading gap must be real**, which MULTI_SLOT preserves (it only zeroes
the small internal inter-slot headers).

### 4e. Precompute end-to-end A/B (7B-Coder, 5 case × 5 agent, 32k tokens)

The most recent cycle. Detailed report in
[`kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`](kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md).

| config | reuse (tok) | p50 TTFT | F1 vs lossless | stream |
|---|---|---|---|---|
| lossless ref | 0 | 948 ms | 1.000 | n/a |
| precompute SYNC (host pool) | 867 | 923 ms | 0.374 | default |
| precompute ASYNC (event-wait) | 867 | 930 ms | 0.374 | default |
| **precompute LAYERED (host pool)** | **867** | **918 ms** | **0.508** | **`load_stream`** |
| precompute DEVICE-RESIDENT (diag) | 826 | **960 ms** | 0.447 | default |
| Phase 7: SYNC + DOUBLE SYNC | — | — | 0.375 | default (H1 falsified) |
| Phase 7: SYNC + per-layer event-wait | — | — | 0.375 | default (H3 falsified) |
| Phase 7: SYNC + record_stream | — | — | 0.375 | default (H1b falsified) |
| Phase 7: SYNC a1 (agent_count=1) | — | — | 0.369 | default (H4 falsified) |
| Phase 7: LAYERED a1 | — | — | 0.559 | `load_stream` |

**Speed verdict:** zero speedup (TTFT ≈ lossless for every residency mode).
Phase 6 device-resident (zero H2D) is **slower than lossless** — the
bottleneck is **not** the CPU→GPU transfer, it's the read-path itself
(move_kv_cache + RoPE + alloc overhead cancels the 867-tok reuse gain at
this scale). `benchmark/multi_workflow/analyze_fair_ab.py` was extended
with `precompute_reused_tokens` decomposition and a warmup-parity gate to
ensure agent 1 was not artificially advantaged.

**Accuracy verdict:** F1 ≤ 0.508 across all paths (still lossy). LAYERED's
0.508 is documented as a real but mechanism-specific load_stream +
LayerDoneCounter side-effect — **NOT a transferable correctness
improvement**. Phase 7 falsified all three "fence differently on SYNC"
hypotheses; even if closed to 0.55, accuracy bar is still not met (1.000
lossless).

### 4f. Precompute measurement caveat (FOOTGUN — read this)

The initial precompute A/B reported F1=1.000 lossless by mistake. The
`rows.csv` `output` column was **empty** (0/25 rows) because the driver
writes real outputs only to `outputs.jsonl`'s `output_text` field. The
naive F1 = empty-string-vs-empty-string = 1.0. **Always read
`outputs.jsonl` for F1 measurement; do NOT trust `rows.csv` `output` /
`output_token_f1_vs_baseline` columns** — they can be placeholders or
empty. See memory `precompute-kv-ab-2026-07-02`.

---

## 5. The proven fundamental limit: cross-context KV loss

**Claim.** Non-prefix KV reuse via raw copy + RoPE is **lossy**, and the
loss scales with the volume of reused KV.

**Mechanism.** KV at layers > 0 encodes the preceding prefix. When a
segment is reused under a *new* prefix (different agent role / rotation
order / leading gap), the copied KV is stale — it was computed against a
different preceding context. RoPE only fixes *positions*, not *content
conditioning*.

**Evidence (7B-Coder, full-share position-shift):**

| reused volume | F1 vs lossless | output character |
|---|---|---|
| ~0 tok (lossless) | 1.000 | correct |
| ~870 tok (precompute SYNC, host pool) | 0.374 | valid-but-different |
| ~830 tok (precompute DEVICE-RESIDENT) | 0.447 | valid-but-different |
| ~1400 tok (1 slot) | 0.461 | valid-but-different (on-topic, wrong specifics) |
| ~7100 tok (5 slots, MULTI_SLOT) | 0.000 | empty / garbage |

The ~52 zeroed inter-slot header tokens are **not** the cause — the 7100
tok of stale slot KV is. Confirmed by the single-slot → multi-slot
scaling: more reuse, worse F1, monotonic.

This is the same limit proven earlier by `c2-cacheblend-lossy-not-safe`
and `c2-fundamental-limits`: KV is context-dependent, so byte-exact text
≠ KV-exact when the prefix before the chunk differs. Radix (L1) is safe
because it reuses only an *identical* prefix (same position, same
preceding context); every non-prefix layer is lossy to some degree.
Precompute does not change this — F1=0.374 is consistent with the
fundamental limit; canonical-prefix preamble only helps the preamble
portion (~50–150 tok) stay lossless, the file content at shifted
positions stays lossy.

---

## 6. Conclusion & the only remaining path

- **Speed: solved.** MULTI_SLOT demonstrates that code-aware reuse alone
  can give 7.5× TTFT speedup (97% slot utilization). The speed bar is met.
  Precompute's diagnostic device-resident mode (zero H2D transfer) is
  *slower* than lossless → transfer is NOT the bottleneck; the read-path
  itself costs ~867-tok worth of overhead at this scale.
- **Accuracy: the open problem, root cause now proven.** Substantial
  cross-context KV reuse via raw copy + RoPE is lossy (F1 0.46 → 0.00 as
  reuse grows). The loss is fundamental to raw-copy+RoPE, not to the
  chunking/copy mechanism. Precompute (Phases 1–3, 4B/4C) confirms:
  F1=0.374 lossy, no speedup. Phase 7 investigation falsified all
  fence-difference hypotheses for LAYERED's F1=0.508 — that gain is a
  load-stream side-effect, not transferable.
- **Only remaining path to both bars: true CacheBlend** — for each copied
  chunk, recompute attention under the new context instead of raw-copy +
  RoPE. Expensive (the `c2-fundamental-limits` cost), **not yet built**,
  awaits explicit user sign-off. **Precompute is its prerequisite, now built.**
- **Concrete next speed lever**: precompute async overlap via SGLang's
  HiCache `LayerDoneCounter` mechanism (plan Phase 4A-REVISED, ~30 GPU-min).
  Same lossy F1, but may break the 0× speedup at the ~867-tok scale.
  Awaiting user sign-off.

Partial-sharing remains the regime where AST chunking has a genuine
accuracy advantage (F1 0.62 vs 0.51) — useful as a fallback / niche, but
it does not solve the substantial-reuse loss.

---

## 7. Artifacts index

**Reports**
- `kvcomm_ab/CROSS_POSITION_REPORT.md` — cross-position fix + 7B + partial-share results
- `kvcomm_ab/KV_BREAKDOWN_REPORT.html` — visual KV-breakdown + multi-slot results
- `kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md` — Phase 7 LAYERED F1 investigation (most recent cycle)
- `kvcomm_ab/precompute_ab_report/COMPARISON.txt` — 7-way precompute A/B table
- `kvcomm_ab/precompute_ab_report/SUMMARY.txt` — 5-way summary table
- `direction_3_phase_c_d_20260627.html` — L4 Phase C/D architecture deep-dive (valid)
- `project_progress_20260627.html` — 2026-06-27 snapshot (⚠️ RETRACTION banner; superseded)

**Reproducible launchers** (`kvcomm_ab/run_*.sh`)
- Lossless references: `run_lossless.sh`, `run_7b_lossless.sh`, `run_7b_ps_lossless.sh`, `run_7b_ps12_lossless.sh`, `run_7b_precompute_ab_lossless.sh`
- Single-slot staged: `run_l2_coarse_staged.sh`, `run_l4_staged.sh`, `run_7b_l2_staged.sh`, `run_7b_l4_staged.sh`, `run_7b_l4_fill.sh`, `run_7b_ps_l2_staged.sh`, `run_7b_ps_l4_fill.sh`, `run_7b_ps12_l2_staged.sh`, `run_7b_ps12_l4_fill.sh`
- C2-direct (gap-zero): `run_l2_coarse.sh`, `run_l4_c2.sh`
- **MULTI_SLOT**: `run_7b_multislot_l2.sh`, `run_7b_multislot_l4.sh`, `run_7b_ps12_multislot_l4.sh`
- **Precompute** (Phases 1–3, 4B/4C + 6 + 7): `run_7b_precompute_ab.sh` (LAYERED 5x5 default) + `..._sync_a1.sh`, `..._layered_a1.sh`, `..._sync_doublesync.sh`, `..._sync_perlayerwait.sh`, `..._sync_recordstream.sh`, `..._sync_bytecmp.sh`, `..._layered_bytecmp.sh` (8 variants)

**Per-config rows + outputs**: `kvcomm_ab/{lossless,7b_*,l2_*,l4_*,7b_multislot_l2,7b_precompute_ab_*}/` (`rows.csv`, `outputs.jsonl`, `FAIR_SUMMARY.md`, `sglang_server.log`). **F1 from `outputs.jsonl` `output_text`, NOT `rows.csv` `output`.**

**Analyzer reports**: `kvcomm_ab/report*/FAIR_AB_REPORT.md`,
`kvcomm_ab/precompute_ab_report/COMPARISON.txt`

**Precompute pipeline**
- `scripts/precompute_codebase_kv.py` — offline KV extractor (AST chunks → CPU → .bin)
- `python/sglang/srt/mem_cache/codebase_kv_loader.py` — server-start disk→CPU loader + read-path residency branching
- `python/sglang/srt/mem_cache/test_codebase_kv_loader.py` — unit tests

**Code**
- `python/sglang/srt/mem_cache/radix_cache.py` — L1/L2/L3(deprecated)/L4/C2/MULTI_SLOT/PRECOMPUTE + Phase 7 fence hooks (default OFF) + leak-detector fix
- `python/sglang/srt/mem_cache/ast_chunker.py` — L4 server-side AST chunker
- `python/sglang/srt/mem_cache/hiradix_cache.py` — server-start precompute loader trigger
- `python/sglang/srt/managers/scheduler.py` — `codebase_kv_producer_id` consumer wiring
- `benchmark/multi_workflow/bench_giant_codebase_reuse.py` — benchmark driver (`--precompute-*` flags)
- `benchmark/multi_workflow/analyze_fair_ab.py` — fair A/B analyzer (decomposed counters + `precompute_reused_tokens`)

**Paper evidence (kept, not active)**: v44-era `ttft_agenttemplatekv/` reports, `correctness_validation_report_20260624.md`, `swe_*_v44_*` runs.

---

*Last updated 2026-07-02 (precompute + Phase 7 cycle). Numbers are honest
fair-A/B measurements; all retracted claims are labeled as such. If a
number here contradicts an older doc, this file is correct.*
