# HANDOFF — sglang-kvflow (2026-07-06)

> **READ FIRST**:
> 1. [CANONICAL_TARGET.md](./CANONICAL_TARGET.md) — single project goal + current state.
> 2. [NEXT_SESSION_PROMPT.md](./NEXT_SESSION_PROMPT.md) — prompt to paste into a new session.
> 3. [results/SESSION_WRAP.md](./results/SESSION_WRAP.md)
>    — **current state** (R19 / R26 / R27 3-way, post-2026-07-06 wrap).
> 4. [results/CODE_AWARE_LOSSY_KV_PROGRESS.md](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md)
>    — full development timeline, results tables, proven fundamental limit.
>
> The auto-loaded [memory index](../home/gfy/.claude/projects/-home-gfy/memory/MEMORY.md)
> has the key invariants.

---

## TL;DR

- **Branch**: `fix/placeholder-pool-activation` HEAD is at the R26/R27 wrap-up
  commit. Working tree should be clean after wrap-up.
- **3-way comparison (post-R27)**:

  | Config | Model × Agents | Speedup | FAIL_acc |
  |---|---|---|---|
  | R19 BEST | 7B-Coder × 5 | 1.29× | **60%** (accuracy-optimal) |
  | R26 | 3B-Instruct × 3 | **2.014×** | 27% (speed-optimal) |
  | R27 | 3B-Coder × 3 | 1.900× | 0% (avoid for critique) |

  See [results/SESSION_WRAP.md](./results/SESSION_WRAP.md) for full breakdown.

- **Speed bar: MET.** MULTI_SLOT copy (`SGLANG_CACHEBLEND_MULTI_SLOT=1`)
  breaks the 1-slot reuse ceiling: 97% utilization (5 slots ≈ 7100 tok),
  hitter p50 TTFT = 124 ms = **7.5× vs lossless (932 ms)**. The speed
  problem is solved by code-aware reuse alone.
- **Round 2 Selective Chunk Refresh (2026-07-02)** — implemented + measured
  (code in `python/sglang/srt/mem_cache/radix_cache.py`,
  `scheduler_output_processor_mixin.py`). At FRAC=0.6, code reuse drops
  659→111 (algorithm fires), but F1 stays at 0.503 (vs 0.508 baseline,
  Δ < noise) and TTFT regresses 975→1037 ms. **Negative result**: F1
  is NOT chunk-size dependent; cross-context KV loss is dominated by
  prompt prefix mismatch (canonical preamble vs agent role + task),
  not by copy volume. Full report `results/lossy_alg_round2/REPORT.md`.
- **Round 3 Direction A — extended canonical preamble (2026-07-02)** —
  **POSITIVE result**. Three iterations progressively include more
  agent-prompt structure in the precompute preamble:
  - v1: preamble + system + instruction → F1 0.508 → **0.541** (+6.5%)
  - v2: v1 + role/case placeholders → F1 → **0.580** (+14.2% cum)
  - v3: v2 + upstream context placeholder → F1 → **0.604** (+18.9% cum)

  Each step shrinks the prefix gap by ~30 tokens and yields
  +0.024-0.039 F1 improvement. **TTFT stays ~1.0× lossless** (read-path
  overhead cancels 694-tok reuse savings — Phase 6 finding, confirmed).
  Full report `results/lossy_alg_round3/REPORT.md`.

  **Root cause confirmed**: cross-context KV loss is prompt-prefix
  driven, NOT chunk-size/volume dependent. Direction A directly attacks
  this at the source by aligning precompute extraction context with the
  agent prompt's fixed structure elements.

  Remaining gap: role/case/upstream VALUES still vary (~80 tokens),
  need per-role (5× cost) or per-case extraction to close further.
  Speed remains ~1.0× lossless precompute — fundamental read-path
  overhead limit. Combining Direction A preamble with MULTI_SLOT-style
  file-level chunks could give both speed and F1.
- **Round 4 coarse file-level chunks + Direction A preamble (2026-07-02)**
  — **FAILED** due to allocator.py:189 device-mismatch RuntimeError
  (pre-existing bug, exposed by coarse mode's larger chunks). Coarse
  pool: 25 chunks × 1352 tok avg = 33813 tok (vs AST 72×224 = 16109).
  All configs (host_size 2/3/4 GB, max-tasks 1/5) crashed with same
  error. Not pursuing coarse path further.
- **Round 5 Direction A v3 + Selective Refresh FRAC=0.4 (2026-07-02)**
  — **F1 unchanged** (still 0.604 vs A v3 baseline). Code reuse drops
  694→468 (-33%) but F1 doesn't improve. Confirms F1 plateau is real
  and NOT chunk-selection addressable.
- **Final state (2026-07-02)**: Speed bar ❌ (~1.0×, no real speedup);
  Accuracy bar △ (F1 0.604 vs lossless 1.0, +18.9% from baseline).
  The ONLY known path to BOTH bars is True CacheBlend (attention
  recompute) — multi-week project.
- **Round 6 Direction A v4 per-role extraction (2026-07-02)** — **no
  improvement**. Per-role pool (extract with actual `implementer` value
  replacing ROLE placeholder): F1=0.601 (vs Direction A v3 0.604,
  Δ<noise). Per-role F1 varies wildly (implementer 0.457, others ~0.71)
  for unclear reasons — likely noise + agent 1 source effects.
  Per-role extraction is NOT worth the 5× cost.

**6 rounds summary**: Direction A v3 (Direction A preamble + AST chunks) is
the best — F1=0.604 (+18.9% from baseline), TTFT 933ms (~1.02× lossless,
not a real speedup). Both bars remain unmet by a fundamental limit:
raw-copy+RoPE under different prefix is inherently lossy, and precompute
read-path overhead cancels reuse savings at 7B scale.

## 🎯 Algorithm history (collapsed — superseded)

R10 / R17 / R18 were 18-round Pareto sweeps that landed on R10 (1.86×/F1=0.515)
and R17 (1.87×/F1=0.549) as interim optima. That regime was **retired 2026-07-03
under the verdict-task reframe** (memory `r21-verdict-accuracy`). The current
algorithmic ceilings are:

| Config | speedup | F1 / accuracy | Use for | Source |
|---|---|---|---|---|
| **R19 BEST** (7B-Coder × 5) | **1.29×** | **80% accuracy agreement, 8% garbage** | Accuracy-first | `results/lossy_alg_round21/FINAL_REPORT.md` |
| R26 (3B-Instruct × 3) | **2.014×** | 27% FAIL_acc | Speed-first | `results/lossy_alg_round26/COMPARISON.md` |
| R27 (3B-Coder × 3) | 1.900× | **0% FAIL_acc** | Avoid (Coder-biased) | `results/lossy_alg_round27/COMPARISON.md` |
| R33-R37 (SWE-bench fix-mode) | — | (no FAIL_TO_PASS verdict; R35 blocked) | Code-task A/B | `results/R34_R37_SUMMARY.md` |

Full 3-way verdict-task comparison table: `results/SESSION_WRAP.md`.
Full R1-R17 algorithm timeline + fundamental limit: `results/CODE_AWARE_LOSSY_KV_PROGRESS.md`.
- **Accuracy bar: NOT MET.** Cross-context KV loss is fundamental to raw-copy
  + RoPE: ~1400 tok → F1 0.46; ~7100 tok → F1 0.000.
- **Precompute pipeline (Phases 1–3, 4B/4C) — implemented + measured**
  (commit `628aeab83`). Offline AST-aware KV precompute → CPU host pool →
  layered async CPU→GPU reuse at task time. End-to-end A/B:
  reuse 867 tok/row, **real F1=0.374 (LOSSY), no speedup** (TTFT
  ~923 ms ≈ lossless 948 ms). Honest verdict: **both bars still unmet**;
  precompute validated as a building block, but raw-copy+RoPE cannot
  deliver speed + accuracy simultaneously.
- **Phase 6 device-resident diagnostic** — direct proof that H2D transfer is
  not the bottleneck (device-resident 960 ms > lossless 948 ms > host-pool
  SYNC 923 ms).
- **Phase 7 LAYERED F1=0.508 anomaly** — investigated + **decision not to
  pursue**. 7-way A/B showed three "fence differently on SYNC" hypotheses
  all FALSIFIED; the gap is a load_stream + LayerDoneCounter side-effect,
  not a transferable correctness improvement. Full report
  `results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`.
- **Retracted claims** (do not cite): L4 "~1.49× production-ready"
  (broken over-copying path); AST-gated L3 "1.448× both bars met"
  (cached_tokens conflated radix prefix + code-aware reuse); LAYERED
  F1=0.508 as a "speed-neutral correctness improvement" (not transferable,
  not above the 1.000 lossless bar).

## Branch state (snapshot 2026-07-07)

| Item | Value |
|---|---|
| HEAD branch | `fix/placeholder-pool-activation` |
| HEAD commit | `59030ca46` (R34-R37 results) |
| Last cycle | R26/R27 verdict-task 3B × 3 speedup (commit `7cc95c21e`) + R33 SWE-bench fix-mode (commit `cf713ba52`) + R34-R37 multi-instance SWE-bench fix-mode + harness parser improvements (commits `d61e6bd1e`, `59030ca46`) |
| Files added (this cycle) | `results/SESSION_WRAP.md`, `results/lossy_alg_round{26,27}/COMPARISON.md` + launchers, `results/swe_generated_patch_kvcomm_r{33,34,35}/`, `results/CODE_AWARE_LOSSY_KV_PROGRESS_R26_R27.html`, `results/ACC_AUDIT_R19_R26_R27.html`, `results/R33_FIX_MODE_REPORT.md`, `results/R34_R37_SUMMARY.md`, `results/HARNESS_CHANGE_NOTES_20260707.md` |
| Files modified (this cycle) | `benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py` (R36 + R37 helpers, default-backward-compatible), `.gitignore` |
| Known limitations | Precompute pipeline ships **default OFF** (gated by `SGLANG_PRECOMPUTE_KV_DIR`). R36 defensive truncation cannot repair already-truncated hunks (model-side truncation is structural). R35 FAIL_TO_PASS verification BLOCKED on env rebuild + network. `--disable-overlap-schedule --max-running-requests 1` required for >3 cases (`_delete_leaf` race). |
| Outstanding work | (P0) True CacheBlend (only path to both bars under format-strict verdict task); (P1) R34-R37 follow-up: rebuild astropy conda env + add R36 `stop_at_last_complete_hunk` heuristic; (P2) R26/R27 next directions (A-F in `results/lossy_alg_round24/DIRECTIONS_MEMO.md`); (P3) cleanup of `giant_codebase/` 88 stale `exact_*` rows.json (R1-R17 evidence, all superseded) |

---

## 1. What this project is

Coding-MAS serving, fast and correct via **code-aware KV cache reuse**.
Fork of SGLang adding a layered lossy reuse path on top of `RadixCache`:

1. **L2** — whole-slot byte-exact reuse + RoPE (cross-position).
2. **L3** — placeholder MiniLM k-NN body — *deprecated* (silent failure).
3. **L4** — AST-boundary chunk reuse (byte-exact per function/class).
4. **C2 / MULTI_SLOT** — CacheBlend gap-prefill + multi-slot batched copy.
5. **PRECOMPUTE** — offline AST-aware KV precompute → CPU host pool → async
   CPU→GPU reuse at task time. *Implemented + A/B-validated 2026-07-02;
   raw-copy+RoPE still lossy; validated as a building block for the next
   (true CacheBlend) iteration.*

Paper context: AgentTemplateKV submission to EuroSys 2026.

## 2. Current results (7B-Coder, full-share position-shift, fair A/B)

### 2a. Multi-slot ceiling test

| config | reuse | p50 TTFT | speedup vs lossless | F1 vs lossless |
|---|---|---|---|---|
| lossless (reference) | 0 | 932 ms | 1.0× | 1.000 |
| single-slot staged (L2, 1 slot) | ~1300 tok | ~820 ms | ~1.14× | 0.461 |
| **MULTI_SLOT (5 slots)** | **~7100 tok** | **124 ms** | **7.5×** | **0.000** |

### 2b. Precompute end-to-end A/B (5 case × 5 agent, 32k tokens)

| config | reuse | p50 TTFT | F1 vs lossless | stream |
|---|---|---|---|---|
| lossless ref | 0 tok | 948 ms | 1.000 | n/a |
| precompute SYNC (host pool) | 867 tok | 923 ms | 0.374 | default |
| precompute ASYNC (event-wait) | 867 tok | 930 ms | 0.374 | default |
| precompute LAYERED (host pool) | 867 tok | 918 ms | **0.508** | `load_stream` |
| precompute DEVICE-RESIDENT | 826 tok | 960 ms | 0.447 | default |
| (Phase 7) SYNC + DOUBLE SYNC | — | — | 0.375 | default |
| (Phase 7) SYNC + per-layer wait | — | — | 0.375 | default |
| (Phase 7) SYNC + record_stream | — | — | 0.375 | default |
| (Phase 7) SYNC a1 (agent_count=1) | — | — | 0.369 | default |
| (Phase 7) LAYERED a1 | — | — | 0.559 | `load_stream` |

**Speed verdict**: zero speedup (TTFT ≈ lossless for all four residency modes).
**Phase 6 verdict**: device-resident (zero H2D) is also slower than lossless —
the bottleneck is **not** the CPU→GPU transfer, it's the read-path itself
(move_kv_cache + RoPE + alloc overhead cancels the ~867-tok reuse gain at
this scale).
**Accuracy verdict**: F1 ≤ 0.508 across all paths (still lossy). LAYERED's
0.508 is documented as a real but mechanism-specific side-effect, NOT a
transferable correctness improvement (Phase 7).

### 2c. Partial-share + rotation (AST's niche, 7B-Coder, 12-task/42-case)

| metric | L2 whole-slot | L4 AST+fill | verdict |
|---|---|---|---|
| p50 TTFT | 927 ms | 966 ms | L4/L2 = **0.96× (AST slower)** |
| avg codeaware reused | 1399 | 516 | AST reuses less |
| F1 vs lossless | 0.513 | **0.622** | **AST more accurate** |

Full tables (3B/7B, full-share/partial-share) and the development timeline
are in [results/CODE_AWARE_LOSSY_KV_PROGRESS.md](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md).

## 3. The proven fundamental limit

Non-prefix KV reuse via raw copy + RoPE is **lossy** because KV at
layers > 0 encodes the preceding prefix. Reusing segments under a new
prefix copies stale KV. Confirmed with data: 1400 tok reused → F1 0.46;
7100 tok reused → F1 0.00. Precompute pipeline does not change this
(F1=0.374 for SYNC path). Only **true CacheBlend** (recompute attention
for copied chunks under the new context) can give speed AND accuracy.

## 4. Non-negotiable invariants

- **L3 (MiniLM k-NN body) is OFF by default** (`SGLANG_PLACEHOLDER_KNN_MATCH=0`).
  Do not re-enable. (Memory: `l3-placeholder-knn-deprecated`.)
- **Byte-exact match is the reuse gate.** No drift tolerance / MiniLM
  fallback at the reuse layer. AST anchors decide alignment, not matching.
- **Speedup ONLY from more reuse.** No KV-cache scheduling for speed.
- **New features ship OFF by default** (env var / CLI flag to enable).
- **For benchmark runs > 3 cases, add**
  `--disable-overlap-schedule --max-running-requests 1`
  (`_delete_leaf` assertion crash). `--force-evict` is NOT a real server flag.
  (Memory: `_delete-leaf-bug-2026-06-24`.)
- **Do NOT run `--vary-code`** for repeatable benchmarks. Use `--no-vary-code`.
- **Do NOT re-track `swebench_local_envs/` (21G).**
- **Do NOT track `results/codebase_kv/` (1.2 GB / run)** — gitignored.
- **F1 measurement: read `outputs.jsonl`'s `output_text` field.** Do NOT
  trust `rows.csv`'s `output` / `output_token_f1_vs_baseline` columns —
  they are placeholders / can be empty. Initial precompute A/B
  erroneously reported F1=1.000 because of this (now corrected).
  (Memory: `precompute-kv-ab-2026-07-02`.)

## 5. Outstanding work

| P | Task | Why | Gate |
|---|---|---|---|
| **P0** | True CacheBlend (attention recompute for copied chunks) | The only path to BOTH speed and accuracy; raw-copy+RoPE is proven lossy | User sign-off (fresh algorithmic change); precompute is its prerequisite |
| **P1** | Precompute async overlap via SGLang HiCache `LayerDoneCounter` mechanism | The concrete next speed lever; reuses the same per-layer-event machinery SGLang uses for radix HiCache (see plan Phase 4A-REVISED) | User sign-off; ~30 GPU-min |
| P2 | Cleanup commit (10 stale doc deletions + untracked raw result dirs) | Working tree noise | User confirms shape |
| P3 | Partial-share niche productization (AST > L2 accuracy bar) | Already MET accuracy bar at 0.96× speed | None |
| P4 | Re-run partial-share with more tasks for robust AST-vs-L2 average | 12-task/42-case result is noisy | None |

## 6. Key reference docs

| Doc | Purpose |
|---|---|
| [CANONICAL_TARGET.md](./CANONICAL_TARGET.md) | Single source of truth: goal, current state, invariants |
| [NEXT_SESSION_PROMPT.md](./NEXT_SESSION_PROMPT.md) | Paste-into-new-session prompt |
| [results/SESSION_WRAP.md](./results/SESSION_WRAP.md) | R19/R26/R27 verdict-task 3-way comparison (2026-07-06, canonical for current algorithmic ceiling) |
| [results/CODE_AWARE_LOSSY_KV_PROGRESS.md](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md) | Master timeline R1-R17 + fundamental limit |
| [results/CODE_AWARE_LOSSY_KV_PROGRESS.html](./results/CODE_AWARE_LOSSY_KV_PROGRESS.html) | 21-slide visual deck (R1-R37 comprehensive) |
| [results/R34_R37_SUMMARY.md](./results/R34_R37_SUMMARY.md) | R33-R37 SWE-bench fix-mode evidence |
| [results/HARNESS_CHANGE_NOTES_20260707.md](./results/HARNESS_CHANGE_NOTES_20260707.md) | R36 + R37 defensive parser + first-hunk-vs-gold helper |
| [results/kvcomm_ab/CROSS_POSITION_REPORT.md](./results/kvcomm_ab/CROSS_POSITION_REPORT.md) | Cross-position fix + 7B + partial-share results |
| [results/kvcomm_ab/KV_BREAKDOWN_REPORT.html](./results/kvcomm_ab/KV_BREAKDOWN_REPORT.html) | Visual KV-breakdown + multi-slot results |
| [results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md](./results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md) | Phase 7 LAYERED F1 investigation |
| [results/kvcomm_ab/precompute_ab_report/COMPARISON.txt](./results/kvcomm_ab/precompute_ab_report/COMPARISON.txt) | 7-way precompute A/B table |
| [results/direction_3_phase_c_d_20260627.html](./results/direction_3_phase_c_d_20260627.html) | L4 Phase C/D architecture deep-dive (still valid) |

## 7. Common commands

```bash
# L4 chunker + pool unit tests
python -m pytest test/registered/unit/mem_cache/test_ast_chunker.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool_read.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool_policy.py \
                   test/registered/unit/mem_cache/test_codebase_kv_loader.py -v

# Precompute pipeline A/B (5 case × 5 agent, 7B-Coder, position-shift, canonical-prefix)
bash results/kvcomm_ab/run_7b_precompute_ab.sh   # LAYERED (default)
# SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=0 / SGLANG_PRECOMPUTE_DEVICE_RESIDENT=1 / etc.
# Eight variant launchers in results/kvcomm_ab/run_7b_precompute_ab_*.sh

# Precompute KV extraction (one-time, populates results/codebase_kv/pandas_5case/)
SGLANG_PRECOMPUTE_CODEBASE_KV=1 \
SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case \
SGLANG_PRECOMPUTE_MAX_FILES=25 \
  python scripts/precompute_codebase_kv.py \
    --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
    --repo-root results/giant_codebase/pandas_src \
    --model Qwen/Qwen2.5-Coder-7B-Instruct

# Fair A/B analysis (decomposed counters, source exclusion, F1 from outputs.jsonl)
python benchmark/multi_workflow/analyze_fair_ab.py \
    --baseline results/kvcomm_ab/7b_precompute_ab_lossless \
    --experimental results/kvcomm_ab/7b_precompute_ab_layered \
    --lossless results/kvcomm_ab/7b_precompute_ab_lossless
```

Key env toggles (all default OFF unless noted):
- `SGLANG_CACHEBLEND_MULTI_SLOT=1` + `SGLANG_CACHEBLEND_COMPACT=0` — multi-slot copy
- `SGLANG_CACHEBLEND_CHUNK=1` + `SGLANG_CACHEBLEND_BATCH=1` — C2 batched executor
- `SGLANG_CHUNKED_PLACEHOLDER_KNN=1` + `SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1` — L4 read/write
- `SGLANG_CHUNK_COARSE=1` — L2 whole-slot; `SGLANG_CHUNK_TOPLEVEL=1 SGLANG_CHUNK_FILL_GAPS=1` — L4 AST
- `SGLANG_PLACEHOLDER_KNN_MATCH=0` — L3 OFF (default, keep off)
- `SGLANG_PRECOMPUTE_KV_DIR=…` — precompute on (also need `SGLANG_CHUNKED_PLACEHOLDER_KNN=1`)
- `SGLANG_PRECOMPUTE_HOST_LOAD_ASYNC=1` — layered async CPU→GPU (LAYERED path)
- `SGLANG_PRECOMPUTE_CANONICAL_PREFIX=1` — canonical preamble (accuracy lever)
- `SGLANG_PRECOMPUTE_DEVICE_RESIDENT=1` — diagnostic: load KV straight to GPU
- `SGLANG_KVFLOW_{DOUBLE_SYNC,PERLAYERWAIT,RECORDSTREAM_SYNC,BYTECMP_DUMP}=1` — Phase 7 hooks (all default OFF)
- `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` — warn-only (multi-slot leak workaround)

## 8. What NOT to do

1. **Don't re-enable L3 (MiniLM k-NN body).** Deprecated; 8.2% silent failure.
2. **Don't propose drift tolerance / MiniLM fallback at the reuse layer.** Byte-exact only.
3. **Don't run `--vary-code`** for measurement.
4. **Don't run > 3 cases without** `--disable-overlap-schedule --max-running-requests 1`.
5. **Don't re-track `swebench_local_envs/` (21G) or `results/codebase_kv/` (1.2 GB/run).**
6. **Don't cite the retracted claims** ("~1.49× production-ready", "1.448× both bars met", "LAYERED F1=0.508 as correctness fix").
7. **Don't trust `rows.csv`'s output / output_token_f1 columns.** Read `outputs.jsonl`.
8. **Don't cite 2.10× giant-codebase fair A/B** as "fast + accurate" — its F1 column is a default 1.0 placeholder (no in-run baseline), not real accuracy.

## 9. Memory pointers (auto-load each session)

- `multi-slot-copy-2026-07-01` — MULTI_SLOT: 7.5× speed, F1=0.000 (latest speed ceiling)
- `precompute-kv-ab-2026-07-02` — precompute end-to-end A/B + Phase 7 verdict (most recent cycle)
- `r25-oracle-8pct-unk-2026-07-06` — R25 oracle model: 0% garbage, 80% accuracy agreement (R19 ceiling)
- `r26-r27-3b-speedup-2026-07-06` — 3B × 3 = ~2× speedup, counterintuitive Coder ≠ critique accuracy
- `cross-position-fix-works-2026-06-30` — cross-position slot_id fix unblocked byte-exact reuse
- `c2-cacheblend-lossy-not-safe-2026-06-28` — raw-copy+RoPE is lossy (the fundamental limit)
- `c2-fundamental-limits-2026-06-28` — proven limits + vary-code speed bar history
- `fair-measurement-prefix-conflation-2026-06-30` — why 1.448× was retracted
- `l3-placeholder-knn-deprecated` — why L3 is off
- `_delete-leaf-bug-2026-06-24` — the >3-case assertion crash & workaround
- `giant-codebase-benchmark-swesmith` — 50-task × 5-agent benchmark
- `output-path` — results go to `results/`, not `/tmp`

---

**Last refreshed**: 2026-07-07, after R26/R27 verdict-task cycle + R33-R37
SWE-bench fix-mode cycle. Next refresh trigger: R34-R37 follow-up (env
rebuild + R36 `stop_at_last_complete_hunk`), True CacheBlend kernel work,
or a fresh fair multi-case headline number.