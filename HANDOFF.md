# HANDOFF — sglang-kvflow (2026-07-01)

> **READ FIRST**:
> 1. [CANONICAL_TARGET.md](./CANONICAL_TARGET.md) — single project goal + current state.
> 2. [results/CODE_AWARE_LOSSY_KV_PROGRESS.md](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md)
>    — full development timeline, results tables, proven fundamental limit.
> 3. [results/kvcomm_ab/KV_BREAKDOWN_REPORT.html](./results/kvcomm_ab/KV_BREAKDOWN_REPORT.html)
>    — visual KV-breakdown + multi-slot results.
>
> The auto-loaded [memory index](../home/gfy/.claude/projects/-home-gfy/memory/MEMORY.md)
> has the key invariants.

---

## TL;DR

- **Branch**: `fix/placeholder-pool-activation` (uncommitted changes on top
  of the multi-slot work).
- **Speed bar: MET.** MULTI_SLOT copy (`SGLANG_CACHEBLEND_MULTI_SLOT=1`)
  breaks the 1-slot reuse ceiling: 97% utilization (5 slots ≈ 7100 tok),
  hitter p50 TTFT = 124 ms = **7.5× vs lossless (932 ms)**. The speed
  problem is solved by code-aware reuse alone.
- **Accuracy bar: NOT MET for substantial reuse.** MULTI_SLOT hitters
  output garbage (F1=0.000). Single-slot (≈1400 tok) F1≈0.46
  (valid-but-different). Root cause is proven: **cross-context KV loss**
  — raw copy + RoPE of KV under a different prefix is lossy, and the loss
  scales with reuse volume. See `c2-cacheblend-lossy-not-safe`,
  `multi-slot-copy-2026-07-01`.
- **Only remaining path**: true CacheBlend (attention recompute for
  copied chunks under the new context). Expensive, **not yet built**,
  awaits explicit user sign-off.
- **Retracted claims** (do not cite): L4 "~1.49× production-ready"
  (broken over-copying path); AST-gated L3 "1.448× both bars met"
  (cached_tokens conflated radix prefix + code-aware reuse). See
  `fair-measurement-prefix-conflation-2026-06-30`.

## Branch state (snapshot 2026-07-01)

| Item | Value |
|---|---|
| HEAD branch | `fix/placeholder-pool-activation` |
| Latest landed work | MULTI_SLOT copy (stage leading gap + copy all 5 slots + zero inter-slot headers) |
| Goal status | Speed bar MET; accuracy bar NOT met for substantial reuse |
| Known limitation | MULTI_SLOT copied spans occasionally leak (not radix-evictable); mitigated with `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` |

---

## 1. What this project is

Coding-MAS serving, fast and correct via **code-aware KV cache reuse**.
Fork of SGLang adding a layered lossy reuse path on top of `RadixCache`:

1. **L2** — whole-slot byte-exact reuse + RoPE (cross-position).
2. **L3** — placeholder MiniLM k-NN body — *deprecated* (silent failure).
3. **L4** — AST-boundary chunk reuse (byte-exact per function/class).
4. **C2 / MULTI_SLOT** — CacheBlend gap-prefill + multi-slot batched copy.

Paper context: AgentTemplateKV submission to EuroSys 2026.

## 2. The current result (7B-Coder, full-share position-shift, fair A/B)

| config | reuse | p50 TTFT | F1 vs lossless |
|---|---|---|---|
| lossless | 0 | 932 ms | 1.000 |
| single-slot staged (L2) | ~1300 tok | ~820 ms | 0.461 |
| **MULTI_SLOT (5 slots)** | ~7100 tok | **124 ms (7.5×)** | **0.000** |

Full tables (3B/7B, full-share/partial-share) and the development
timeline are in
[results/CODE_AWARE_LOSSY_KV_PROGRESS.md](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md).

## 3. The proven fundamental limit

Non-prefix KV reuse via raw copy + RoPE is **lossy** because KV at
layers > 0 encodes the preceding prefix. Reusing segments under a new
prefix copies stale KV. Confirmed with data: 1400 tok reused → F1 0.46;
7100 tok reused → F1 0.00. Only **true CacheBlend** (recompute attention
for copied chunks under the new context) can give speed AND accuracy.

## 4. Non-negotiable invariants

- **L3 (MiniLM k-NN body) is OFF by default** (`SGLANG_PLACEHOLDER_KNN_MATCH=0`).
  Do not re-enable. (Memory: `l3-placeholder-knn-deprecated`.)
- **Byte-exact match is the reuse gate.** No drift tolerance / MiniLM
  fallback at the reuse layer. AST anchors decide alignment, not matching.
- **Speedup ONLY from more reuse.** No KV-cache scheduling for speed.
- **New features ship OFF by default.**
- **For benchmark runs > 3 cases, add**
  `--force-evict --disable-overlap-schedule --max-running-requests 1`
  (`_delete_leaf` assertion crash). (Memory: `_delete-leaf-bug-2026-06-24`.)
- **Do NOT run `--vary-code`** for repeatable benchmarks. Use `--no-vary-code`.
- **Do NOT re-track `swebench_local_envs/` (21G).**

## 5. Outstanding work

| P | Task | Why | Gate |
|---|---|---|---|
| **P0** | True CacheBlend (attention recompute for copied chunks) | The only path to BOTH speed and accuracy; raw-copy+RoPE is proven lossy | User sign-off (fresh algorithmic change) |
| P1 | Fix MULTI_SLOT copied-span leak (ephemeral copy / proper radix insertion) | Leak forces `STRICT_MEM_CHECK_DURING_IDLE=0`; bounded but unclean | None, but optional if CacheBlend supersedes multi-slot |
| P2 | Re-run partial-share with more tasks for a robust AST-vs-L2 average | 12-task/42-case result is noisy; AST meets accuracy bar not speed | None |

## 6. Key reference docs

| Doc | Purpose |
|---|---|
| [CANONICAL_TARGET.md](./CANONICAL_TARGET.md) | Single source of truth: goal, current state, invariants |
| [results/CODE_AWARE_LOSSY_KV_PROGRESS.md](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md) | Master timeline + results + fundamental limit |
| [results/kvcomm_ab/CROSS_POSITION_REPORT.md](./results/kvcomm_ab/CROSS_POSITION_REPORT.md) | Cross-position fix + 7B + partial-share results |
| [results/kvcomm_ab/KV_BREAKDOWN_REPORT.html](./results/kvcomm_ab/KV_BREAKDOWN_REPORT.html) | Visual KV-breakdown + multi-slot results |
| [results/direction_3_phase_c_d_20260627.html](./results/direction_3_phase_c_d_20260627.html) | L4 Phase C/D architecture deep-dive (still valid) |
| ⚠️ [results/project_progress_20260627.html](./results/project_progress_20260627.html) | Has a RETRACTION banner — speedup claims superseded |

## 7. Common commands

```bash
# L4 chunker + pool unit tests
python -m pytest test/registered/unit/mem_cache/test_ast_chunker.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool_read.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool_policy.py -v

# MULTI_SLOT A/B (full-share position-shift, 7B, 12 tasks)
bash results/kvcomm_ab/run_7b_multislot_l2.sh
# lossless reference
bash results/kvcomm_ab/run_7b_lossless.sh
# fair A/B analysis
python benchmark/multi_workflow/analyze_fair_ab.py \
    --baseline results/kvcomm_ab/7b_lossless \
    --experimental results/kvcomm_ab/7b_multislot_l2 \
    --lossless results/kvcomm_ab/7b_lossless
```

Key env toggles (all default OFF unless noted):
- `SGLANG_CACHEBLEND_MULTI_SLOT=1` + `SGLANG_CACHEBLEND_COMPACT=0` — multi-slot copy
- `SGLANG_CACHEBLEND_CHUNK=1` + `SGLANG_CACHEBLEND_BATCH=1` — C2 batched executor
- `SGLANG_CHUNKED_PLACEHOLDER_KNN=1` + `SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1` — L4 read/write
- `SGLANG_CHUNK_COARSE=1` — L2 whole-slot; `SGLANG_CHUNK_TOPLEVEL=1 SGLANG_CHUNK_FILL_GAPS=1` — L4 AST
- `SGLANG_PLACEHOLDER_KNN_MATCH=0` — L3 OFF (default, keep off)
- `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` — warn-only (multi-slot leak workaround)

## 8. What NOT to do

1. **Don't re-enable L3 (MiniLM k-NN body).** Deprecated; 8.2% silent failure.
2. **Don't propose drift tolerance / MiniLM fallback at the reuse layer.** Byte-exact only.
3. **Don't run `--vary-code`** for measurement.
4. **Don't run > 3 cases without** `--force-evict --disable-overlap-schedule --max-running-requests 1`.
5. **Don't re-track `swebench_local_envs/` (21G).**
6. **Don't cite the retracted claims** ("~1.49× production-ready", "1.448× both bars met").

## 9. Memory pointers (auto-load each session)

- `multi-slot-copy-2026-07-01` — MULTI_SLOT: 7.5× speed, F1=0.000 (latest)
- `cross-position-fix-works-2026-06-30` — cross-position slot_id fix unblocked byte-exact reuse
- `c2-cacheblend-lossy-not-safe-2026-06-28` — raw-copy+RoPE is lossy (the fundamental limit)
- `c2-fundamental-limits-2026-06-28` — proven limits + vary-code speed bar history
- `fair-measurement-prefix-conflation-2026-06-30` — why 1.448× was retracted
- `l3-placeholder-knn-deprecated` — why L3 is off
- `_delete-leaf-bug-2026-06-24` — the >3-case assertion crash & workaround
- `giant-codebase-benchmark-swesmith` — 50-task × 5-agent benchmark
- `output-path` — results go to `results/`, not `/tmp`

---

**Last refreshed**: 2026-07-01, after MULTI_SLOT measurement. Next refresh
trigger: true CacheBlend decision, or a fresh fair multi-case headline number.
