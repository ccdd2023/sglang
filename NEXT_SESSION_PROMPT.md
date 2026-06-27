# sglang-kvflow — New Session Handoff Prompt

> **Purpose**: paste this into a new Claude session to get up to speed
> in 5 minutes. The auto-loaded memory index gives you the key invariants;
> the three files in §1 give you the project state; §3 gives you the
> constraints you must not violate.

---

## 1. Read in this order

1. `/home/gfy/CodeMAS_Project/sglang-kvflow/CANONICAL_TARGET.md`
   — the single source of truth for project goal + 4-layer architecture
2. `/home/gfy/CodeMAS_Project/sglang-kvflow/HANDOFF.md`
   — polished end-to-end handoff (189 lines)
3. `/home/gfy/CodeMAS_Project/sglang-kvflow/results/project_progress_20260627.html`
   — comprehensive visual progress report (open in browser)

If you have only 60 seconds: read §"TL;DR" and §"Outstanding work" in
HANDOFF.md, then skim CANONICAL_TARGET.md §"THE SINGLE CURRENT GOAL".

## 2. Branch and state (2026-06-27)

- **HEAD branch**: `fix/placeholder-pool-activation` (25 commits ahead of
  `main`, clean working tree, latest commit `57881a45d`)
- **Goal**: validate Direction #3 (L4 AST chunk pool) end-to-end
- **Tests**: 35/35 new Direction-#3 tests pass; full mem_cache suite ~85/~85
- **Production baseline**: 1.31× (L1+L2, L3 OFF)
- **Current target**: ~1.49× (L1+L2+L4 chunk pool, pending P0 smoke)
- **Research (deprecated)**: 1.65× (L1+L2+L3, 8.2% silent failure rate)

## 3. Non-negotiable constraints

These are rules you must not propose to violate.

1. **L3 (MiniLM k-NN body) is deprecated for production.** Default
   `SGLANG_PLACEHOLDER_KNN_MATCH=0`. Do not propose re-enabling it. (See
   memory: `l3-placeholder-knn-deprecated`.)
2. **L4 byte-exact is binary.** L4 chunks are reused only when
   `(slot_id, signature, byte_start, byte_end, token_ids)` all match
   exactly. Do NOT propose drift tolerance, MiniLM fallback, or lossy
   matching at the chunk layer.
3. **For benchmark runs > 3 cases, you MUST add**
   `--force-evict --disable-overlap-schedule --max-running-requests 1` to
   avoid the `_delete_leaf` assertion crash. (See memory:
   `_delete-leaf-bug-2026-06-24`.)
4. **Do NOT run `--vary-code`** for repeatable benchmarks. Use
   `--no-vary-code`.
5. **Do NOT re-track `swebench_local_envs/` (21G)** in git. It's
   gitignored for a reason.

## 4. What to do next (prioritized)

| P | Task | Command |
|---|---|---|
| **P0** | Run giant-codebase 5-task smoke with `SGLANG_CHUNKED_PLACEHOLDER_KNN=1 SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1` to validate the predicted ~1.49× speedup | See `HANDOFF.md §8 Common commands` |
| **P1** | Once P0 confirms ≥1.4×, merge `fix/placeholder-pool-activation` → `main` (clean, no conflicts) | `git checkout main && git merge --no-ff fix/placeholder-pool-activation` |
| **P2** | (only if P0 is borderline) Phase E whitespace-drift tolerance, gated on `placeholder_chunk_pool_skip_byte_drift_count` telemetry | Open new design doc, requires user sign-off |

## 5. Key files and where to find things

- `python/sglang/srt/mem_cache/radix_cache.py` (4895 lines)
  - `:698-710` — 11 telemetry counters (5 Phase B + 6 Phase D)
  - `:1494` — `ASTChunker` import (Phase A)
  - `:1654` — `_store_placeholder_chunks` write path (Phase B)
  - `:1796` — `_try_placeholder_chunk_lossy_match` read entry (Phase C)
  - `:1868` — `_build_chunk_plan` (Phase C)
  - `:1990` — `_find_byte_exact_chunk_entry` (Phase C, strict byte-exact)
  - `:2017` — `_execute_chunk_plan` (Phase C, alloc + move_kv + rope_delta)
- `python/sglang/srt/mem_cache/ast_chunker.py` — server-side AST chunker
  (~220 LOC, Phase A)
- `benchmark/multi_workflow/bench_giant_codebase_reuse.py` (584 LOC) —
  main benchmark driver. `parse_args` at `:508`, `main` at `:561`.
- `test/registered/unit/mem_cache/` — 5 new test files:
  - `test_ast_chunker.py` (13 tests, Phase A)
  - `test_placeholder_chunk_pool.py` (8 tests, Phase B)
  - `test_radix_cache_concurrency.py` (2 tests, Phase B)
  - `test_placeholder_chunk_pool_read.py` (9 tests, Phase C)
  - `test_placeholder_chunk_pool_policy.py` (3 tests, Phase D)

## 6. Memory entries relevant to this project

These auto-load in every session:

- `direction-3-phase-c-d` — Direction #3 read-path + telemetry (this sprint)
- `l3-placeholder-knn-deprecated` — why L3 is off
- `sglang-kvflow-session-handoff-2026-06-27` — session index pointer
- `100-case-force-evict-fix` — why `--force-evict` is required
- `_delete-leaf-bug-2026-06-24` — assertion crash & workaround
- `v44-cycle-history` — v44 k-NN cycle (all superseded 2026-06-27)
- `sglang-kvflow-placeholder-pool-bugs` — 3 critical activation bugs (Jun 26)
- `giant-codebase-benchmark-swesmith` — 50-task × 5-agent baseline
- `output-path` — results go to `results/`, not `/tmp`
- `code-aware-kv-reuse-exact-text-match` — exact-text-match invariant

## 7. Common pitfalls

- `match_prefix` runs L3 (deprecated, off by default) **before** L4
  (chunk pool, on by opt-in). They are **independent** — L4 does not
  inherit L3's decisions.
- The `placeholder_chunk_pool` key is `(slot_id, chunk_signature)`. When
  debugging pool-miss, check **both** fields.
- The chunker's `_byte_to_token_offset` is pure-Python AST and
  deterministic. If you see drift, the source bytes changed, not the
  chunker.
- L4 doesn't introduce a fractional confidence. `ChunkDecision.confidence`
  is binary: 1.0 for byte-exact hit, 0.0 for dense fallback.
- The Phase D telemetry counter `placeholder_chunk_pool_skip_byte_drift_count`
  is what tells you if Phase E drift tolerance would be worth pursuing.
- The 4-layer cache speedups are **cumulative**, not additive. L1+L2+L4
  ~1.49× is `1.20 × 1.09 × 1.14`, not `1.20 + 0.11 + 0.18`.

## 8. When in doubt

- The 3 files in §1 cover 95% of context questions.
- The 10 memory entries in §6 cover 90% of the rest.
- Ask the user only for: changes to project direction, new architectural
  decisions, or out-of-scope research extensions. Everything else can be
  inferred from the docs above.

---

**Snapshot**: 2026-06-27, after Phase C/D land. Next refresh trigger:
P0 giant-codebase smoke result.
