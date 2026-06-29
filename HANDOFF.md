# HANDOFF — sglang-kvflow

> ⚠️ **READ FIRST**:
> 1. [CANONICAL_TARGET.md](./CANONICAL_TARGET.md) — the *single* project goal
>    statement and 4-layer architecture.
> 2. [results/project_progress_20260627.html](./results/project_progress_20260627.html)
>    — comprehensive progress report (this file's visual companion).
> 3. The auto-loaded [memory index](../home/gfy/.claude/projects/-home-gfy/memory/MEMORY.md)
>    has the key invariants (L3 deprecated, byte-exact only, etc.).
>
> **If you have 5 minutes**: read CANONICAL_TARGET.md + this file's TL;DR +
> Outstanding work.
> **If you have 30 minutes**: read everything below in order.

---

## TL;DR

- **Branch**: `fix/placeholder-pool-activation` (33 commits ahead of `main`, clean working tree).
- **Goal (2026-06-29)**: code-aware KV reuse (AST-Gated L3 + offset alignment)
  with **good-enough TTFT speedup AND accuracy ≥ general L3** under the same prompts.
  - **BOTH bars MET in BOTH scenarios** (commit `4c1f77fa8`).
  - Vary-code: 1.448× ≥ L3 1.441× (speed MET), F1 0.240 > L3 0.193 (accuracy strictly better).
  - Same-code: 1.243× = L3 matched baseline, F1 0.402 = L3 (offset gate does not fire → no regression).
  - The vary-code speed bar (previously the only unmet condition) is closed by the
    offset-aligned AST gate (`SGLANG_L3_AST_GATE_OFFSET=1`): makes the fast L3 whole-slot
    copy fire under vary-code instead of rejecting → slow C2.
- **Tests**: 28/28 in the new Direction-#3 test files; ~85/85 across the full mem_cache suite.
- **Status**: offset-aligned AST gate landed (`4c1f77fa8`). **Visual summary**: `results/contribution_summary_20260629.html`.

## Branch state (snapshot 2026-06-29)

| Item | Value |
|---|---|
| HEAD branch | `fix/placeholder-pool-activation` |
| Ahead of `main` | 34 commits |
| Latest commit | `4c1f77fa8` feat: offset-aligned AST gate — close vary-code speed bar |
| Goal status | **MET** — both bars (speedup + accuracy ≥ general L3) met in both scenarios |
| Latest giant-codebase runs | `diag_{vary_l3off,vary_l3base(combo2),novary_l3off,novary_l3base,combo2_*}` |
| Superseded claim | L4 "~1.49× production-ready" (2026-06-27) — falsified by flat-prefix ceiling |

---

## 1. What this project is

**Coding-MAS serving**, fast and correct via code-aware KV cache reuse. This is
a fork of SGLang at `sglang-kvflow` adding:

1. **L2 whole-slot byte-exact reuse** — fuzzy match on token-id hashes
2. **L3 placeholder MiniLM k-NN body** — *deprecated* (8.2% silent failure on real
   workload)
3. **L4 AST chunk pool** — byte-exact per-function/class chunk reuse (Direction #3)

Paper context: AgentTemplateKV submission to EuroSys 2026 (branch
`agenttemplatekv-eurosys-2026-06`); HEAD numbers **+21.3% TTFT, +1.9% E2E, +64%
cached** vs SGLang baseline. Full 38-pp breakdown is in
`docs/agenttemplatekv_paper/` and is intentionally not duplicated here.

## 2. The 4-layer cache

See [CANONICAL_TARGET.md](./CANONICAL_TARGET.md) §"The 4-Layer Cache Architecture"
for the authoritative table. Quick reference:

| Layer | Mechanism | Cumulative | Status |
|---|---|---|---|
| L1 | Prefix cache (radix) | 1.20× | production |
| L2 | Whole-slot byte-exact | 1.31× | production |
| L3 | MiniLM k-NN body | 1.65× | **DEPRECATED** (8.2% silent fail) |
| L4 | AST chunk pool | ~1.49× | **CURRENT TARGET** |

## 3. Direction #3 — AST chunk pool

Four phases, one sprint (Jun 26 – Jun 27). All on `fix/placeholder-pool-activation`.

| Phase | Commit | Code location | Tests |
|---|---|---|---|
| A — AST chunker | `7fb1a5bb2` | `ast_chunker.py` (~220 LOC, pure-Python AST) | 13 |
| B — write path | `8599afcfc` | `_store_placeholder_chunks` at `radix_cache.py:1654` | 8 |
| C — read path | `5197823bf` | `_try_placeholder_chunk_lossy_match` at `:1796`, `_build_chunk_plan` at `:1868`, `_execute_chunk_plan` at `:2017` | 9 |
| D — telemetry | `fea64d4cc` | 11 counters at `radix_cache.py:698-710` | 3 |

For the architectural diagram, byte-exact decision flow, and per-counter
specification, see [results/project_progress_20260627.html](./results/project_progress_20260627.html).

## 4. Non-negotiable invariants

These are the rules a new session **must not** propose to violate.

- **L3 is deprecated for production.** Default `SGLANG_PLACEHOLDER_KNN_MATCH=0`.
  Do not propose re-enabling it. The 8.2% silent failure rate on the
  giant-codebase 50-task run is the dominant failure mode that triggered
  deprecation. (See memory: `l3-placeholder-knn-deprecated`.)
- **L4 byte-exact is binary.** L4 chunks are reused only when
  `(slot_id, signature, byte_start, byte_end, token_ids)` all match exactly.
  Do **not** propose drift tolerance, MiniLM fallback, or lossy matching at
  the chunk layer. L3's `cos ≥ 0.85` gate is what made L3 unsafe.
- **For benchmark runs > 3 cases, you MUST add**
  `--force-evict --disable-overlap-schedule --max-running-requests 1` to
  avoid the `_delete_leaf` assertion that crashes normal-evict. (See
  memory: `_delete-leaf-bug-2026-06-24`.)
- **Do NOT run `--vary-code`** for repeatable benchmarks. Use `--no-vary-code`.

## 5. Operational caveats

| Caveat | Effect | Workaround |
|---|---|---|
| `--vary-code` mutates source across runs | Unrepeatable benchmark numbers | Always use `--no-vary-code` for production measurement |
| `_delete_leaf` assertion under > 3 cases | Crashes normal-evict path | Add `--force-evict --disable-overlap-schedule --max-running-requests 1` |
| RoPE delta requires head-only rotation | Existing `_apply_rope_delta_to_head` handles this; do not rewrite | Reuse as-is |
| 21G `swebench_local_envs/` in results/ | Slow git operations if re-tracked | Already gitignored; do not re-add |
| `radix_cache.py` is 4895 lines | Hard to navigate | Use the line numbers in §3 / §6 |

## 6. Outstanding work

| P | Task | Why | Gate |
|---|---|---|---|
| **P0** | Run giant-codebase 5-task smoke with `SGLANG_CHUNKED_PLACEHOLDER_KNN=1 SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1` to validate the predicted ~1.49× speedup | Phase C/D is code-complete but unvalidated on real workload | None — ready to launch |
| **P1** | Merge `fix/placeholder-pool-activation` → `main` | 25 commits, clean, no conflicts | After P0 ≥ 1.4× |
| **P2** | Phase E — whitespace-drift tolerance gated on `placeholder_chunk_pool_skip_byte_drift_count` telemetry | Recover L3's lost ~0.16× if it's the dominant skip reason | Only if P0 borderline |

## 7. Key reference docs

| Doc | Purpose |
|---|---|
| [CANONICAL_TARGET.md](./CANONICAL_TARGET.md) | Single source of truth for project goal, 4-layer architecture, deprecation policy |
| [results/project_progress_20260627.html](./results/project_progress_20260627.html) | Comprehensive visual progress report |
| [results/direction_3_phase_c_d_20260627.html](./results/direction_3_phase_c_d_20260627.html) | Phase C/D deep-dive |
| [results/giant_codebase/runs/giant_pandas_50_l3_off_20260627_051353/report/REPORT.md](./results/giant_codebase/runs/giant_pandas_50_l3_off_20260627_051353/report/REPORT.md) | Production baseline (1.31×) |
| [results/giant_codebase/runs/giant_pandas_50_postfix_20260627_024916/report/REPORT.md](./results/giant_codebase/runs/giant_pandas_50_postfix_20260627_024916/report/REPORT.md) | L3-ON research run (1.65×, deprecated) |
| [results/ast_alignment_v3_20260626/REPORT.md](./results/ast_alignment_v3_20260626/REPORT.md) | 91.8% byte-identical AST alignment measurement |
| [memory/v44-cycle-history.md](../home/gfy/.claude/projects/-home-gfy/memory/v44-cycle-history.md) | v44 k-NN cycle historical (8 evidence lines, all deprecated 2026-06-27) |

## 8. Common commands

```bash
# Run Direction #3 unit tests (35 tests, ~60s)
python -m pytest test/registered/unit/mem_cache/test_ast_chunker.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool.py \
                   test/registered/unit/mem_cache/test_radix_cache_concurrency.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool_read.py \
                   test/registered/unit/mem_cache/test_placeholder_chunk_pool_policy.py -v

# Run full mem_cache suite
python -m pytest test/registered/unit/mem_cache/ -v

# Giant-codebase 5-task smoke (P0)
python benchmark/multi_workflow/bench_giant_codebase_reuse.py \
    --manifest  results/giant_codebase/tasks/pandas_50.jsonl \
    --repo-root results/giant_codebase/pandas_src \
    --out-dir   results/giant_codebase/runs/giant_pandas_5_l4_on_$(date -u +%Y%m%d_%H%M%S) \
    --model     Qwen/Qwen2.5-3B-Instruct \
    --max-tasks 5 --agent-count 5 \
    --mode      placeholder_knn_reuse --segment-count 5 \
    --no-vary-code
# NOTE: for > 3 tasks add --force-evict --disable-overlap-schedule --max-running-requests 1
```

Env-var toggles:

- `SGLANG_CHUNKED_PLACEHOLDER_KNN=1` — turn the chunk pool write-path on
- `SGLANG_CHUNKED_PLACEHOLDER_KNN_MATCH=1` — turn the chunk pool read-path on
- `SGLANG_PLACEHOLDER_KNN_MATCH=0` (default) — keep L3 off in production

## 9. What NOT to do

1. **Don't re-enable L3 (MiniLM k-NN body).** It is deprecated. 8.2% silent failure.
2. **Don't propose drift tolerance or MiniLM fallback at the L4 chunk layer.** L4 is
   byte-exact by design. Phase E drift tolerance, if pursued at all, is a separate
   opt-in flag and gated on telemetry.
3. **Don't run the giant-codebase bench without `--no-vary-code`.** Results
   become unrepeatable.
4. **Don't run > 3 cases without `--force-evict --disable-overlap-schedule
   --max-running-requests 1`.** You'll hit the `_delete_leaf` assertion crash.
5. **Don't re-track `swebench_local_envs/` (21G).** It's gitignored for a reason.

## 10. Memory pointers

These are the memory entries that auto-load in every session. New sessions
should skim them:

- `direction-3-phase-c-d` — Direction #3 read-path + telemetry (this sprint)
- `l3-placeholder-knn-deprecated` — why L3 is off
- `sglang-kvflow-session-handoff-2026-06-27` — session index pointer
- `100-case-force-evict-fix` — why `--force-evict` is required
- `_delete-leaf-bug-2026-06-24` — the assertion crash & its workaround
- `v44-cycle-history` — v44 k-NN cycle (8 evidence lines, all superseded 2026-06-27)
- `sglang-kvflow-placeholder-pool-bugs` — 3 critical activation bugs (Jun 26)
- `giant-codebase-benchmark-swesmith` — 50-task × 5-agent baseline run

---

**Last refreshed**: 2026-06-27, after Phase C/D land. Next refresh trigger: P0
giant-codebase smoke result, or merge to main.
