# Giant-Codebase Re-Run — Post-Fix (2026-06-27)

**Branch**: `fix/placeholder-pool-activation`
**Setup**: 50 pandas tasks × 5 agents = 250 requests, single chunk
**Driver**: `bench_giant_codebase_reuse.py --no-vary-code --chunk-size 50`
**Model**: Qwen2.5-3B-Instruct
**Run dir**: `results/giant_codebase/runs/giant_pandas_50_postfix_20260627_024916/`

---

## TL;DR

The pre-fix Gate 2 measurement (50 pandas × 5 agents) showed only
**1.31× speedup** but `placeholder_anchor_store_entry_count = 0` on
every row — the placeholder pool never grew, so the speedup was
purely prefix-cache contribution. The root cause was the
`--vary-code` driver flag prepending 16 bytes (`# Agent {idx} variant\n`)
per agent at `bench_giant_codebase_reuse.py:302-310`, which corrupted
the placeholder-pool match against warmed planner entries.

After this re-run with `--no-vary-code` and `--chunk-size 50` (so the
pool survives the full 50-task span), the placeholder pool actually
fires and contributes the missing speedup.

| Metric | Pre-fix (giant_pandas_50_20260626) | Post-fix (this run) | Delta |
|---|---:|---:|---:|
| Placeholder pool hits | 0 | **350** | +350 |
| Matched slots | 0 | **350** | +350 |
| Pool size max | 0 | **1 529** entries | +1 529 |
| Avg TTFT | 444 ms | **353 ms** | **−20.6%** |
| Avg cached ratio | 20.4% | **53.6%** | **+33.2 pp** |
| Speedup vs `prefix_cache_only` baseline (581 ms) | 1.31× | **1.65×** | **+0.34×** |
| Speedup vs pre-fix (pool turned on) | — | **1.26×** | new |

`_delete_leaf` race fix from Task D held: 50 tasks completed in a
single server lifetime (~5 minutes, 5-7 s/task) without any
auto-relaunch. Pre-fix this would have relaunched the server every
3 tasks (17 chunks).

---

## Per-task rollup

Pool growth + hit-rate summary across the 50 tasks (5 agents each):

- **Task 0**: pool_size 5 → 29, hits 20, matched_slots 20 (warm-up phase, pool just got built by the planner)
- **Task 1-4**: pool grows 40 → 169, hits 8/3/6/9 respectively
- **Task 5-49**: pool continues to grow monotonically across 50 tasks to a max of 1 529 entries (some agents land 0 hits because the slot text didn't match any warmed entry — that's expected; the planner uses a different sibling-file set per task)
- 350 / 250 = **1.4 hits/req** average; **53.6%** of all tokens are now served from cache (vs 20.4% pre-fix)

---

## Driver flags used

```bash
python -m benchmark.multi_workflow.bench_giant_codebase_reuse \
    --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
    --repo-root  results/giant_codebase/pandas_src \
    --out-dir    results/giant_codebase/runs/giant_pandas_50_postfix_20260627_024916 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 30000 --python /home/gfy/.conda/envs/sglang-kvflow/bin/python \
    --mem-fraction-static 0.85 --max-total-tokens 65536 --hicache-ratio 1.5 \
    --max-tasks 50 --agent-count 5 \
    --mode placeholder_knn_reuse --segment-count 5 --max-file-chars 8000 \
    --agent-max-tokens 64 --lossy-max-zero-gap 4 \
    --no-vary-code --chunk-size 50 --warm-planner
```

The two flag changes from the pre-fix run:
- `--vary-code` → `--no-vary-code` (preserve byte-equal slot text across agents so the k-NN body can match warmed entries)
- `--chunk-size 3` → `--chunk-size 50` (no server relaunch mid-bench, pool survives end-to-end)

---

## Companion work in this session

This run was the validation step for two other tasks landed in the
same session on `fix/placeholder-pool-activation`:

1. **Task D — `_delete_leaf` race fix** (commit `d85ca7f4`):
   wrapped `inc_lock_ref` / `dec_lock_ref` in
   `with self.anchor_kv_store_lock:` to prevent the race between
   `lock_ref` mutation and concurrent eviction. Without this fix,
   50 tasks in a single chunk would likely have triggered the
   `RuntimeError: dictionary changed size during iteration` /
   `AssertionError` symptoms documented in HANDOFF.md.

2. **Task E — Direction #3 Phase A: AST chunker** (commit `7fb1a5bb`):
   `python/sglang/srt/mem_cache/ast_chunker.py` mirrors MAScoder's
   `PythonCodeAnchorExtractor` server-side. Phase B will use it to
   recover the 8.2% non-byte-identical matches (cos=1.0, 2-token
   boundary drift) that the current byte-exact path cannot reuse.

---

## Raw data

- `rows.csv` — 250 rows × 52 columns (per-task per-agent metrics)
- `sglang_server.log` — full sglang server log
- `report/REPORT.md` — this file