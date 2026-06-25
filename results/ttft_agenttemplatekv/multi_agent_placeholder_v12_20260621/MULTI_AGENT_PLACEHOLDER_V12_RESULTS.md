# Phase 2 Cost-Aware Abort Guard — End-to-End TTFT Results (v12)

**Date**: 2026-06-21
**Plan**: `/home/gfy/.claude/plans/humble-strolling-cerf.md`
**Goal**: Add a cost-aware abort guard to `_try_placeholder_knn_lossy_match_body`
that skips the per-slot KV copy + RoPE rotation when `entry_len × layer_num > SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS`. Fixes the agent 4-5 regression observed in v11n (`MULTI_AGENT_PLACEHOLDER_RESULTS.md`).

## What was built

| File | Change |
|---|---|
| `python/sglang/srt/mem_cache/radix_cache.py` | New env var `SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS` (default `114688`); cost guard in `_try_placeholder_knn_lossy_match_body` after `entry_len = min(...)`; per-span INFO log; telemetry increment. |
| `python/sglang/srt/managers/schedule_batch.py` | Init `placeholder_anchor_pool_skipped_cost_count = 0` in `Req.__init__`. |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | Emit `placeholder_anchor_pool_skipped_cost_count` to `observability_fields`. |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | Add key to streaming + non-streaming `lossy_keys`. |
| `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` | Add CSV column extraction + row dict field. |
| `python/sglang/srt/mem_cache/test_placeholder_knn.py` | New `_FakeRadixCacheWithBody` stub + 3 cost-guard tests. |

**Total: ~157 LOC (26 server-side + 130 tests + 1 plumbing).**

## Unit tests

All **56 tests pass** (53 from v11 + 3 new cost-guard tests):

```
test_placeholder_knn.PlaceholderCostGuardTests.test_cost_guard_aborts_large_copy      OK
test_placeholder_knn.PlaceholderCostGuardTests.test_cost_guard_allows_small_copy       OK
test_placeholder_knn.PlaceholderCostGuardTests.test_cost_guard_disabled_with_zero_threshold  OK
```

The tests use a `_FakeRadixCacheWithBody` stub with `token_to_kv_pool_allocator.get_kvcache().layer_num = 28` (Qwen2.5-7B) and verify:
- `entry_len=4096, layer_num=28, threshold=57344 → cost=114688 > 57344 → skip_cost=1` (aborts)
- `entry_len=512, layer_num=28, threshold=57344 → cost=14336 < 57344 → skip_cost=0, miss_count=1` (allows alloc-fail path)
- `max_rope_ops=0` → guard is off regardless of cost (v10c convention)

## End-to-end calibration sweep

The same multi-agent stress harness (`--agent-counts 1,2,3,4,5 --agent-max-cases 1 --agent-length-buckets 8000 --agent-max-tokens 1`) was run with different `SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS` values:

### Per-agent workflow TTFT (avg)

| Threshold | agent 1 | agent 2 | agent 3 | agent 4 | agent 5 | cost_guard fires? |
|---|---:|---:|---:|---:|---:|---|
| `prefix_cache_only` (no k-NN) | 259 | 299 | 358 | 390 | 441 | n/a |
| **v11n guard off** | 262 | **156** | **283** | 745 | 1019 | no |
| **v12 default (114688)** | 262 | 156 | 254 | 741 | 1047 | no (cost=62860 < 114688) |
| **v12 80K** | 262 | 174 | 275 | 738 | 1032 | no |
| **v12 50K** | 275 | 584 | 847 | 1114 | 1393 | **always** (worse — too aggressive) |

### Per-agent speedup vs prefix_cache_only

| Threshold | agent 1 | agent 2 | agent 3 | agent 4 | agent 5 |
|---|---:|---:|---:|---:|---:|
| v11n guard off | 0.99× | **1.92×** | **1.27×** | 0.52× | 0.43× |
| v12 default (114688) | 0.99× | **1.92×** | **1.41×** | 0.53× | 0.42× |
| v12 80K | 0.99× | 1.81× | 1.31× | 0.54× | 0.43× |
| v12 50K | 0.94× | 0.51× | 0.42× | 0.35× | 0.32× |

### Cost telemetry

For v12 default (114688) — no slots aborted:

| agent | role | hit | match | skipped_tokens | skip_cost |
|---:|---|---:|---:|---:|---:|
| 1 | implementer | 0 | 0 | 0 | 0 |
| 2 | implementer | 1 | 1 | 2245 | 0 |
| 2 | debugger | 1 | 1 | 2245 | 0 |
| 3 | implementer | 1 | 1 | 2245 | 0 |
| 3 | debugger | 1 | 1 | 2245 | 0 |
| 3 | reviewer | 1 | 1 | 2245 | 0 |
| 4 | implementer | 1 | 1 | 2245 | 0 |
| 4 | reviewer | 1 | 1 | 2245 | 0 |
| 5 | debugger | 1 | 1 | 2245 | 0 |
| 5 | verifier | 1 | 1 | 2245 | 0 |

For v12 50K — every slot aborted (cost=62860 > 50000):

| agent | role | hit | match | skipped_tokens | skip_cost |
|---:|---|---:|---:|---:|---:|
| 2-5 | all | 0 | 0 | 0 | 1 |

## Honest assessment

**Phase 2 as designed (per-slot static threshold) does NOT fix the v11n regression at agent 4-5.** The mechanism works correctly:
- Guard fires when `cost > threshold` ✓
- Guard is off when `cost ≤ threshold` ✓
- `max_rope_ops=0` disables cleanly ✓
- Telemetry populates correctly ✓
- Per-span INFO log fires on abort ✓

But the **calibration is wrong**. The per-slot cost is constant (62860 = 2245 tokens × 28 layers) across all agents 2-5 because the pool entry size doesn't grow — but the **per-agent savings shrinks** with agent_count because more upstream text accumulates and the "saved prefill" becomes a smaller fraction of total work.

The 50K threshold aborts everything (worse than baseline) because the cost model doesn't account for **cumulative overhead** (k-NN search, pool lookup, refcount, write-back).

## What Phase 2 actually delivers

1. **Mechanism is correct**: when an individual copy is too expensive, the guard aborts it cleanly.
2. **Telemetry is complete**: `placeholder_anchor_pool_skipped_cost_count` surfaces in CSV; per-span INFO log fires.
3. **Tests pass**: 56 unit tests including 3 new cost-guard tests.
4. **Default is safe**: `114688` is too high to fire in this benchmark, so behavior matches v11n (no regression vs the v11 baseline).
5. **Off switch works**: `SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS=0` cleanly disables.

## What's needed (Phase 2.1 — separate task)

The v11n regression at agent 4-5 is **not caused by per-slot cost** — it's caused by:
- Cumulative per-agent overhead (k-NN search + pool lookup × N slots per request)
- Long slots (~2245 tokens) where copy+rotation cost ≈ prefill cost

A working fix would require either:
- **Per-agent cumulative cost** tracking: abort the whole request when cumulative copy cost exceeds cumulative savings.
- **Adaptive cost measurement**: time the first copy, predict total cost, abort if predicted > savings.
- **RoPE delta rotation cost reduction**: chunk the rotation, run on a worker pool, or skip rotation when `delta == 0`.

These are out of scope for Phase 2 v1 (which provides the mechanism + telemetry + unit tests). Phase 2.1 would address the actual algorithm refinement.

## Default recommendation

**Keep `SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS=114688` as the v12 default.** It's effectively a no-op in this benchmark (cost never exceeds threshold) but provides the infrastructure for future calibration. Users who want to be conservative can set `SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS=50000` (aggressive aborts, loses speedup) or `SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS=0` (disable guard entirely).

## Reproduction

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

# Default (114688, no behavior change vs v11n)
SGLANG_PLACEHOLDER_KNN_MATCH=1 SGLANG_PLACEHOLDER_STORE_ENABLED=1 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --agent-counts 1,2,3,4,5 --agent-max-cases 1 \
    --agent-length-buckets 8000 --agent-max-tokens 1 \
    --segment-counts 1 --files-per-case 1 \
    --disable-hierarchical-cache --skip-e6 --skip-e8 \
    --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v12_REPRO

# Aggressive (50000, all slots abort — demonstrates guard fires)
SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS=50000 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --agent-counts 1,2,3,4,5 --agent-max-cases 1 \
    --agent-length-buckets 8000 --agent-max-tokens 1 \
    --segment-counts 1 --files-per-case 1 \
    --disable-hierarchical-cache --skip-e6 --skip-e8 \
    --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v12_50K_REPRO

# Off (cleanly disables guard)
SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS=0 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --agent-counts 1,2,3,4,5 --agent-max-cases 1 \
    --agent-length-buckets 8000 --agent-max-tokens 1 \
    --segment-counts 1 --files-per-case 1 \
    --disable-hierarchical-cache --skip-e6 --skip-e8 \
    --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v12_DISABLED_REPRO
```

## Files

- This report: `results/ttft_agenttemplatekv/multi_agent_placeholder_v12_20260621/MULTI_AGENT_PLACEHOLDER_V12_RESULTS.md`
- Raw CSV (default): `results/ttft_agenttemplatekv/multi_agent_placeholder_v12_20260621/ttft_stress_table.csv`
- Raw CSV (50K aggressive): `results/ttft_agenttemplatekv/multi_agent_placeholder_v12_50K_20260621/ttft_stress_table.csv`
- Raw CSV (DISABLED): `results/ttft_agenttemplatekv/multi_agent_placeholder_v12_DISABLED_20260621/ttft_stress_table.csv`
- v11 baseline: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/MULTI_AGENT_PLACEHOLDER_RESULTS.md`
- v11 implementation: `results/selective_ast_reuse/placeholder_knv_kv_reuse_v11_20260621.md`
- Plan: `/home/gfy/.claude/plans/humble-strolling-cerf.md`
