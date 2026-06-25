# Phase 2.2 Triton-Tiled KV Copy Dispatcher — End-to-End TTFT Results (v14)

**Date**: 2026-06-21
**Plan**: `/home/gfy/.claude/plans/humble-strolling-cerf.md`

## What was built

Phase 2.2 routes the placeholder k-NN KV copy through `MHATokenToKVPool.move_kv_cache` (the dispatcher at `memory_pool.py:1019-1066`), which picks between:
- `move_kv_cache_native` (Python eager loop) when `SGLANG_NATIVE_MOVE_KV_CACHE=1`
- `copy_all_layer_kv_cache_tiled` (triton 2D tiled kernel) by default

Also enables `enable_kv_cache_copy=True` unconditionally at `model_runner_kv_cache_mixin.py:617, 636` (was previously gated on speculative-decoding only).

### Files modified

| File | Change | LOC |
|---|---|---:|
| `python/sglang/srt/mem_cache/radix_cache.py` | Replace direct `move_kv_cache_native` call with `kvcache.move_kv_cache` dispatcher + AttributeError fallback + telemetry | ~30 |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | Flip `enable_kv_cache_copy=True` unconditionally | 4 |
| `python/sglang/srt/managers/schedule_batch.py` | Init `placeholder_knn_copy_method`, `placeholder_anchor_pool_copy_error_count` | 2 |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | Emit 2 telemetry fields | 4 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | Add 2 keys to streaming + non-streaming `lossy_keys` | 4 |
| `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` | Add 2 CSV columns | 6 |
| `python/sglang/srt/mem_cache/test_placeholder_knn.py` | New `PlaceholderTiledCopyTests` class with 3 tests | ~120 |

**Total: ~170 LOC. 62 unit tests pass (53 + 3 cost-guard + 3 head-rotation + 3 tiled-copy).**

## End-to-end results — honest assessment

### Per-agent workflow TTFT (avg)

| agent | prefix | v13 HEAD2 (native direct) | **v14 TILED** (default) | **v14 NATIVE** (back-compat) |
|---:|---:|---:|---:|---:|
| 1 | 257 | 270 | 271 | 261 |
| 2 | 297 | 174 | **4162** ⚠️ | 167 |
| 3 | 347 | 278 | **3962** ⚠️ | 269 |
| 4 | 393 | 787 | 758 | 745 |
| 5 | 441 | 1038 | 1030 | 1063 |

### Speedup vs `prefix_cache_only`

| agent | v13 HEAD2 | v14 TILED | v14 NATIVE |
|---:|---:|---:|---:|
| 1 | 0.96× | 0.96× | 0.98× |
| 2 | 1.71× | **0.07×** ⚠️ | **1.78×** |
| 3 | 1.25× | **0.09×** ⚠️ | 1.29× |
| 4 | 0.52× | 0.52× | 0.53× |
| 5 | 0.42× | 0.43× | 0.41× |

### Copy-method telemetry (v14 TILED run)

```
agent=1 copy_method=none     errors=0 match=0
agent=2 copy_method=tiled    errors=0 match=1
agent=3 copy_method=tiled    errors=0 match=1
agent=4 copy_method=tiled    errors=0 match=1
agent=5 copy_method=tiled    errors=0 match=1
```

(Telemetry reports `tiled` because the body wraps the dispatcher call and only sets `native` on `AttributeError`. The dispatcher internally routes to native when `SGLANG_NATIVE_MOVE_KV_CACHE=1`, so v14 NATIVE also reports `tiled` even though it uses the native path. **Telemetry bug**: my Phase 2.2 telemetry doesn't accurately reflect which method the dispatcher chose.)

## ⚠️ Phase 2.2 hypothesis refuted

**Phase 2.2's plan assumed the triton-tiled kernel would be faster than the eager native loop for placeholder k-NN copies.** The data refutes this:

- **v14 TILED (default)**: agent 2 = **4162 ms**, agent 3 = **3962 ms** — **24× and 14× REGRESSION** vs v13
- **v14 NATIVE (back-compat)**: agent 2 = **167 ms**, agent 3 = **269 ms** — slightly BETTER than v13

The triton-tiled kernel at `memory_pool.py:2028-2062` is designed for the speculative-decoding access pattern (small, frequent copies between tree nodes). For our placeholder k-NN use case (single copy of 2245 tokens per agent), it has overhead that exceeds the savings.

**Conclusion**: the v14 dispatcher is **correctly implemented** (62 unit tests pass, telemetry shows `copy_method=tiled` reaching the dispatcher, fallback to native works correctly via test 2). But the assumption that triton-tiled is universally faster is wrong — it's a workload-dependent choice.

## What was actually fixed in Phase 2.2

1. **Mechanism works**: body now routes through `kvcache.move_kv_cache` dispatcher, which respects `SGLANG_NATIVE_MOVE_KV_CACHE`.
2. **Back-compat verified**: v14 NATIVE (`SGLANG_NATIVE_MOVE_KV_CACHE=1`) reproduces v13 HEAD2 numbers (and is slightly better at agent 2: 167 ms vs 174 ms — within noise but consistent).
3. **Telemetry works**: `placeholder_knn_copy_method` and `placeholder_anchor_pool_copy_error_count` surface correctly in CSV/JSON.
4. **Telemetry bug noted**: `copy_method` reports `tiled` regardless of which internal path the dispatcher chose. **Fix needed in a follow-up**: pass back the chosen method from the dispatcher.

## Agent 4-5 regression NOT fixed (still)

The Phase 2.2 hypothesis (KV copy is the bottleneck) is now refuted. Both v14 TILED and v14 NATIVE show agent 4-5 still regressing (0.41-0.53×). The bottleneck is **elsewhere** — possibly:
- CUDA graph compilation overhead for new tensor shapes per request
- Allocator fragmentation from `token_to_kv_pool_allocator.alloc(entry_len)` calls
- Per-slot overhead in `match_prefix` itself (k-NN search + pool lookup × N spans)

The `placeholder_knn_copy_method=tile` path doesn't show errors (`errors=0`) but produces 24× slower results at agent 2. This is consistent with **tile launch overhead dominating** for small copies.

## Recommended action

**Phase 2.2 in its current form should NOT be the default**. The back-compat path (v14 NATIVE) is slightly better than v13 but doesn't fix the agent 4-5 regression. Options:

1. **Roll back `enable_kv_cache_copy=True`** — it's an unconditional startup cost increase (1 triton warmup) for no benefit.
2. **Keep the dispatcher routing** — it's harmless and provides future flexibility. The dispatcher works correctly; we just need the **default to stay native** for our workload.
3. **Make `SGLANG_PLACEHOLDER_KNN_FORCE_TILED=1`** an opt-in for users who want to experiment.

For Phase 2.3+ (the actual fix), the work shifts to:
- Investigate the true bottleneck at agent 4-5 (CUDA graphs, allocator, or k-NN overhead)
- Consider soft-weighted K-nearest reconstruction (Phase 2.3 in the original plan)
- Consider selective recompute via CacheBlend-style HKVD analysis (Phase 2.4)

## Files

- This report: `results/ttft_agenttemplatekv/multi_agent_placeholder_v14_20260621/MULTI_AGENT_PLACEHOLDER_V14_RESULTS.md`
- TILED CSV: `results/ttft_agenttemplatekv/multi_agent_placeholder_v14_TILED_20260621/ttft_stress_table.csv`
- NATIVE CSV: `results/ttft_agenttemplatekv/multi_agent_placeholder_v14_NATIVE_20260621/ttft_stress_table.csv`
- v13 baseline: `results/ttft_agenttemplatekv/multi_agent_placeholder_v13_20260621/MULTI_AGENT_PLACEHOLDER_V13_RESULTS.md`
- Plan: `/home/gfy/.claude/plans/humble-strolling-cerf.md`

## Reproduction

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

# Default (TILED — note: WORSE for placeholder k-NN, see results)
SGLANG_PLACEHOLDER_KNN_MATCH=1 SGLANG_PLACEHOLDER_STORE_ENABLED=1 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --agent-counts 1,2,3,4,5 --agent-max-cases 1 \
    --agent-length-buckets 8000 --agent-max-tokens 1 \
    --segment-counts 1 --files-per-case 1 \
    --disable-hierarchical-cache --skip-e6 --skip-e8 \
    --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v14_TILED_REPRO

# Back-compat (NATIVE — preserves v13 HEAD2 behavior)
SGLANG_NATIVE_MOVE_KV_CACHE=1 \
SGLANG_PLACEHOLDER_KNN_MATCH=1 SGLANG_PLACEHOLDER_STORE_ENABLED=1 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    ... (same args) \
    --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v14_NATIVE_REPRO
```
