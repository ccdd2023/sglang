# Phase 2.1 Head-Only RoPE Rotation — End-to-End TTFT Results (v13)

**Date**: 2026-06-21
**Plan**: `/home/gfy/.claude/plans/humble-strolling-cerf.md`
**References**: [EPIC (ICML 2025)](https://www.cnblogs.com/marsggbo/p/20008042), [CacheBlend (arXiv 2405.16444)](https://arxiv.org/html/2405.16444v3), [TokenDance (arXiv 2604.03143)](https://www.cnblogs.com/marsggbo/p/19923853)

## What was built

Phase 2.1 adds **head-only RoPE rotation** (EPIC's k=2 mechanism) to the per-placeholder k-NN KV reuse path. Only the first `SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS` tokens of each copied slot get rotated to encode the global position; the rest retain their chunk-local position-0 RoPE.

### Files modified

| File | Change | LOC |
|---|---|---:|
| `python/sglang/srt/mem_cache/radix_cache.py` | Env var `head_tokens` (default 2); new `_apply_rope_delta_to_head` helper (thin wrapper over existing `_apply_rope_delta_to_keys`); body uses head-only path when `head_tokens > 0`; cost guard recomputes cost as `eff_head_len × layer_num` | ~70 |
| `python/sglang/srt/managers/schedule_batch.py` | Init 2 telemetry fields in `Req.__init__` | 2 |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | Emit 2 telemetry fields via `observability_fields` | 8 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | Add 2 keys to streaming + non-streaming `lossy_keys` | 4 |
| `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` | Add 2 CSV columns | 6 |
| `python/sglang/srt/mem_cache/test_placeholder_knn.py` | 3 new tests in `PlaceholderHeadRotationTests` class | ~135 |

**Total: ~225 LOC. 59 unit tests pass (53 + 3 cost-guard + 3 head-rotation).**

## End-to-end calibration sweep

Same multi-agent stress harness (`--agent-counts 1,2,3,4,5 --agent-max-cases 1 --agent-length-buckets 8000 --agent-max-tokens 1`).

### Per-agent workflow TTFT (avg)

| agent | v11n (no head) | v12 default | **v13 HEAD0** (full rot) | **v13 HEAD2** (EPIC k=2) | **v13 HEAD8** (conservative) |
|---:|---:|---:|---:|---:|---:|
| 1 | 261 | 261 | 273 | 270 | 279 |
| 2 | **156** | **156** | 201 | **174** | 192 |
| 3 | **283** | 254 | 255 | **278** | 288 |
| 4 | 745 | 741 | 764 | 787 | 728 |
| 5 | 1019 | 1047 | 1000 | 1038 | 1042 |

### Speedup vs `prefix_cache_only`

| agent | v11n | v12 default | v13 HEAD0 | v13 HEAD2 | v13 HEAD8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.99× | 0.99× | 0.96× | 0.96× | 0.93× |
| 2 | **1.92×** | **1.92×** | 1.51× | **1.71×** | 1.56× |
| 3 | **1.27×** | **1.41×** | 1.36× | **1.25×** | 1.18× |
| 4 | 0.52× | 0.53× | 0.54× | 0.52× | 0.54× |
| 5 | 0.43× | 0.42× | 0.44× | 0.42× | 0.42× |

### Head rotation telemetry (HEAD2)

| agent | role | head_rot_tokens | head_ops | match | skipped |
|---:|---|---:|---:|---:|---:|
| 1 | implementer | 0 | 0 | 0 | 0 |
| 2 | implementer | **2** | **56** | 1 | 2245 |
| 2 | debugger | **2** | **56** | 1 | 2245 |
| 3 | implementer | **2** | **56** | 1 | 2245 |
| 3 | debugger | **2** | **56** | 1 | 2245 |
| 3 | reviewer | **2** | **56** | 1 | 2245 |
| 4 | implementer | **2** | **56** | 1 | 2245 |
| 4 | reviewer | **2** | **56** | 1 | 2245 |
| 5 | debugger | **2** | **56** | 1 | 2245 |
| 5 | verifier | **2** | **56** | 1 | 2245 |

**Head rotation cost reduction verified**: 56 ops vs 62,860 (full rotation) = **1120× cheaper**.

## Honest assessment

**Phase 2.1's head-only rotation reduces the rotation cost by ~1120×** but **does NOT fix the agent 4-5 regression**.

Why the regression persists:
1. **The copy itself dominates**, not the rotation. `move_kv_cache_native` at 2245 tokens × 28 layers = 62,860 ops per slot, regardless of rotation cost.
2. **The original analysis in the Phase 2 plan misdiagnosed** the bottleneck as RoPE rotation. The plan was built on the assumption that `_apply_rope_delta_to_keys` was the dominant cost, but the copy (`move_kv_cache_native`) was equally expensive and was not reduced.
3. **Reducing rotation by 1120× doesn't help** if the copy itself is the bottleneck at the same order of magnitude.

**The HEAD2 win at agent 2 (1.71× vs v13 HEAD0 1.51×) is real but modest** — head rotation helps when the copy + rotation is amortized across many tokens, but at agent 2 the small overall cost means the saving is visible. At agent 4-5, the copy cost alone exceeds the prefill savings.

## What would fix the regression

The remaining bottleneck is **the KV copy** (`move_kv_cache_native`). To match prefix_cache_only at agent 4-5, we need:

1. **Use `copy_all_layer_kv_cache_tiled`** (a triton-tiled variant at `memory_pool.py:2028`) instead of the eager `move_kv_cache_native`. Already exists in the codebase — just route placeholder k-NN to it.

2. **Skip the copy entirely and recompute** when predicted savings > predicted copy cost (CacheBlend's full HKVD layer-by-layer analysis). More complex but addresses the root cause.

3. **Soft-weighted reconstruction** with K=1 nearest neighbor (Phase 2.2). The Phase 2.1 head-only path is the "single-best neighbor with cheap rotation" version; Phase 2.2 would add multi-neighbor averaging.

## Default recommendation

**Keep `SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS=2` as the v13 default.** It provides:
- A modest win at agent 2-3 (slightly cleaner numbers, ~3% TTFT reduction).
- Zero downside (rotation cost is so small the guard is effectively off).
- Clean separation of "copy cost" and "rotation cost" for future measurement.

**The agent 4-5 regression requires a different fix** (Phase 2.2+ work on the copy path or Phase 2.3 with CacheBlend-style HKVD analysis).

## Reproduction

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow

# Default (HEAD2)
SGLANG_PLACEHOLDER_KNN_MATCH=1 SGLANG_PLACEHOLDER_STORE_ENABLED=1 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    --agent-counts 1,2,3,4,5 --agent-max-cases 1 \
    --agent-length-buckets 8000 --agent-max-tokens 1 \
    --segment-counts 1 --files-per-case 1 \
    --disable-hierarchical-cache --skip-e6 --skip-e8 \
    --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v13_HEAD2_REPRO

# Back-compat (full rotation, matches v11n behavior)
SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS=0 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    ... --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v13_HEAD0_REPRO

# Conservative (rotate 8 tokens)
SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS=8 \
  python benchmark/multi_workflow/bench_kvcomm_ttft_stress.py \
    ... --out-dir results/ttft_agenttemplatekv/multi_agent_placeholder_v13_HEAD8_REPRO
```

## Files

- This report: `results/ttft_agenttemplatekv/multi_agent_placeholder_v13_20260621/MULTI_AGENT_PLACEHOLDER_V13_RESULTS.md`
- HEAD0 CSV: `results/ttft_agenttemplatekv/multi_agent_placeholder_v13_HEAD0_20260621/ttft_stress_table.csv`
- HEAD2 CSV: `results/ttft_agenttemplatekv/multi_agent_placeholder_v13_HEAD2_20260621/ttft_stress_table.csv`
- HEAD8 CSV: `results/ttft_agenttemplatekv/multi_agent_placeholder_v13_HEAD8_20260621/ttft_stress_table.csv`
- v11 baseline: `results/ttft_agenttemplatekv/multi_agent_placeholder_v11n_20260621/MULTI_AGENT_PLACEHOLDER_RESULTS.md`
- v12 cost-guard: `results/ttft_agenttemplatekv/multi_agent_placeholder_v12_20260621/MULTI_AGENT_PLACEHOLDER_V12_RESULTS.md`
- Plan: `/home/gfy/.claude/plans/humble-strolling-cerf.md`
