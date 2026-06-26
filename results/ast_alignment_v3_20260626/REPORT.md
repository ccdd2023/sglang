# AST-Alignment Partial-Match Hit Rate — Measurement Report v2

**Date**: 2026-06-26  
**Plan**: `/home/gfy/.claude/plans/whimsical-stirring-thimble.md` (Direction #3 measurement)  
**Workload**: 60-case stratified sweep (manifest_500.json), 5 agents per task, segment_count=3, mode=`placeholder_knn_reuse`, Qwen2.5-3B-Instruct  
**Fixes applied this session**: (1) HiRadixCache.match_prefix now calls placeholder k-NN body (previously omitted); (2) cap `overlap_len` at `entry_len` in `_try_placeholder_knn_lossy_match_body` to avoid negative copy_len when prefix cache overshoots slot end.

## Headline

- **Requests sent**: 100
- **Placeholder pool hits**: 184
- **Placeholder pool misses**: 117
- **AST_ALIGN structured rows**: 184
- **Prefix-cache reuse ratio**: 0.5997

## AST-Alignment Analysis

For each placeholder pool match, the structured log captures slot/match token ranges + sha1 of slot/match text.

| Metric | Count | % |
|--------|------:|--:|
| AST_ALIGN rows | 184 | — |
| cos ≥ 0.99 (near-perfect) | 184 | 100.0% |
| byte-identical (slot_sha1 == match_sha1) | 169 | 91.8% |
| start_token aligned | 170 | 92.4% |
| end_token aligned | 169 | 91.8% |
| **both start AND end aligned (AST-aligned hit rate)** | **169** | **91.8%** |

## Per-Agent Breakdown

| Agent | Requests | Pool Hits | Pool Misses | Mean Cached Ratio | Mean TTFT (ms) |
|-------|---------:|----------:|------------:|-------------------:|---------------:|
| `auditor` | 20 | 36 | 24 | 0.5872 | 355 |
| `debugger` | 20 | 38 | 22 | 0.6204 | 339 |
| `implementer` | 20 | 34 | 27 | 0.5549 | 437 |
| `reviewer` | 20 | 38 | 22 | 0.6204 | 335 |
| `verifier` | 20 | 38 | 22 | 0.6198 | 339 |

## Decision

**AST-aligned hit rate = 91.8%** (≥ 30% threshold). **Direction #3 is worth pursuing.** The placeholder k-NN body is operational and finds AST-aligned matches at high rate.

## Fixes Applied This Session

**Bug 1: HiRadixCache.match_prefix never invoked placeholder k-NN body.**

The HiRadixCache class (used by sglang when `--enable-hierarchical-cache` is on) had its own `match_prefix` override at `/home/gfy/CodeMAS_Project/sglang-kvflow/python/sglang/srt/mem_cache/hiradix_cache.py:1398` that called `_resolve_lossy_match` and `_try_lossy_fuzzy_match` but **not** `_try_placeholder_knn_lossy_match`. The placeholder pool was being stored (via `cache_finished_req` calling `_store_placeholder_anchor_kv`) but never queried, so `placeholder_anchor_pool_hit_count` stayed at 0 across all requests. **Fix**: added the missing `_try_placeholder_knn_lossy_match` call to HiRadixCache.match_prefix (mirroring radix_cache.py:686-700).

**Bug 2: `copy_len` could go negative when prefix cache overshoots slot end.**

In `_try_placeholder_knn_lossy_match_body` at `radix_cache.py:2782`, the calc was `copy_len = entry_len - overlap_len` where `overlap_len = max(0, prefix_len - start)`. When `prefix_len > end` (hicache shared across salts, prefix cache can extend past the slot), `overlap_len > entry_len`, producing negative `copy_len` which was then skipped via the `copy_len <= 0` check (so no hit counted). **Fix**: cap `overlap_len = min(overlap_len, entry_len)` before the subtraction.

## Caveats and Open Items

1. **`placeholder_anchor_store_entry_count` is reported as 0 in the response metadata** even when the pool grew (timing issue — `_store_placeholder_anchor_kv` runs in `cache_finished_req` AFTER `_append_lossy_observability` reads req attributes for streaming metadata). Server-side POOL_DIAG confirms actual storage. Fix would move `_store_placeholder_anchor_kv` to run during prefill (e.g., in `_try_placeholder_knn_lossy_match_body` after each match).
2. **The remaining ~8% non-byte-identical matches** (slot text differs slightly from match text but cos=1.0) suggest whitespace / tokenization divergence. AST-boundary chunked prefill (Direction #3) would help here by allowing partial-match reuse at function boundaries.