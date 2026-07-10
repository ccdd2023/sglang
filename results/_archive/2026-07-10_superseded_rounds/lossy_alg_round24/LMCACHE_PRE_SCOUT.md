# R24 LMCache Pre-Scout (2026-07-06)

## TL;DR

**LMCache is already integrated into sglang mainline**, which our sglang-kvflow fork
is based on. The integration is `LMCRadixCache` (subclass of `RadixCache`), enabled by
`--enable-lmcache` + `LMCACHE_CONFIG_FILE=/path/to/config.yaml`. **Tier C5 in the
directions memo is now confirmed feasible at 1-2 weeks**, possibly faster.

The integration does NOT yet handle our chunked-reuse + selective-refresh decisions.
Bridging our `placeholder_chunk_pool_hit_count` path onto LMCache's `match_prefix`
override is the actual engineering work.

## What exists

| Path | Description |
|---|---|
| `python/sglang/srt/mem_cache/storage/lmcache/__init__.py` | LMCache storage backend |
| `python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py` | `LMCRadixCache(RadixCache)` |
| `python/sglang/srt/mem_cache/storage/lmcache/example_config.yaml` | Config template (chunk_size=256, local_cpu=true) |
| `python/sglang/srt/mem_cache/storage/lmcache/unit_test.py` | Unit tests included |
| `python/sglang/srt/server_args.py:enable_lmcache: bool = False` | `--enable-lmcache` flag |

### LMCRadixCache overrides
- `match_prefix` — same RadixCache signature
- `cache_finished_req` — promotes matched blocks into CPU store
- `evict` — policy for CPU store

### What's NOT there
- Per-chunk byte-exact match across agent boundaries (KVCOMM-style)
- Selective Refresh (skip largest X% of chunks)
- Direction A preamble (SGLANG_PRECOMPUTE_CANONICAL_PREFIX)
- Precompute-on-disk pool integration (only CPU offloading)
- Cross-position content-derived slot_id

## Verdict

**Tier C5 (LMCache integration v0)** is feasible and 1-2 weeks. The work is:
1. Install `lmcache` from PyPI (`pip install lmcache`); check no version conflict
2. Run sglang-kvflow benchmark with `--enable-lmcache` and see whether
   `LMCRadixCache` matches our `placeholder_chunk_pool_hit_count` path or
   shadows it (they may conflict)
3. Wire Selective Refresh decisions into `LMCRadixCache.match_prefix` override
4. Compare verdicts: R19 BEST vs R19+LMCache (verdict task-completion metric)

If LMCache shadows our chunk pool → we have a 1-tier simpler architecture (good).
If LMCache runs alongside → we have both prefix cache + chunk pool (better).

**However**, this is OUT OF SCOPE for the current 23-round session unless user signs
off on R25 (a real experiment, not just pre-scout).

## Files referenced

- `python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py:44` — `class LMCRadixCache(RadixCache)`
- `python/sglang/srt/server_args.py` — `enable_lmcache` flag (L4871)
- `python/sglang/srt/mem_cache/storage/lmcache/example_config.yaml` — config template

## Memory update

Memory `r24-verdict-algorithmic-ceiling` and `c5-lmcache-integration-feasible-2026-07-06`
now point to this scout.

---

*Pre-scout only — no new benchmark, no new code. ~5 min read of the LMCache integration
surface.*
