# 100-case Pass@1 Status (2026-06-09)

## Headline

**UNBLOCKED on 24 GB testbed.** The 5-case OOM is fixed by the
new `--force-evict` flag (sets `SGLANG_RADIX_FORCE_EVICT=1`).

## 5-case discriminative dataset result

Output: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_5_forceevict/`

| Case | synth | apply | test_rc | pass@1 |
|---|---:|---:|---:|---:|
| django-11138 | False | — | — | 0 |
| django-11149 | True | 0 | 1 | 0 |
| matplotlib-21568 | True | 0 | 1 | 0 |
| requests-5414 | True | 0 | 1 | 0 |
| requests-6028 | True | 0 | 1 | 0 |

**pass@1: 0/5** (model quality on this 5-case subset; the 28-case
baseline is 5/28 = 17.9%). All 5 cases completed end-to-end (exit
code 0), no `RuntimeError: Out of memory` in the server log.

## Code changes (commit `c21d3b2f1`)

- `python/sglang/srt/mem_cache/base_prefix_cache.py` —
  `EvictParams.force: bool = False`
- `python/sglang/srt/mem_cache/radix_cache.py` —
  `RadixCache._force_evict_locked` + dispatch in `evict()`
- `python/sglang/srt/mem_cache/common.py` — retry in
  `evict_from_tree_cache` with `force=True` when normal evict freed
  fewer than `num_tokens`
- `python/sglang/srt/mem_cache/test_anchor_match.py` — 4 new unit
  tests (38/38 pass)
- `benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py` —
  new `--force-evict` flag

## Paper update (commit `0d058ef`)

- `evaluation.tex:91` — 100-case expansion deferred paragraph now
  documents the `--force-evict` fix as the unblock path. The 28-case
  result remains the official headline.

## 6 prior unblock attempts (all documented in REPORT.md)

| Step | Approach | Outcome |
|---|---|---|
| 2 | Default | OOM |
| 2.4 | `--kv-allocator-defrag` (allocator defrag) | OOM (defrag path runs but `evictable_leaves` is empty) |
| 2.5 | `--cpu-offload-gb 32` | No OOM, 0-byte patches, aiohttp timeouts |
| 2.6 | `--disable-overlap-schedule` | OOM (reduces `evictable_size_` 58k→44k) |
| 2.7 | aggressive combo | OOM (byte-identical to 2.6) |
| 2.8 | file pre-truncation (--max-file-chars 5000) | No OOM, but search-anchor broken → 0/5 pass@1 |
| 2.9 | `--chunked-prefill-size 5500` | OOM (cache state degrades) |
| 3 | small-ctx | 0/5 (not comparable) |

## Next step

Kick off the 100-case build (8-12 h overnight) + base smoke (6-10 h)
+ pass@1 driver (8-12 h) on the 24 GB RTX 4090 testbed with
`--force-evict` enabled. Datasets:

- 100-case manifest: `results/repo_level_datasets/swe_verified_100_instances.json`
- 5-case discriminative: `results/swebench_local_envs/manifest_5.json`
  + `_5_new_discriminative_instances.json`

Full details: `REPORT.md` (next to this file) and
`/home/gfy/CodeMAS_Project/sglang-kvflow/HANDOFF.md` (new
"100-case Pass@1 expansion status" section).
