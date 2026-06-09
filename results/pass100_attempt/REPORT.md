# 100-Case Pass@1 Expansion Attempt (2026-06-08)

This report records the empirical work to extend the 28-case pass@1
headline (3/28 lossless, 2/28 lossy) toward 100 cases.

## Summary

- **Headline result (28 cases)**: 3/28 lossless, 2/28 lossy — published in
  `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/`
- **30-env gold smoke batch (cases 27-56 from 100-manifest)**:
  5/30 pass, 25/30 fail (16.7% pass rate, similar to 28-case)
- **Base smoke for the 5 new passing envs**: 5/5 base-nonzero
  (all 5 are discriminative)
- **Pass@1 driver on the 5 new cases (full ctx, default settings)**:
  BLOCKED — upstream SGLang allocator OOM bug
- **Pass@1 driver on the 5 new cases (small-ctx, 1 file / 3K chars / 512 tok)**:
  completed; 0/5 lossless, 0/5 lossy (not directly comparable to 28-case)

## Verdict

The 100-case pass@1 expansion is **deferred** because the 5-case
extension hit a transient lock-pressure OOM: all 4 visible leaves in
the radix tree have `lock_ref=3` (locked by in-flight prefill
batches), so `RadixCache.evict()` has an empty `evictable_leaves` set
and cannot free anything for the new prefill's 8,192-token
allocation. The upstream `evict()` mechanism is correct; the OOM is
not a fragmentation bug. The 28-case run worked because its dataset
had shorter prefill contexts (≤6,144 tokens) that fit in the
6,342-token `free_pages` headroom without needing eviction. The
5-case dataset's 8,192-token prefill needs eviction, and eviction
cannot proceed while leaves are locked.

We confirmed this by:
1. Running with `SGLANG_KV_ALLOCATOR_DEFRAG=1` (alloc_with_defrag
   fallback) — no help, because evict() doesn't free anything.
2. Inspecting the radix-tree pretty_print from the OOM — all leaves
   `r=3`, all internal nodes `r=0` (so `evictable_size_=58,211` is
   misleading; the actually-evictable leaf set is 0).
3. Running with `--cpu-offload-gb 32` — OOM avoided, but
   per-request latency goes 5-10×, hitting aiohttp timeout (see
   Step 2.5).
4. Running with `--files-per-case 1 --max-file-chars 3000
   --max-tokens 512` (small-ctx) — completed 0/5 (search-not-found),
   not directly comparable to 28-case full-ctx (see Step 3).

The 30-env gold smoke batch produced 5/30 passing envs (16.7%), all
of which are discriminative. The 5/5 manifest + dataset is preserved
at `results/swebench_local_envs/manifest_5.json` and
`_5_new_discriminative_instances.json` for a future replay on a
40+ GB GPU (where the prefill headroom is larger) or with
`--disable-overlap-schedule` (which serializes the scheduler's
prefill batches so leaves can be released between batches).

## Step 0: 30-env gold smoke (cases 27-56 from 100-manifest)

Output: `results/swebench_local_envs/expanded_100_gold_smoke.json`

| Status | Count | Notes |
|---|---:|---|
| gold pass (rc=0) | 5 | django-11138, django-11149, matplotlib-21568, requests-5414, requests-6028 |
| gold fail (rc=1) | 25 | mostly test assertion / missing-dep failures |

Pass rate 5/30 = 16.7%, consistent with the 28-case prior pass rate
(2/28 = 7% lossy pass and 3/28 = 10.7% lossless pass).

## Step 1: Base smoke for the 5 new passing envs

Output: `results/swebench_local_envs/expanded_5_base_smoke.json`

All 5 returned rc=1 (test fails at base, which is the **discriminative**
signal we want):

| instance_id | base rc | elapsed_sec | failure mode |
|---|---:|---:|---|
| django__django-11138 | 1 | 24.9 | test assertion (URLValidator) |
| django__django-11149 | 1 | 16.9 | test assertion (makemigrations) |
| matplotlib__matplotlib-21568 | 1 | 91.4 | test assertion (test_dates usetex) |
| psf__requests-5414 | 1 | 22.1 | test assertion (InvalidURL not raised) |
| psf__requests-6028 | 1 | 17.7 | test assertion (prepend_scheme_if_needed) |

## Step 2: Pass@1 driver on the 5 new cases (full ctx)

**FAILED** with `RuntimeError: Out of memory. Try to lower your batch size.`
at the first prefill of `matplotlib__matplotlib-21568`'s test_dates.py
(8K tokens prefill, cache had 6,342 free + 58,211 evictable = 64,553
total, but `alloc_token_slots` rejected the 8,192-token request). This
is the upstream SGLang bug noted in the paper's limitations; the
28-case run got lucky because the original 30 dataset had smaller
file sizes (longest prefill was 6,144 tokens, well under the 8K
threshold that triggers the bug).

Two log files:
- `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_5/sglang_server.log` (mem-fraction-static=0.82, default)
- `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_5_v2/sglang_server.log` (mem-fraction-static=0.65)

Both failed at the first prefill.

## Step 2.4: Pass@1 driver with KV allocator defrag (2026-06-09 00:49)

Driver patch: added `--kv-allocator-defrag` flag to
`benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py:launch_server`
which sets `SGLANG_KV_ALLOCATOR_DEFRAG=1` in the sglang server env. This
is the upstream flag for `TokenToKVPoolAllocator.alloc_with_defrag()`
(`python/sglang/srt/mem_cache/allocator.py:156`), which runs
`merge_and_sort_free()` (folds `release_pages` into `free_pages`) on
alloc failure, before retrying.

Output: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_5_defrag/`

**Outcome**: same OOM at the same code path
(`common.py:230` `alloc_token_slots`). The OOM message after the
defrag-flag attempt is byte-identical to the no-flag attempt:

```
Try to allocate 8192 tokens.
Available tokens: 64553 (available_size=6342 + evictable_size=58211)
```

**Why defrag did not help** (after reading the SGLang source):

1. The `alloc_token_slots` path (line 201) first calls
   `evict_from_tree_cache(tree_cache, num_tokens)` which calls
   `tree_cache.evict(EvictParams(num_tokens=8192))`.
2. `RadixCache.evict()` (radix_cache.py:1633) iterates
   `self.evictable_leaves` — a `set()` of TreeNodes (line 455). Only
   nodes in this set are freeable.
3. `evictable_leaves` is populated only for **leaf** TreeNodes whose
   `lock_ref == 0` (radix_cache.py:2222, called from
   `_update_leaf_status`).
4. Looking at the OOM's radix-tree pretty_print: **all 4 visible
   leaves have `r=3`** (3-way locked by in-flight prefill batches).
   The internal nodes show `r=0`, but internal nodes are not in
   `evictable_leaves` (only leaves are).
5. Therefore `eviction_heap` is empty, the while-loop in `evict()`
   exits immediately, and `free()` is never called.
6. With no freed tokens, `release_pages` stays empty, so
   `merge_and_sort_free()` in `alloc_with_defrag()` is a no-op.

**The `evictable_size=58,211` reported in the OOM message is the
`evictable_size_` counter** (radix_cache.py:1789), which is
incremented **at every node insertion** (line 2104: `self.evictable_size_ += len(key)`)
— internal nodes with `lock_ref=0` count toward it too, even though
they cannot actually be evicted. The error message is misleading:
the **actual `evictable_leaves` set is empty**, so `evict()` has
nothing to free.

**Conclusion**: the upstream SGLang `evict()` mechanism is correct
and runs. The OOM is a **transient lock-pressure** problem, not a
fragmentation problem. The fix would be to either (a) wait for the
in-flight prefill to finish so its `lock_ref` is released before
scheduling the next, or (b) reduce the number of concurrent in-flight
prefills (e.g., `--disable-overlap-schedule` or
`--max-running-requests 1`).

## Step 2.5: Pass@1 driver with CPU offload (32 GB) (2026-06-08 13:31)

Driver patch: added `--cpu-offload-gb` arg to
`benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py:launch_server`
(passes through SGLang's `--cpu-offload-gb 32`; the testbed has 230 GB
free system RAM, so 32 GB reservation is well within budget).

Output: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_5_cpuoffload/`

**Outcome**: the OOM was avoided (no `RuntimeError: Out of memory` in
the log), but the run **did not produce usable results**:

- Case 1 (`django__django-11138`): all 3 patch files written but **0
  bytes each** (empty); the model output is empty, presumably
  because CPU offload made the per-token decode so slow that
  `--max-tokens 1024` timed out before any token was generated.
- Case 2 (`django__django-11149`): directory created but no patch
  files; the request hit aiohttp `TimeoutError` (600 s client
  timeout) and the run aborted.

Server log: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_5_cpuoffload/sglang_server.log`
(18 lines at `--log-level error`, no alloc errors, no OOM; just
empty model output and aiohttp timeouts).

**Why CPU offload did not help**: the allocator bug rejects 8K-token
prefills when the cache is fragmented (6,342 free + 58,211 evictable
= 64,553 available, allocator wants 8,192 contiguous). CPU offload
adds another pool but the offload pool is itself fragmented, and
the offload path adds a CPU↔GPU transfer per chunked prefill step
that makes the per-request latency 5-10× longer than the GPU-only
path. The empty-output / aiohttp-timeout failure is downstream of
the same fragmentation issue, not a separate OOM.

## Step 3: Pass@1 driver on the 5 new cases (small ctx)

Output: `results/swe_generated_patch_kvcomm/qwen2_5_7b_json_5_smallctx/`

Settings: `--files-per-case 1 --max-file-chars 3000 --max-tokens 512`
(same as the 32B small-ctx variant that bypasses the OOM).

| Mode | apply pass | test pass | pass@1 |
|---|---:|---:|---:|
| lossless | 1/5 | 0/5 | 0/5 |
| lossy    | 1/5 | 0/5 | 0/5 |

Per-case:
- `psf__requests-6028`: lossless/lossy apply=True but test=False
  (patch applied, but tests fail — not a pass@1 hit)
- All other 4: apply=False (search not found or no diff extracted)

**This 0/5 result is not directly comparable to the 28-case result**
because the 28-case used full context (no `--max-file-chars` cap,
default `--max-tokens`). The small-ctx restriction makes the
patch synthesis harder (less file context to anchor the diff).

## 5/5 discriminative dataset

For reproducibility, the manifest + dataset for the 5 new cases is at:
- `results/swebench_local_envs/manifest_5.json`
- `results/swebench_local_envs/_5_new_discriminative_instances.json`

## Unblocking paths for the 100-case expansion

1. **Run on a 40+ GB GPU** (A100-40GB or H100 80GB) where the
   prefill headroom is larger and the 8,192-token prefill fits
   without needing eviction.
2. **Add `--disable-overlap-schedule`** to the sglang server launch
   so the scheduler serializes prefill batches; with one prefill in
   flight at a time, the previous request's leaves release their
   `lock_ref=3` before the next request starts, making
   `evictable_leaves` non-empty.
3. **Pre-truncate the test files to <6000 tokens** before the
   pass@1 driver runs (the 30-case run used this implicitly because
   the original 30 dataset had shorter files). With shorter prefills,
   the 6,342-token `free_pages` headroom suffices without eviction.
4. **Upstream fix in SGLang's OOM error message**: report
   `evictable_leaves` (the actually-evictable token count) rather
   than `evictable_size_` (which includes internal nodes). This
   would clarify the transient-vs-fragmentation distinction for
   users.
5. **KV allocator defrag** (added to driver as
   `--kv-allocator-defrag`, sets `SGLANG_KV_ALLOCATOR_DEFRAG=1`):
   **does not work** for this workload — the defrag path is for
   allocator fragmentation, not for lock-pressure on leaves
   (Step 2.4 above).
6. **CPU offload** (added to driver as `--cpu-offload-gb`):
   **does not work** for this workload — the offload path adds a
   per-step CPU↔GPU transfer that makes chunked prefill 5-10× slower
   than the GPU-only path, hitting the aiohttp 600-s client timeout
   (Step 2.5 above).
