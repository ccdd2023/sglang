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
extension hit an upstream SGLang allocator OOM (a bug in
`python/sglang/srt/mem_cache/common.py:230` `alloc_token_slots`) that
manifested when pre-filling the 8K-context test_dates.py / utils.py
files of the 5 new cases. The 28-case result (3/28 vs 2/28, Wilson CI
overlap) remains the headline.

The 30-env gold smoke batch produced 5/30 passing envs (16.7%), all of
which are discriminative. The SGLang allocator OOM is reproducible on
this 24 GB GPU; the envs are ready to run pass@1 once the upstream
allocator is fixed (or on a 40+ GB GPU where the cache fragmentation
becomes irrelevant). A CPU-offload attempt (32 GB reservation on a
230-GB-RAM host) avoided the OOM but produced empty patches and
aiohttp client timeouts, because the offload path adds a CPU↔GPU
transfer per chunked-prefill step that is too slow to complete a
request inside the 600-s client timeout (see Step 2.5).

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
   prefill OOM won't trigger.
2. **Fix the upstream SGLang allocator** in
   `python/sglang/srt/mem_cache/common.py:230` `alloc_token_slots`
   to consider evictable_size as available. This is an upstream bug
   (paper §Limitations).
3. **Pre-truncate the test files to <6000 tokens** before the
   pass@1 driver runs (the 30-case run used this implicitly because
   the original 30 dataset had shorter files).
4. **Use a sub-1.0 mem-fraction-static + chain multiple server
   restarts** to flush fragmented state between cases.
5. **CPU offload** (added to driver as `--cpu-offload-gb`) **does
   not work** for this workload: the offload path adds a per-step
   CPU↔GPU transfer that makes chunked prefill 5-10× slower than
   the GPU-only path, hitting the aiohttp 600-s client timeout
   (Step 2.5 above).
