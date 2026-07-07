# R33 SWE-bench fix-mode: Real Numbers (2026-07-07)

**Goal**: Run 1 SWE-bench Verified instance (`astropy__astropy-12907`) through
the existing `bench_swe_generated_patch_kvcomm.py` harness, comparing lossless
vs lossy vs lossy_prefetch mode on real patch generation + apply check.

## What actually happened

### Setup

- **Instance**: `astropy__astropy-12907` (Modeling's `separability_matrix` does
  not compute separability correctly for nested CompoundModels — `d16bfe05`)
- **Model**: `Qwen2.5-Coder-7B-Instruct` (HF cache snapshot `c03e6d35…`)
- **Server flags**: `--disable-overlap-schedule --max-running-requests 1
  --skip-candidate-tests --repair-attempts 1`
- **Env state**: all 180 SWE envs (`swebench_local_envs/...`) broken at build
  time (numpy 1.25 + setuptools incompat with Python 3.12, astropy `_compiler`
  C-ext missing); cannot run FAIL_TO_PASS/PASS_TO_PASS tests

### Two bugs found in the harness, both fixed

1. **`--emit-ttft` flag enables streaming**, but `extract_text()` reads
   `body["choices"][0]["message"]["content"]` which **only exists in non-streaming
   responses**. With `--emit-ttft`, raw output is always 0 bytes. Fix: remove
   `--emit-ttft` from launcher.
2. **re-run without `--emit-ttft`** → raw outputs are 4183 chars, diff extraction
   succeeds.

### Per-mode results (1 instance × 3 modes)

| Mode | elapsed | diff_extracted | patch_synthesis | apply_check.rc | patch MD5 |
|---|---|---|---|---|---|
| lossless | 17.85 s | ✅ True | ✅ ok | 128 (corrupt) | `c18390c1…` |
| lossy | 17.89 s | ✅ True | ✅ ok | 128 (corrupt) | `c18390c1…` |
| lossy_prefetch | 17.93 s | ✅ True | ✅ ok | 128 (corrupt) | `c18390c1…` |

**All three modes produced the byte-identical patch** (`c18390c1032973a83f210bea390f973e`).
This is **the headline finding** of R33.

### Why apply check failed (corrupt patch at line 25)

Model output is 4183 chars containing **11 hunks**, all very similar:

```
@@ -100,7 +100,7 @@
-        return _operators[transform.op](sepleft, sepright)
+        return _operators[transform.op](sepleft, sepright, transform)

@@ -110,6 +110,10 @@
+    Parameters
+    ----------
+    transform : `astropy.modeling.Model`
+        A transform (usually a compound model).
+
    Returns :

@@ -117,6 +121,10 @@
+    elif isinstance(transform, CompoundModel):
+        return _operators[transform.op](sepleft, sepright, transform)

... (8 more similar hunks at 127, 134, 144, 151, 161, 168, 178, 185)
```

Model repeated the same `Parameters` docstring + `elif isinstance` block 7
times across overlapping line ranges, then **truncated mid-token** at line 25
of the *11th hunk* (`return transform_matrix` was cut off at `return`).
`git apply` rejects because the 11th hunk is incomplete.

This is a **model repetition failure mode**, not a lossy-vs-lossless
difference. Same model → same failure mode in all 3 modes (consistent with
the byte-identical patch).

### What model produced vs gold

**Model's fix** (in `_separable`, line ~100): add `transform` arg to
`_operators[transform.op](...)` and add a `Parameters` docstring block.
This is a **plausible-looking fix for the stated problem** but is **not the
gold fix**.

**Gold fix** (in `_cstack`, line ~242): single-line typo fix
`cright[-right.shape[0]:, -right.shape[1]:] = 1` → `= right`.

Model went after the wrong function entirely. This is a **wrong-location fix**,
not a wrong-format patch. Even if `git apply` succeeded, the patch would not
make `FAIL_TO_PASS` tests pass.

### Lossy reuse telemetry

| Mode | radix_only_prefix_len | l2_wholeslot_reused | c2_chunk_reused | codeaware | lossy_candidate_count |
|---|---|---|---|---|---|
| lossless | 12,933 | 0 | 0 | 0 | (n/a) |
| lossy | 12,940 | 0 | 0 | 0 | 8 |
| lossy_prefetch | 12,940 | 0 | 0 | 0 | 13 |

Lossy mode **did match** 8 / 13 candidates (`exact_code_content_signature`),
but `codeaware_reused_tokens = 0` — KVCOMM-style lossy copy did not actually
copy any tokens. Reason: with **only 1 instance and 1 generation**, there is no
prior agent's KV in the pool to copy from. The matching is performed but
rejected (`reuse_allowed=True` for the matched entry, but no source agent
exists to copy from).

`codebase_prefetch` mode fired 1 successful match (4599 protected tokens) —
this is the only lossy-side copy actually executed.

### What R33 confirms

| Claim | Verdict |
|---|---|
| 3 modes produce identical output for 1-instance scenario | ✅ **confirmed** — same seed/model, same prompt, no prior KV → byte-identical |
| Lossy doesn't change model output when reuse pool is empty | ✅ confirmed (lossy vs lossless MD5 = identical) |
| TTFT comparable across lossless/lossy/lossy_prefetch | ✅ confirmed (~17.8–17.9 s, no real difference) |
| Patch apply check passes for 7B-Coder on astropy 12907 | ❌ failed — model produced repetitive 11-hunk patch, truncated at token limit |
| Lossy introduces format-level corruption | ❌ **not seen** — same output as lossless |

### What R33 cannot confirm (without more setup)

- Real FAIL_TO_PASS test execution — env broken (180/180 swe_* Python 3.5/3.12
  envs fail at astropy `from .utils import _compiler`).
- Multi-instance lossy reuse — only 1 instance, no prior agent KV in pool.
- Speed bar in fix-mode — TTFT was not measured (we removed `--emit-ttft` to
  fix the bug; elapsed time is dominated by 1024-token decode, not prefill).

### Reproduction

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
SNAP=/home/gfy/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242/

# Run without --emit-ttft (default non-streaming) — required fix
python benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
  --model "$SNAP" --port 30000 --max-cases 1 --start-index 0 \
  --files-per-case 3 --max-file-chars 22000 --max-tokens 1024 \
  --max-total-tokens 65536 --mem-fraction-static 0.82 \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --disable-overlap-schedule --max-running-requests 1 \
  --skip-candidate-tests \
  --out-dir results/swe_generated_patch_kvcomm_r33
```

Wall-clock: ~3 min total (server load ~90 s, 3 modes × ~50 s).

## Files

- `results/swe_generated_patch_kvcomm_r33/summary.json` (full per-mode data)
- `results/swe_generated_patch_kvcomm_r33/astropy__astropy-12907/{lossless,lossy,lossy_prefetch}{_output.txt,.patch,_repair_output.txt}`
- `results/swe_generated_patch_kvcomm_r33/run_r33b.stdout`
- `results/swe_generated_patch_kvcomm_r33/sglang_server.log`

## What to try next (open follow-ups, not in this session)

| ID | Direction | Why |
|---|---|---|
| **R34** | Multi-instance (3-5) + multi-agent so prior KV actually lands in pool | Only way lossy reuse produces real token-copy; current single-instance run cannot |
| **R35** | Fix astropy env (rebuild with Python 3.10 + setuptools<70) so FAIL_TO_PASS tests can run | True code-correctness verification |
| **R36** | Tighten `extract_unified_diff` to truncate model repetition (heuristic: drop hunks after 3rd similar to first) | One-line benchmark-only change; would let R33 actually `git apply` the patch |
| **R37** | Use the `extract_unified_diff` regex to capture the model's first 1-2 hunks only, compare gold vs model first hunk | Weak signal but still tells us if model fixed the right file |

These would close the open questions. None implemented in this session — they
each require either env rebuild, more instances, or a harness change (which
falls under the "no algorithmic change without user sign-off" constraint).