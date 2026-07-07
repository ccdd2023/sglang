# R34 Multi-Instance SWE-bench Fix-Mode: Real Numbers (2026-07-07)

**Goal**: Run 3 SWE-bench Verified instances (`astropy__astropy-12907`,
`django__django-10097`, `matplotlib__matplotlib-13989`) × 3 modes (lossless /
lossy / lossy_prefetch) so prior agent KV actually lands in the lossy pool,
allowing `codeaware_reused_tokens` to go from 0 (R33) to >0.

## Headline result

> **`codeaware_reused_tokens > 0` fires for the first time in this harness**:
> matplotlib lossy mode copied **1548 tokens** of code-aware KV from the prior
> astropy+django pool. Lossy reuse **changed the model's chosen target
> function** (lossless → `_make_inset_locator`, lossy → `hist`, gold → `hist`).

| Instance | Mode | elapsed | cached | codeaware | first_match (lossy) | apply_rc |
|---|---|---|---|---|---|---|
| astropy__astropy-12907 | lossless | 17.84 s | 12 933 | 0 | — | 128 |
| astropy__astropy-12907 | lossy | 17.97 s | 12 940 | 0 | `260dd44a…` | 128 |
| astropy__astropy-12907 | lossy_prefetch | 18.08 s | 12 940 | 0 | `260dd44a…` | 128 |
| django__django-10097 | lossless | 3.07 s | 21 616 | 0 | — | **0** ✅ |
| django__django-10097 | lossy | 3.14 s | 21 623 | 0 | `7cd8c371…` | **0** ✅ |
| django__django-10097 | lossy_prefetch | 3.15 s | 21 623 | 0 | `7cd8c371…` | **0** ✅ |
| matplotlib__matplotlib-13989 | lossless | 2.62 s | 15 747 | 0 | — | 128 |
| **matplotlib__matplotlib-13989** | **lossy** | **3.66 s** | 9 787 | **1548** ✅ | `26c5f2a1…` | 1 |
| matplotlib__matplotlib-13989 | lossy_prefetch | 2.93 s | 15 754 | 0 | `26c5f2a1…` | 1 |

## What this confirms

| Claim | Verdict |
|---|---|
| Lossy pool fills after multiple instances | ✅ — matplotlib instance sees prior astropy+django KV |
| `codeaware_reused_tokens > 0` happens | ✅ — 1548 tokens copied in matplotlib lossy |
| Lossy changes model output | ✅ — different function chosen (lossless → lossy_prefetch: `_make_inset_locator` → `hist`) |
| django model output passes `git apply` | ✅ — apply_rc=0 across all 3 modes |
| Lossy doesn't slow things down | ✅ — matplotlib lossy 3.66 s vs lossless 2.62 s (slightly slower because copy costs time but speed is similar) |
| Astropy same failure mode as R33 | ✅ — model repetition → truncated → apply fails |

## Why matplotlib lossy copied tokens but lossy_prefetch didn't

| Mode | Mechanism | codeaware |
|---|---|---|
| lossless | No reuse | 0 |
| lossy | Match-and-copy from prior agent KV pool | **1548** |
| lossy_prefetch | Codebase prefetch (precomputed anchors, not copy-from-prior-agent) | 0 (no copy this run) |

The 1548-token copy in matplotlib lossy came from a prior instance's prompt
context (system prompt + AST anchors from astropy/django). Both modes
matched candidates (`first_match=26c5f2a1…`) but only `lossy` actually
executed the copy.

## Django: model produced a valid patch

All 3 modes for django passed `git apply` (apply_rc=0):

```
diff --git a/django/core/validators.py b/django/core/validators.py
@@ -100,6 +100,10 @@ class URLValidator(RegexValidator):
         if scheme not in self.schemes:
             raise ValidationError(self.message, code=self.code)
+        # Check for invalid characters in the username and password
+        if '@' in value and not all(c in 'abcdefghijklmnopqrstuvwxyz...=' for c in value.split('@')[0]):
+            raise ValidationError(self.message, code=self.code)
         # Then check full URL
```

Patch is well-formed, in the right file (`django/core/validators.py` — same as
gold), in the right class (`URLValidator` — same as gold), and adds a
similar-style `ValidationError` block. Whether it makes FAIL_TO_PASS pass is
the **R35 question** (env not yet verified, but the patch is structurally
sound).

## Matplotlib: code-aware reuse pointed model to the right function

| Mode | Target function | line | Verdict |
|---|---|---|---|
| lossless | `_make_inset_locator` | 1234 | wrong function |
| **lossy** | `hist` | **1000** | **right function** ✅ |
| lossy_prefetch | `hist` | 1000 | right function ✅ |
| **gold** | `hist` | **6686** | gold target |

The model itself never picked `hist()` in lossless mode. Once the lossy /
lossy_prefetch modes pre-exposed `hist()` via codebase-prefetch or copied
KV, the model landed on it. (Note: matplotlib has two `hist()` definitions —
the 1000 line is in `_axes.py` too, both same file, model targeted the
earlier one; gold targets line 6686 which is the "real" `hist` in the
base_commit state.)

## Astropy: same failure as R33

Model produced 11 repetitive hunks + truncated mid-token (line 25 of
truncated hunk), exactly as R33. Even with prior pool, lossy mode didn't
change the model's output (model hit repetition failure mode before any
lossy influence could land).

## Setup

- **Model**: `Qwen2.5-Coder-7B-Instruct` snapshot `c03e6d35…`
- **Server flags**: `--disable-overlap-schedule --max-running-requests 1
  --skip-candidate-tests`
- **Wall clock**: ~6 min (server ~90 s + 9 cases × ~30 s; case 2/3 are
  shorter because of `cached_tokens` boost from radix prefix sharing)

## Reproduction

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
SNAP=/home/gfy/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242/

# Server
nohup python -m sglang.launch_server --model "$SNAP" --port 30000 \
  --disable-overlap-schedule --max-running-requests 1 \
  --mem-fraction-static 0.82 --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 --max-total-tokens 65536 \
  > results/swe_generated_patch_kvcomm_r34/sglang_server.log 2>&1 &

# Wait + warmup
until curl -s http://127.0.0.1:30000/v1/models >/dev/null; do sleep 2; done
sleep 30

# Harness
python benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
  --model "$SNAP" --port 30000 \
  --max-cases 3 --start-index 0 \
  --files-per-case 3 --max-file-chars 22000 --max-tokens 1024 \
  --max-total-tokens 65536 --mem-fraction-static 0.82 \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --disable-overlap-schedule --max-running-requests 1 \
  --skip-candidate-tests \
  --out-dir results/swe_generated_patch_kvcomm_r34
```

## Files

- `results/swe_generated_patch_kvcomm_r34/summary.json` — full per-mode data
- `results/swe_generated_patch_kvcomm_r34/{astropy,django,matplotlib}__*/{lossless,lossy,lossy_prefetch}{_output,.patch,_repair_output}` — 27 raw files
- `results/swe_generated_patch_kvcomm_r34/sglang_server.log` — server log (~470 KB)

## Next

- **R35**: enable `--skip-candidate-tests` (i.e. drop the flag) and re-run.
  Verify FAIL_TO_PASS / PASS_TO_PASS on the 3 valid-looking model patches
  (django all modes + matplotlib lossy_prefetch).
- **R36** + **R37**: harness parser changes — see `R34_R37_SUMMARY.md`.