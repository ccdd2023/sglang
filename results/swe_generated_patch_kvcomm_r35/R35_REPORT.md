# R35 FAIL_TO_PASS Verification: Blocked on Env (2026-07-07)

**Goal**: Use 3 conda envs (astropy/django/matplotlib) to verify model patches
pass FAIL_TO_PASS / PASS_TO_PASS tests. R33 was unable to do this because of
broken env state.

## Status: BLOCKED

R35 cannot run end-to-end without:

1. **Network access for `git fetch`** — `setup_swebench_local_env.py` always
   runs `git fetch --all --tags` (line 122). On this machine the fetch fails:
   ```
   fatal: unable to access 'https://github.com/astropy/astropy.git/':
   gnutls_handshake() failed: The TLS connection was non-properly terminated.
   ```
   The 3 local repo checkouts are already at the correct base_commit
   (`d16bfe05…` for astropy), so the fetch is logically unnecessary, but the
   script does not check `base_commit` first.

2. **Conda envs for 3 instances** — `/home/gfy/.conda/envs/` contains only
   `sglang-kvflow` and `sglang-kvflow-lmcache`. The
   `swe_astropy_astropy_12907_gold` / `swe_django_django_10097_gold` /
   `swe_matplotlib_matplotlib_13989_gold` envs named in
   `local_env_report.md` (2026-06-01) are gone.

3. **Per-instance Python+deps rebuild** — each env needs:
   - Python 3.10 + `setuptools<70` (per env_report)
   - `pip install -e .[test]` for the repo at the right base_commit
   - ~10-15 min per env × 3 = 30-45 min wall clock
   - This is **out of session scope** without explicit user sign-off.

## What was tried

- `python benchmark/multi_workflow/setup_swebench_local_env.py
  --instance-id astropy__astropy-12907 --mode candidate
  --candidate-patch results/swebench_local_envs/patches/astropy__astropy-12907/gold.patch`
  → failed at `ensure_repo()` step (git fetch error).

## What R35 would have answered (deferred to future work)

If R35 could run, the headline table would be:

| Instance | Mode | apply_rc | candidate_test_pass | FAIL_TO_PASS | PASS_TO_PASS |
|---|---|---|---|---|---|
| astropy | lossless | 128 | (env) | (env) | (env) |
| astropy | lossy | 128 | (env) | (env) | (env) |
| django | lossless | **0** | (env) | (env) | (env) |
| django | lossy | **0** | (env) | (env) | (env) |
| matplotlib | lossless | 128 | (env) | (env) | (env) |
| matplotlib | lossy | 1 | (env) | (env) | (env) |

We know `django` patches apply cleanly and `astropy` patches are structurally
broken (truncated), so the **expected R35 result** was:
- django: candidate test should fail (model added wrong ValidationError logic) → documents "model understood task but picked wrong fix"
- astropy: candidate test would have rejected (apply failed before test even ran)
- matplotlib: candidate test depends on whether lossless's `_make_inset_locator` change vs lossy_prefetch's `hist` change passes the test

Without running, R35 cannot confirm or refute the **lossy-reuse-doesn't-degrade-accuracy**
claim in the SWE-bench sense (we have it only for the verdict task from R26/R27).

## Mitigation options (not pursued in this session)

| Option | Effort | Notes |
|---|---|---|
| Add `--skip-fetch` flag to setup_swebench_local_env.py | 5 min | Reasonable next session, but requires another harness change |
| Patch setup_swebench_local_env.py inline to skip fetch when env var set | 5 min | Same as above but env-var-gated (default OFF preserves old behavior) |
| Rebuild conda envs from scratch | 30-45 min | Out of session scope; needs user sign-off |
| Run tests directly without the SWE-bench harness (`pytest tests/`) | 10 min | Bypass env setup; quick check that at least the test files exist |

## Conclusion

R35 is documented as **blocked**. The honest finding is that this session
cannot fully verify FAIL_TO_PASS / PASS_TO_PASS on real model patches
without env rebuild. R34 + R36 + R37 still provide:

- ✅ First-ever `codeaware_reused_tokens > 0` (R34)
- ✅ Defensive parser truncation (R36) — added but didn't help R33's truncated patch (the truncation is structural, not excess)
- ✅ First-hunk vs gold weak signal (R37) — works as expected

These three together give a much stronger answer than R33 alone, even without R35.