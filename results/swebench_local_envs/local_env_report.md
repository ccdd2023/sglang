# SWE-bench Local Repo Environment Report

Generated on: 2026-06-01

## Purpose

This report records the local, non-Docker setup for the three real repo-level SWE-bench Verified cases used by the KVCOMM exact code-base reuse benchmark.

Docker-based SWE-bench harness execution is blocked on this machine because the current user cannot access `/var/run/docker.sock`. The fallback path uses local git checkouts plus conda environments and the official SWE-bench per-repo version specs.

## Dataset

Source dataset: `results/repo_level_datasets/swe_verified_3_instances.json`

| Instance | Repo | Base commit | Version | Local env | Result |
|---|---|---|---|---|---|
| `astropy__astropy-12907` | `astropy/astropy` | `d16bfe05a744909de4b27f5875fe0d4ed41ce607` | `4.3` | `swe_astropy_astropy_12907_gold` | PASS |
| `django__django-10097` | `django/django` | `b9cf764be62e77b4777b3a75ec256f6209a57671` | `2.2` | `swe_django_django_10097_gold` | PASS |
| `matplotlib__matplotlib-13989` | `matplotlib/matplotlib` | `a3e2897bfaf9eaac1d6649da535c4e721c89fa69` | `3.0` | `swe_matplotlib_matplotlib_13989_gold` | PASS |

## Gold Test Results

| Instance | Mode | Target tests | Result | Notes |
|---|---|---:|---|---|
| `astropy__astropy-12907` | gold | 2 | `2 passed in 0.32s` | NumPy binary compatibility warning observed, target tests passed. |
| `django__django-10097` | gold | 363 | `Ran 363 tests ... OK (skipped=5)` | The discriminative target is `validators.tests.TestSimpleValidators`, matching the URLValidator patch. |
| `matplotlib__matplotlib-13989` | gold | 1 | `1 passed, 11 warnings in 1.70s` | Required local FreeType pin: `freetype=2.10.4`; test warns expected FreeType 2.6.1. |

## Base vs Gold Validation

| Instance | Base result | Gold result | Validation status |
|---|---|---|---|
| `astropy__astropy-12907` | 2 failed | 2 passed | PASS: fail-to-pass reproduced |
| `django__django-10097` | 363 ran, 6 failed, 5 skipped | 363 ran, 0 failed, 5 skipped | PASS: fail-to-pass reproduced on URLValidator tests |
| `matplotlib__matplotlib-13989` | 1 failed | 1 passed | PASS: fail-to-pass reproduced |

Important correction: the first five `FAIL_TO_PASS` labels in the Django metadata target `auth_tests` and pass on both base and gold. They are not discriminative for this instance. The patch modifies `django/core/validators.py`, so the validation target was changed to `validators.tests.TestSimpleValidators`, which directly exercises the added URL cases in `tests/validators/valid_urls.txt` and `tests/validators/invalid_urls.txt`.

Report JSON files:

| Instance | Base report | Gold report |
|---|---|---|
| `astropy__astropy-12907` | `results/swebench_local_envs/reports/astropy__astropy-12907/base_report.json` | `results/swebench_local_envs/reports/astropy__astropy-12907/gold_report.json` |
| `django__django-10097` | `results/swebench_local_envs/reports/django__django-10097/base_report.json` | `results/swebench_local_envs/reports/django__django-10097/gold_report.json` |
| `matplotlib__matplotlib-13989` | `results/swebench_local_envs/reports/matplotlib__matplotlib-13989/base_report.json` | `results/swebench_local_envs/reports/matplotlib__matplotlib-13989/gold_report.json` |

## Commands

```bash
/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/setup_swebench_local_env.py \
  --instance-id astropy__astropy-12907 \
  --mode gold \
  --timeout 1200

/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/setup_swebench_local_env.py \
  --instance-id matplotlib__matplotlib-13989 \
  --mode gold \
  --skip-pre-install \
  --timeout 1200

/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/setup_swebench_local_env.py \
  --instance-id django__django-10097 \
  --mode gold \
  --skip-pre-install \
  --test-target validators.tests.TestSimpleValidators \
  --timeout 1200
```

Matplotlib needed one manual native dependency correction before the final passing run:

```bash
/home/gfy/miniconda3/bin/conda install -y \
  -n swe_matplotlib_matplotlib_13989_gold \
  freetype=2.10.4 pkg-config qhull libpng
```

## Interpretation

- The selected real repo snapshots are executable locally after applying the SWE-bench test patches and gold patches.
- This validates that the repo-level KV reuse benchmark uses real, testable codebases rather than synthetic multi-file prompts.
- Base-vs-gold fail-to-pass is reproduced for all three cases.
- The current local validation is gold-patch pass/fail. It does not yet run generated model patches through the SWE-bench grader.
- Exact content signatures remain the reuse safety gate. AST or template metadata is only used to identify candidate code-base spans and route reuse hints.
