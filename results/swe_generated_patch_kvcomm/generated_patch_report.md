# SWE-bench Generated Patch KVCOMM Report

Generated on: 2026-06-01

## Purpose

This experiment connects KVCOMM exact code-base reuse to local SWE-bench-style candidate patch testing. It uses the three real repo cases from `results/repo_level_datasets/swe_verified_3_instances.json`.

Generation flow:

1. Planner warmup request inserts code-base anchors.
2. Lossless request generates a candidate unified diff without lossy anchors.
3. Lossy request generates a candidate unified diff with exact-content KVCOMM reuse enabled.
4. Each extracted diff is applied with `setup_swebench_local_env.py --mode candidate` and tested locally.

## Result Summary

| Instance | Lossless elapsed | Lossy elapsed | Speedup | Lossless cached | Lossy cached | Lossy match | Candidate result |
|---|---:|---:|---:|---:|---:|---|---|
| `astropy__astropy-12907` | 13818.44 ms | 4936.44 ms | 2.799x | 378 | 2808 | `exact_code_content_signature` | both patches invalid |
| `django__django-10097` | 2206.12 ms | 13528.69 ms | 0.163x | 636 | 637 | `exact_code_content_signature` | both patches invalid |
| `matplotlib__matplotlib-13989` | 2664.78 ms | 4171.15 ms | 0.639x | 541 | 542 | `exact_code_content_signature` | both patches invalid |

Aggregate candidate accuracy:

| Metric | Value |
|---|---:|
| Generated diffs extracted | 6 / 6 |
| Candidate patches applied cleanly | 0 / 6 |
| Candidate patches passed tests | 0 / 6 |
| Lossy exact-content reuse hits | 3 / 3 |

## Failure Modes

| Instance | Mode | Failure |
|---|---|---|
| `astropy__astropy-12907` | lossless | `git apply`: corrupt patch at line 81 |
| `astropy__astropy-12907` | lossy | `git apply`: corrupt patch at line 29 |
| `django__django-10097` | lossless | `git apply`: hunk does not apply at `django/core/validators.py:104` |
| `django__django-10097` | lossy | `git apply`: corrupt patch at line 42 |
| `matplotlib__matplotlib-13989` | lossless | `git apply`: hunk does not apply at `lib/matplotlib/axes/_axes.py:1044` |
| `matplotlib__matplotlib-13989` | lossy | `git apply`: hunk does not apply at `lib/matplotlib/axes/_axes.py:1044` |

## Interpretation

- The generated-patch evaluation harness works end to end: generation output is saved, diff is extracted, candidate patch is applied, and local SWE-bench tests are invoked.
- KVCOMM reuse is active in lossy mode and gated by exact code content, matching the contribution-3 safety contract.
- Qwen2.5-3B-Instruct is not strong enough under the current prompt to produce applyable SWE-bench patches for these repo-level cases.
- Current generated-patch accuracy is therefore blocked by patch synthesis quality, not by the local testing harness.
- For the paper, the current reliable accuracy evidence remains base-vs-gold pass/fail plus output similarity. Generated model patch pass/fail should be reported only after using a stronger coding model or a constrained patch-generation/editing format.

## Larger-Model Retests

Additional local retests were run on the same three SWE-bench Verified cases after the 3B run.

| Model | Diffs extracted | Cleanly applied | Passed tests | Lossy exact-code hits | Main observation |
|---|---:|---:|---:|---:|---|
| `Qwen2.5-7B-Instruct` | 3 / 6 | 0 / 3 | 0 / 6 | 3 / 3 | Some diffs extracted, but all extracted patches failed `git apply`. |
| `Qwen3-8B` | 0 / 6 | 0 / 0 | 0 / 6 | 2 / 3 | Output stayed in reasoning/analysis form and did not emit unified diffs. |

Qwen3-8B was used as the local/official Qwen3 text model; an official local `Qwen3-9B` checkpoint was not available in the current model directory. The generated-patch conclusion is unchanged: model-side patch synthesis is the current bottleneck, while KVCOMM reuse continues to be gated by exact code content when anchors are present.

Raw larger-model outputs:

- `results/swe_generated_patch_kvcomm/qwen2_5_7b/summary.json`
- `results/swe_generated_patch_kvcomm/qwen3_8b/summary.json`

Raw outputs and patches:

- `results/swe_generated_patch_kvcomm/summary.json`
- `results/swe_generated_patch_kvcomm/*/lossless_output.txt`
- `results/swe_generated_patch_kvcomm/*/lossy_output.txt`
- `results/swe_generated_patch_kvcomm/*/lossless.patch`
- `results/swe_generated_patch_kvcomm/*/lossy.patch`
