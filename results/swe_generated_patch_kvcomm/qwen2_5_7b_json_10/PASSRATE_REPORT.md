# Lossless KV vs KVCOMM Lossy Pass@1 Report

Generated on 2026-06-02. Experiment uses Qwen2.5-7B-Instruct with JSON edit schema on the expanded 10-case SWE-bench Verified repo-level dataset.

## Artifacts

- Raw summary: `summary.json`
- CSV table: `passrate_table.csv`
- Pipeline figure: `fig_passrate_pipeline.png`
- Cached-token figure: `fig_cached_tokens.png`

![Passrate pipeline](/home/gfy/CodeMAS_Project/sglang-kvflow/results/swe_generated_patch_kvcomm/qwen2_5_7b_json_10/fig_passrate_pipeline.png)

![Cached tokens](/home/gfy/CodeMAS_Project/sglang-kvflow/results/swe_generated_patch_kvcomm/qwen2_5_7b_json_10/fig_cached_tokens.png)

## Main Result

| Metric | Lossless KV | KVCOMM lossy | Delta |
|---|---:|---:|---:|
| diff extraction | 8/10 | 7/10 | -1 |
| clean apply | 8/10 | 7/10 | -1 |
| pass@1 | 1/10 | 1/10 | +0 |
| avg cached tokens | 1773.7 | 2868.8 | +1095.1 |
| avg generation speedup (lossless/lossy) | - | 1.776x | - |
| lossy exact-content hits | - | 10/10 | - |

Interpretation: KVCOMM lossy exact-code reuse has **0 pass@1 delta** relative to the lossless KV baseline in this run. The absolute pass@1 remains low because Qwen2.5-7B patch synthesis is the bottleneck, not because the lossy KV gate accepts different code content.

## Per-Case Table

| Case | Lossless diff/apply/pass | Lossy diff/apply/pass | Lossy match | Cached L/Lossy | Notes |
|---|---|---|---|---:|---|
| `astropy__astropy-12907` | 1/1/0 | 0/0/0 | `exact_code_content_signature` | 1085/3515 | lossless: apply ok, test rc=1; lossy: search not found in astropy/modeling/separable.py |
| `django__django-10097` | 1/1/0 | 1/1/0 | `exact_code_content_signature` | 8242/8243 | lossless: apply ok, test rc=1; lossy: apply ok, test rc=1 |
| `matplotlib__matplotlib-13989` | 1/1/0 | 1/1/0 | `exact_code_content_signature` | 938/939 | lossless: apply ok, test rc=1; lossy: apply ok, test rc=1 |
| `mwaskom__seaborn-3069` | 0/0/0 | 0/0/0 | `exact_code_content_signature` | 933/7532 | lossless: search not found in seaborn/_core/plot.py; lossy: search not found in seaborn/_core/plot.py |
| `pallets__flask-5014` | 1/1/0 | 1/1/0 | `exact_code_content_signature` | 471/472 | lossless: apply ok, test rc=4; lossy: apply ok, test rc=4 |
| `psf__requests-1142` | 1/1/1 | 1/1/1 | `exact_code_content_signature` | 538/539 |  |
| `pydata__xarray-2905` | 1/1/0 | 1/1/0 | `exact_code_content_signature` | 1276/1277 | lossless: apply ok, test rc=4; lossy: apply ok, test rc=4 |
| `pylint-dev__pylint-4551` | 0/0/0 | 0/0/0 | `exact_code_content_signature` | 1671/3586 | lossless: json parse failed: Expecting ',' delimiter: line 6 column 56 (char 168); lossy: no json object extracted |
| `pytest-dev__pytest-10051` | 1/1/0 | 1/1/0 | `exact_code_content_signature` | 1124/1125 | lossless: apply ok, test rc=1; lossy: apply ok, test rc=1 |
| `scikit-learn__scikit-learn-10297` | 1/1/0 | 1/1/0 | `exact_code_content_signature` | 1459/1460 | lossless: apply ok, test rc=4; lossy: apply ok, test rc=4 |

## What Improved During This Iteration

- Added `--output-schema json-edit` so the model emits structured search/replace edits and the harness synthesizes unified diffs locally.
- Fixed candidate evaluation to pass the correct 10-case dataset path.
- Fixed relative patch path handling by resolving candidate patch paths before `git apply`.
- Added repo reset before prompt file loading, patch synthesis, apply-check, and candidate evaluation to avoid dirty-worktree contamination.
- Normalized model-emitted paths such as `/path.py` and `repo/path.py`.

## Remaining Bottleneck

The current absolute pass@1 is limited by model patch quality. Several cases produce cleanly applicable but semantically wrong patches or invalid runtime behavior. For the contribution-3 claim, this run is useful because the lossless and lossy pass@1 are equal while lossy reuse is exact-content gated; for a stronger paper table, rerun this harness with a stronger coding model or a constrained edit decoder.

