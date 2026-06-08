# Repo-Level Exact Code-Base KV Reuse Report

Model: `/home/gfy/models/Qwen2.5-3B-Instruct`
Generated from: `results/real_codebase_exact_reuse/repo_dataset_combined_summary.json`

Experiment order: cold lossless baseline without anchor metadata, then planner warmup with anchor metadata, then lossy KVCOMM reuse.

## Dataset

| Case | Repo | Files | Total lines |
|---|---|---:|---:|
| `astropy__astropy-12907` | `astropy__astropy` | 3 | 2598 |
| `django__django-10097` | `django__django` | 3 | 2210 |
| `matplotlib__matplotlib-13989` | `matplotlib__matplotlib` | 3 | 2713 |

## Aggregate Results

| Metric | Value |
|---|---:|
| HF avg reusable segment length | 6519.8 tokens |
| HF layer-24 key cosine avg/min | 0.998970 / 0.998675 |
| HF layer-24 value cosine avg/min | 0.995992 / 0.994523 |
| sglang avg speedup | 1.544x |
| sglang cached tokens, lossless -> lossy | 29.7 -> 9860.3 |
| Output exact-match rate | 50.0% |
| Output token F1 avg | 0.7847 |

## sglang Exact-Reuse Runs

| Case | Agent | cached lossless | cached lossy | speedup | token F1 | Match reason | Matched content |
|---|---|---:|---:|---:|---:|---|---|
| `astropy__astropy-12907` | implementer | 0 | 12967 | 1.768x | 1.0000 | `exact_code_content_signature` | `0fbd5418727e` |
| `astropy__astropy-12907` | debugger | 18 | 7295 | 1.366x | 0.4200 | `exact_code_content_signature` | `b0abee69dc54` |
| `django__django-10097` | implementer | 40 | 12312 | 1.686x | 1.0000 | `exact_code_content_signature` | `2ecf197e9787` |
| `django__django-10097` | debugger | 40 | 6221 | 1.323x | 0.3579 | `exact_code_content_signature` | `474ee16520f1` |
| `matplotlib__matplotlib-13989` | implementer | 40 | 13594 | 1.779x | 0.9302 | `exact_code_content_signature` | `1a1015da3fe1` |
| `matplotlib__matplotlib-13989` | debugger | 40 | 6773 | 1.340x | 1.0000 | `exact_code_content_signature` | `caeaeafb5704` |

## Accuracy Risk Cases

| Case | Agent | token F1 | speedup | Notes |
|---|---|---:|---:|---|
| `astropy__astropy-12907` | debugger | 0.4200 | 1.366x | Lossy output diverged under deterministic decoding; needs task-level pass/fail validation. |
| `django__django-10097` | debugger | 0.3579 | 1.323x | Lossy output diverged under deterministic decoding; needs task-level pass/fail validation. |

## Interpretation

- Exact code-content signatures are the reuse gate; AST/anchor fields only locate code-base segments.
- RoPE delta gives high key cosine on real repo files, but values and later-layer keys still reflect upstream-context differences.
- Cached-token gains are large on multi-file repo prompts; output stability varies, so final accuracy claims should use SWE-bench-style pass/fail or patch-level validation rather than token overlap alone.

