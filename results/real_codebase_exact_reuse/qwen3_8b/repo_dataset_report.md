# Repo-Level Exact Code-Base KV Reuse Report

Model: `/home/gfy/models/Qwen3-8B`
Generated from: `results/real_codebase_exact_reuse/qwen3_8b/combined_summary.json`

Experiment order: cold lossless baseline without anchor metadata, then planner warmup with anchor metadata, then lossy KVCOMM reuse.

## Dataset

| Case | Repo | Files | Total lines |
|---|---|---:|---:|
| `astropy__astropy-12907` | `astropy__astropy` | 2 | 1392 |
| `django__django-10097` | `django__django` | 2 | 1295 |
| `matplotlib__matplotlib-13989` | `matplotlib__matplotlib` | 2 | 1632 |

## Aggregate Results

| Metric | Value |
|---|---:|
| HF avg reusable segment length | 5571.2 tokens |
| HF layer-35 key cosine avg/min | 0.998766 / 0.998292 |
| HF layer-35 value cosine avg/min | 0.990817 / 0.988023 |
| sglang avg speedup | 1.123x |
| sglang cached tokens, lossless -> lossy | 29.7 -> 2760.3 |
| Output exact-match rate | 100.0% |
| Output token F1 avg | 1.0000 |

## sglang Exact-Reuse Runs

| Case | Agent | cached lossless | cached lossy | speedup | token F1 | Match reason | Matched content |
|---|---|---:|---:|---:|---:|---|---|
| `astropy__astropy-12907` | implementer | 0 | 8192 | 1.364x | 1.0000 | `exact_code_content_signature` | `1a6bee2f986d` |
| `astropy__astropy-12907` | debugger | 18 | 18 | 0.999x | 1.0000 | `exact_code_content_signature` | `339825b3a574` |
| `django__django-10097` | implementer | 40 | 40 | 0.994x | 1.0000 | `exact_code_content_signature` | `44f1f1ad9c2d` |
| `django__django-10097` | debugger | 40 | 40 | 1.002x | 1.0000 | `exact_code_content_signature` | `4a13c0170d6b` |
| `matplotlib__matplotlib-13989` | implementer | 40 | 8232 | 1.382x | 1.0000 | `exact_code_content_signature` | `c75984327088` |
| `matplotlib__matplotlib-13989` | debugger | 40 | 40 | 0.994x | 1.0000 | `exact_code_content_signature` | `db0b97c00e77` |

## Accuracy Risk Cases

No output pairs had token F1 below 0.6.

## Interpretation

- Exact code-content signatures are the reuse gate; AST/anchor fields only locate code-base segments.
- RoPE delta gives high key cosine on real repo files, but values and later-layer keys still reflect upstream-context differences.
- Cached-token gains are large on multi-file repo prompts; output stability varies, so final accuracy claims should use SWE-bench-style pass/fail or patch-level validation rather than token overlap alone.

