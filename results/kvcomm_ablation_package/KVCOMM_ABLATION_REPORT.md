# KVCOMM Contribution-3 Ablation Package

This report packages the experiments needed to defend Code-Base-Aware Lossy KV Reuse.

Key contract: AST/anchor metadata locates candidate code-base spans; actual reuse is gated by exact code content and token span matching.

## Artifact Index

- `ablation_summary.json`: all normalized results.
- `gate_safety_ablation.csv`: matcher/gate safety table.
- `rope_delta_ablation.csv`: HF numeric RoPE delta table.
- `layer_kv_summary.csv`: per-layer correct-delta K/V summary.
- `logit_alignment_ablation.csv`: next-token/logit behavior table.
- `length_gap_ablation.csv`: code-length and position-gap table.
- `template_guidance_ablation.csv`: template/priority/anchor logic table.
- `repo_exact_reuse_qwen3_8b.csv`: real repo-level serving results.
- `generated_patch_model_ablation.csv`: generated-patch model retests.
- `fig_*.png`: figures embedded below.

## 0. How to Judge KVCOMM Lossiness

| Metric | Good threshold used in this report | Role |
| --- | --- | --- |
| K cosine | `>0.99` strong alignment | RoPE/key-position correctness |
| V cosine | `>0.98` acceptable, `>0.99` strong | Context-value stability |
| top-1 agreement | `>95%` near-lossless behavior | Next-token behavior |
| SWE-bench pass@1 delta | `<=1-2 pct` vs lossless | Final task accuracy |

These thresholds are not universal constants from prior work; they are practical acceptance criteria that pair internal KV/logit similarity with downstream task accuracy.

## 1. Safety Gate Ablation

![Gate false accepts](fig_gate_false_accepts.png)

| policy | allowed | false_accepts | false_rejects |
| --- | --- | --- | --- |
| full_kvcomm | 1 | 0 | 0 |
| ast_only | 6 | 5 | 0 |
| span_overlap_only | 6 | 5 | 0 |
| content_only | 1 | 0 | 0 |
| token_text_exact | 1 | 0 | 0 |
| no_gate | 6 | 5 | 0 |

Result: full KVCOMM accepts only exact same code and has zero false accepts in the near-match suite. AST-only/span-only policies accept unsafe near matches, which demonstrates why AST is only a locator and not the reuse gate.

## 2. RoPE Delta Ablation

![RoPE delta cosine](fig_rope_delta_cosine.png)

- Correct delta mean K cosine: 0.999046
- Large wrong-delta mean K cosine (|error| >= 16): 0.949721
- Reusable segment length in this HF ablation: 1326 tokens

Result: the correct RoPE delta gives the closest key alignment. Deliberate delta errors reduce K cosine, especially in later layers, supporting the necessity of position correction for cross-position reuse.

### Per-Layer Correct-Delta KV Summary

| layer | correct_k_cosine | correct_k_mean_abs | correct_k_max_abs | correct_v_cosine | correct_v_mean_abs | correct_v_max_abs | wrong_delta_k_cosine_avg_abs_ge_16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0 | 0.002245 | 0.25 | 1.0 | 0.0 | 0.0 | 0.997066 |
| 18 | 0.997648 | 0.063306 | 11.667969 | 0.990206 | 0.027701 | 13.359375 | 0.935831 |
| 35 | 0.999491 | 0.040242 | 6.625 | 0.996363 | 0.147169 | 15.15625 | 0.916267 |

## 3. Logit-Level Behavior Ablation

![Logit KL](fig_logit_kl_by_position.png)

- Mean KL(B || A): 0.01471622
- Top-1 agreement: 96.9%
- Top-5 overlap agreement: 100.0%

Result: this measures behavior-level drift caused by moving the same code base to another prompt position. It complements K/V cosine and should be reported before task-level pass@1.

## 4. Code Length / Position Gap Ablation

![Length gap K cosine](fig_length_gap_k_cosine.png)

- Final-layer K cosine avg/min across tested lengths and gaps: 0.997866 / 0.992959

Result: this table identifies when the method is numerically safest and where longer gaps or longer code blocks begin to stress alignment.

## 5. Template Guidance Ablation

![Template guidance hits](fig_template_guidance_hits.png)

| variant | exact_content_hits | high_cached_token_hits | avg_speedup_observed_or_estimated | evidence_type |
| --- | --- | --- | --- | --- |
| full_template_priority_anchor | 6 | 2 | 1.12255 | serving_observed |
| no_template | 6 | 2 | 1.0825 | logic_ablation_from_observed_baseline |
| no_priority | 6 | 2 | 1.1025 | logic_ablation_from_observed_baseline |
| no_anchor | 0 | 0 | 1.0 | logic_ablation_from_observed_baseline |
| prefix_cache_only | 0 | 0 | 1.0 | logic_ablation_from_observed_baseline |

Result: anchor metadata is required for cross-position code-base reuse. Template and priority mainly affect scheduling/prefetch effectiveness, while no-anchor and prefix-cache-only cannot express the contribution-3 reuse contract.

## 6. Real Repo-Level Qwen3-8B Exact Reuse

![Cached tokens](fig_repo_cached_tokens.png)

![Speedup](fig_repo_speedup.png)

- Average speedup: 1.123x
- Output exact-match rate: 100.0%
- Output token F1 average: 1.0000

| case | agent | cached_lossless | cached_kvcomm | speedup | token_f1 | match |
| --- | --- | --- | --- | --- | --- | --- |
| astropy__astropy-12907 | implementer | 0 | 8192 | 1.3641 | 1.0 | exact_code_content_signature |
| astropy__astropy-12907 | debugger | 18 | 18 | 0.9985 | 1.0 | exact_code_content_signature |
| django__django-10097 | implementer | 40 | 40 | 0.9942 | 1.0 | exact_code_content_signature |
| django__django-10097 | debugger | 40 | 40 | 1.002 | 1.0 | exact_code_content_signature |
| matplotlib__matplotlib-13989 | implementer | 40 | 8232 | 1.3823 | 1.0 | exact_code_content_signature |
| matplotlib__matplotlib-13989 | debugger | 40 | 40 | 0.9942 | 1.0 | exact_code_content_signature |

Result: Qwen3-8B preserves output exactly in this repo-level exact-code setting. Speedup is strongest when the reusable code base lands in a large cacheable chunk.

## 7. Generated Patch Model Ablation

![Generated patch model ablation](fig_generated_patch_models.png)

| model | diffs_extracted | total_outputs | cleanly_applied | passed_tests | lossy_exact_hits | lossy_total |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B | 6 | 6 | 0 | 0 | 3 | 3 |
| Qwen2.5-7B | 3 | 6 | 0 | 0 | 3 | 3 |
| Qwen3-8B | 0 | 6 | 0 | 0 | 2 | 3 |

Result: the generated-patch harness works, but current local 3B/7B/Qwen3-8B models do not produce applyable SWE-bench patches under the present prompt. This is a model/edit-format bottleneck, while lossy mode still shows exact-content KVCOMM hits when anchors are present.

## 8. Recommended Next Run

Run the same package with a stronger coding model and a constrained edit schema. The target claim should be: KVCOMM has <=1-2 percentage point pass@1 delta versus lossless while reducing prefill latency/cached-token work on shared code-base workflows.
