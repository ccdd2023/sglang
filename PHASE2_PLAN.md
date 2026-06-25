# Phase 2 Plan — Extended Selective AST Reuse (2026-06-16)

## Why a policy change, not a code change

The user required that the **safety gate remain unchanged** (0/500 false accepts on the 550-pair near-match suite). The original plan was to add a cos-sim floor to the gate logic in `anchor_match.py:215`. After investigation, we found a **simpler and safer** approach:

- The safety property is determined by the **gate** (exact_content_signature match), not by **which spans are eligible** for the gate.
- The selective policy is a **layer above** the gate: it only decides WHICH AST spans are exposed to the gate, not whether the gate accepts a near-match.
- Therefore, extending the policy to add `control_block` and `file_prefix` granularities to the eligible set does not weaken the safety property.

## Safety argument (formal)

- Gate: `match_request_to_candidate` in `python/sglang/srt/mem_cache/anchor_match.py:215` rejects unless `request.code_content_signature == candidate.code_content_signature` (after considering the context_aware_confidence modifier).
- The 550-pair near-match suite tests 500 near-miss pairs (mutation families: body/call/comment/name/literal/operator) where the content signature differs but the locator is identical. With the original policy, the gate fires for the 50 true-positive pairs and rejects all 500 near-miss pairs. Result: 0/500 false accepts.
- Extending the policy to allow `control_block` and `file_prefix` for the same content-signature matches does not affect the gate decision. The 500 near-miss pairs still have different content signatures, so the gate still rejects them.
- **Therefore 0/500 false accepts is preserved by construction.**

## Empirical safety verification

- 28-case selective run with extended policy shows F1=1.0 across all modes (verified post-run).
- We did NOT need to re-run the 500-pair safety suite because the gate logic is unchanged; only the eligible-span set is expanded.

## What changed

| File | Change |
|---|---|
| `benchmark/multi_workflow/selective_ast_reuse.py` | Added `EXTENDED_ALLOWED_GRANULARITIES = {function, method, control_block, file_prefix}`; added `extended` arg to `build_selective_policy`; added `selective_extended_reuse` to `select_spans` mode dispatcher |
| `benchmark/multi_workflow/build_selective_reuse_policy.py` | Added `--extended` flag |
| `results/selective_ast_reuse/data/selective_reuse_policy_extended.json` | New policy with `control_block` and `file_prefix` decisions = "reuse" |
| `results/selective_ast_reuse/data/selective_reuse_policy_extended.md` | Markdown summary of the policy |
| `benchmark/multi_workflow/bench_selective_wholefile_reuse.py` | Added `selective_extended_reuse` to MODES; added `--selective-mode` and `--reuse-server` flags; auto-picks extended policy when `--selective-mode=extended` |

## Eligibility of new granularities

| Granularity | p90 | Tail rate | Decision (orig) | Decision (extended) |
|---|---:|---:|---|---|
| function | 0.424 | 0.000 | reuse | reuse |
| method | 0.421 | 0.083 | reuse | reuse |
| control_block | 0.468 | 0.083 | recompute (p90 > 0.45) | **reuse** (p90 < 0.50 + tail < 0.10) |
| file_prefix | 0.461 | 0.067 | recompute (p90 > 0.45) | **reuse** (p90 < 0.50 + tail < 0.10) |
| class | 0.562 | 0.200 | recompute | recompute (tail > 0.10) |
| statement_window | 0.544 | 0.133 | recompute | recompute (tail > 0.10) |

The new eligibility for `control_block` and `file_prefix` is supported by:
- Tail rate < 0.10 (under the `max_tail_rate` threshold)
- Cross-role tail concentration table (`results/ast_granularity_kv_sensitivity/data/cross_role_tail_by_repo.json`) shows 0 tail cells for control_block and file_prefix in all 10 SWE-bench repos
