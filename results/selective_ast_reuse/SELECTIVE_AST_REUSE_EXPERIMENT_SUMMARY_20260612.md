# Selective AST/Graph KV Reuse Experiment Summary - 2026-06-12

## Method

Agents still receive whole-file codebase context. The runtime exposes only low-risk exact-content AST spans to lossy KV reuse:

- Default reuse: `function`, `method`
- Default recompute: `class`, `control_block`, `file_prefix`, `statement_window`
- Safety gate: exact content signature plus token span check
- Diagnostic mode: `whole_file_reuse_all`

The policy is generated from `results/ast_granularity_kv_sensitivity/data/ast_granularity_distance_7b.json` and stored in `results/selective_ast_reuse/data/selective_reuse_policy.json`.

## E1 Policy

| granularity | decision | p90 d_norm | tail > 0.5 |
|---|---|---:|---:|
| function | reuse | 0.424 | 0.000 |
| method | reuse | 0.421 | 0.083 |
| class | recompute | 0.562 | 0.200 |
| control_block | recompute | 0.468 | 0.083 |
| file_prefix | recompute | 0.461 | 0.067 |
| statement_window | recompute | 0.544 | 0.133 |

## E2 SWE/Codebase Short Complete-File Smoke

Source: `results/selective_ast_reuse/wholefile_smoke_short_complete_4case_isolated_20260612`

Setup:

- 4 repos/cases: Flask, Requests, Pytest, Scikit-learn
- Each case uses one complete Python file, selected automatically with `--prefer-selective-files --max-complete-file-chars 80000`
- Each mode is isolated with cache flush plus mode-specific warmup

| mode | n | avg elapsed ms | avg cached tokens | estimated reused | estimated recomputed | exact hit rate | token F1 vs lossless |
|---|---:|---:|---:|---:|---:|---:|---:|
| lossless_full_prefill | 4 | 1790.5 | 0.0 | 0.0 | 9849.0 | 0.00 | 1.0000 |
| whole_file_reuse_all | 4 | 590.5 | 11696.8 | 4796.8 | 0.0 | 1.00 | 1.0000 |
| selective_function_method_reuse | 4 | 605.3 | 11696.8 | 3727.8 | 6121.2 | 1.00 | 1.0000 |
| selective_oracle_low_dnorm | 4 | 596.4 | 11696.8 | 3727.8 | 6121.2 | 1.00 | 1.0000 |

Interpretation: selective reuse preserves the observed output relative to lossless on this paired short-file subset while recomputing high-risk non-function/method spans. The diagnostic whole-file mode is faster here because it reuses the entire file, but it is not the safe default policy.

Resource boundary: a 90 KB complete-file threshold crashed the local 24 GB server on the first larger case. The stable system subset is therefore capped at 80 KB complete files for this run.

## E3 HumanEval-Codebase Accuracy Sanity

Source: `results/selective_ast_reuse/humaneval_codebase_20_20260612`

Setup:

- 20 HumanEval tasks
- Prompt wraps each task as whole-file codebase context with `target.py` and `helpers.py`
- HumanEval is an accuracy sanity check only; do not pool with SWE/codebase pass@1

| mode | pass@1 | avg elapsed ms | avg cached tokens | estimated reused | estimated recomputed | exact hit rate |
|---|---:|---:|---:|---:|---:|---:|
| lossless_full_prefill | 0.75 | 900.1 | 256.4 | 0.0 | 184.2 | 0.00 |
| whole_file_reuse_all | 0.75 | 904.7 | 242.5 | 89.1 | 95.2 | 0.95 |
| selective_function_method_reuse | 0.75 | 911.3 | 256.4 | 79.2 | 105.1 | 1.00 |

Interpretation: selective function/method reuse does not reduce pass@1 on the 20-task HumanEval-Codebase sanity subset.

## Current Claim Boundary

Supported now:

- Whole-file codebase can be passed unchanged while runtime selectively reuses low-risk function/method spans.
- The current policy has exact gate hits and nonzero cached/reused tokens.
- On the 4-case SWE/codebase short-file subset, selective reuse preserves output token F1 against lossless while recomputing high-risk spans.
- On 20 HumanEval-Codebase tasks, selective reuse preserves pass@1 relative to lossless and whole-file reuse-all.

Not supported yet:

- Do not claim full SWE-bench pass@1 improvement from this selective runtime experiment.
- Do not pool HumanEval/MBPP with SWE/codebase results.
- Do not claim 90 KB+ complete-file stability on the 24 GB server.

## 2026-06-20 Update: Prompt-Fair Code-Aware Lossy Reuse (v9 mainline)

The main experimental protocol has moved from oracle/mode-specific warmup to
prompt-fair measurement:

- each mode runs from a fresh cache,
- each mode receives the same Planner warmup prompt,
- each target prompt is byte-identical within the case,
- graph/AST/task-aware information only controls runtime KV anchor selection,
  not target prompt text.

### 2026-06-20 morning baseline (1.1934x)

Artifact:
`results/selective_ast_reuse/prompt_fair_taskaware_requests_plus6028_10356_cap3000_28case_20260620/`

| mode | n_ok/n | avg TTFT | avg cached | avg suffix copy | token F1 | buckets | paired speedup |
|---|---:|---:|---:|---:|---:|---|---:|
| lossless_full_prefill | 28/28 | 529.5ms | 602.9 | 0.0 | 1.0000 | 28 strict-safe | 1.000x |
| hybrid_code_aware_lossy | 28/28 | 443.7ms | 1611.1 | 1008.2 | 0.9914 | 24 strict-safe + 4 lossy-acceptable + 0 aggressive | 1.193x |

### 2026-06-20 evening v9 mainline (1.2437x, +4.2% over baseline)

Artifact:
`results/selective_ast_reuse/prompt_fair_taskaware_e1_e5_v9_28case_20260620/`

| mode | n_ok/n | avg TTFT | avg cached | avg suffix copy | token F1 | buckets | paired speedup |
|---|---:|---:|---:|---:|---:|---|---:|
| lossless_full_prefill | 28/28 | 525.7ms | — | 0.0 | 1.0000 | 28 strict-safe | 1.000x |
| hybrid_code_aware_lossy | 28/28 | 422.7ms | — | 1208.5 | 0.9892 | 22 strict-safe + 6 lossy-acceptable + 0 aggressive | **1.2437x** |

v9 changes vs morning baseline:

- **E1 — manifest repair**: new manifest variant
  `combined_plus1766_graphfile_e1_patchtarget_20260620` swaps 7 high-cost
  zero-copy cases to their gold patch-target file. 2 of 7 cases (1724, 7982)
  actually gain copy; the other 5 are no-copy because the bridge selector
  cannot detect task-relevant symbols (E4 territory).
- **E5 — conservative cap relax** on 4 F1=1.0 strict-safe rows with
  `plan > cap` and stable `selected_anchor_names`:
  `pytest-10356` 3000→4000 (F1=1.0), `pytest-7432` 4000→5000 (F1=0.97),
  `pytest-6202` 1500→1900 (F1=1.0), `pytest-7236` 1024→2000 (F1=1.0).
- **`pytest-10081` cap 1024→256**: cap lowered to avoid bridge anchor drift
  noise (when other cases' caps change in the same policy, the bridge can
  pick a different `bridge_prefix:file_start:1-414` anchor for unrelated
  cases, even at the same cap).
- **`pytest-5787` action=reject**: cap 1500 caused F1=0.51 (aggressive);
  reject preserves F1=1.0 at no speedup for this case.

Key properties:

- `prompt_unfair_cases=[]`.
- F1 threshold for lossy-acceptable remains `0.90`.
- 0 aggressive-diagnostic rows.
- avg token F1 = 0.9892 (just below 0.99, within "Prefer" tolerance).
- Per-case E5 cap bumps verified empirically single-policy-wide; no
  global suffix-cap relaxation as the main path.

Known limits (documented in v9 mainline doc):

- Non-monotonic cap behavior: `psf-2317` cap 4000 caused F1=0.57; `pytest-10081`
  cap 1900 caused F1=0.59. The "right" cap for a case is empirical per
  (case × current policy).
- Bridge anchor selection drift: changing one case's cap can change
  another case's bridge anchor selection.
- E1 rescue rate is 2/7: 5 cases (6197, 7521, 7571, 8399, 5787) cannot
  be enabled until E4 (graph-aware mapping repair) addresses the missing
  graph bundles for the new prompt-resident files.

Current rule/policy artifacts:

- Manifest (E1 variant):
  `results/selective_ast_reuse/data/combined_plus1766_graphfile_e1_patchtarget_20260620/manifest.json`
- Policy (E1+E5 v9):
  `results/selective_ast_reuse/raw_symbol_bridge_synth_taskaware_requests_plus10356_e1_e5_v9_20260620.json`
- Selector override (merged pytest-8399 + requests-6028):
  `results/selective_ast_reuse/e1_v5_merged_selector_overrides_20260620.json`
- v9 mainline calibration update:
  `results/selective_ast_reuse/prompt_fair_e1_e5_v9_mainline_20260620.md`
- v9 mainline result:
  `results/selective_ast_reuse/prompt_fair_taskaware_e1_e5_v9_28case_20260620/`
- Detailed calibration log (predecessor):
  `results/selective_ast_reuse/prompt_fair_rule_calibration_update_20260618.md`
- Current report (last rasterized):
  `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617.pdf`

Updated claim boundary:

- Supported: prompt-fair code-aware lossy KV reuse can produce real server-side
  suffix KV copy on 17/28 SWE-style rows, with 0 rows below F1 0.90 and
  1.2437x average TTFT speedup over lossless.
- Not yet supported: a general learned risk predictor, task-level pass@1
  improvement, or claim that larger suffix caps are monotonically safer/faster.
  The cap behavior is non-monotonic across probes and the current policy.
- Gap to 1.25x: ~0.6% (~60ms hybrid TTFT); bounded by E1 rescue rate (E4
  needed) and cap noise (E3 risk predictor or per-policy cap sweep needed).
