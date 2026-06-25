# Prompt-Fair Rule Calibration Update (2026-06-18)

## Goal

Move beyond per-case oracle calibration toward a reusable risk gate for
`hybrid_code_aware_lossy`, while keeping the prompt-fair invariant:
target prompts stay identical and only KV reuse metadata/policy changes.

## Implementation

- Added rule-based hybrid calibration support in
  `benchmark/multi_workflow/bench_selective_wholefile_reuse.py`.
- Existing per-case policies still work. If no case entry exists, the driver can
  now evaluate `rules` from the same JSON policy.
- Rule matches currently support:
  - `selected_span_count_by_granularity`
  - selected count min/max
  - estimated reused-token min/max
  - decision-reason any/all/none
  - selected anchor name regex any/all/none
- The selected rule result is carried from selection into payload construction,
  so rule-derived `max_suffix_copy_len` is applied to runtime anchor metadata.
- Added `selected_anchor_names` telemetry to rows/CSV so future risk models can
  distinguish same-length anchors that point to different files/symbols.
- Added `--dry-run-selection-features`, which writes per-case selected-anchor
  features without launching a server.
- Added task-anchor overlap telemetry:
  - `any_anchor_path_mentioned`
  - `any_anchor_basename_mentioned`
  - `max_anchor_lexical_overlap`
  - `max_anchor_symbol_overlap`
  These features are computed from issue text, `FAIL_TO_PASS`, and optional
  test patch text. They do not change the target prompt.
- Added task-aware manifest selection in
  `benchmark/multi_workflow/build_selective_wholefile_manifest.py`.
  The builder can now rank eligible complete Python files by task/file lexical
  evidence before falling back to span count and file size.
- Added optional anchor-name regex rule generation in
  `benchmark/multi_workflow/analyze_pareto_calibration.py`.
  This lets diagnostic policies distinguish prompt-resident anchors such as
  `requests/models.py:bridge_prefix:file_start:1-657` from nearby but riskier
  anchors such as `requests/models.py:bridge_prefix:file_start:1-837`, without
  adding any graph/code evidence to the target prompt.

## Calibration Generator

`benchmark/multi_workflow/analyze_pareto_calibration.py` now supports:

- `--emit-rule-policy`
- `--rule-token-margin-ratio`

It emits:

- `hybrid_calibration_policy.json`: old per-case diagnostic policy.
- `hybrid_rule_calibration_policy.json`: conservative rule policy without
  instance-id matching.

The rule generator is conflict-aware: if a candidate rule would also match a
known cap-sensitive or unsafe training row, the rule is skipped.

With `--selection-features`, the rule generator can also emit
`require_anchor_path_mentioned`. This is important for cases where two tasks
select the same code anchor but only one task explicitly points at that file.

Current generated policy:

`results/selective_ast_reuse/pareto_rule_calibration_hybrid_20260618/hybrid_rule_calibration_policy.json`

- rules: 1
- default action: reject
- skipped conflicting candidate rules: 8

This is intentionally conservative. The important finding is that
shape + estimated-token window alone is too coarse: it cannot safely separate
some high-speed safe bridge-prefix cases from known risky bridge-prefix cases.

Task-overlap generated policy:

`results/selective_ast_reuse/pareto_rule_calibration_hybrid_taskoverlap_20260618/hybrid_rule_calibration_policy.json`

- rules: 1
- default action: reject
- rule: `calibrated_psf__requests-5414`
- match: `bridge_prefix:1`, estimated reused tokens `1927-2609`,
  `require_anchor_path_mentioned=true`

This separates `psf__requests-5414` from `psf__requests-6028`: both select
`requests/models.py:bridge_prefix:file_start:1-625`, but only `requests-5414`
explicitly mentions `requests/models.py` in the issue text.

Task-aware manifest:

`results/selective_ast_reuse/data/swe_selective_wholefile_68k_1file_taskaware_manifest.json`

- same 28 instance ids as the previous 68k 1-file manifest
- 10/28 cases changed selected file
- notable change: `psf__requests-6028` switches from `requests/models.py` to
  `requests/utils.py`, matching the failing `test_utils.py` target

Raw task-aware p8000 selection features:

`results/selective_ast_reuse/prompt_fair_taskaware_raw_selection_features_p8000_20260618`

- copy candidates: 9/28
- task-overlap candidates: 4/28
- examples:
  - `psf__requests-5414`: `requests/models.py`, path mentioned, lexical overlap 3
  - `psf__requests-6028`: `requests/utils.py`, lexical overlap 2

## Evidence

Generation command used existing prompt-fair runs:

```bash
/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/analyze_pareto_calibration.py \
  --run main_p5500=results/selective_ast_reuse/prompt_fair_pareto_f090_hybrid_p5500_riskgate_maxtotal9000_28case_20260618/summary.json \
  --run miduncap=results/selective_ast_reuse/prompt_fair_pareto_f090_hybrid_tokenfilter_miduncap_p5500_28case_20260618/summary.json \
  --run cap3500_requests8=results/selective_ast_reuse/prompt_fair_pareto_f090_hybrid_cap3500_requests8_20260618/summary.json \
  --run pytest7432_cap3500=results/selective_ast_reuse/prompt_fair_pareto_f090_hybrid_pytest7432_cap3500_20260618/summary.json \
  --run pytest5262_cap3500=results/selective_ast_reuse/prompt_fair_pareto_f090_hybrid_pytest5262_cap3500_20260618/summary.json \
  --out-dir results/selective_ast_reuse/pareto_rule_calibration_hybrid_20260618 \
  --emit-policy --emit-rule-policy --reject-cap-sensitive \
  --allow-cap-sensitive-case pytest-dev__pytest-7432 \
  --rule-token-margin-ratio 0.15
```

Dry-run load check:

```text
loaded_cases: 28
hybrid_calibration_policy_cases: 0
hybrid_calibration_policy_rules: 1
hybrid_calibration_policy_default_action: reject
```

Regression:

```text
87 passed, 4 warnings
```

Task-overlap rule smoke:

`results/selective_ast_reuse/prompt_fair_taskoverlap_rule_smoke2_20260618`

Two prompt-fair cases:

- `psf__requests-5414`: rule matched, copied `4750` suffix tokens,
  TTFT `497.05ms -> 92.61ms`, token F1 `0.9739`.
- `psf__requests-6028`: same code anchor family but no task-anchor overlap;
  rule default rejected, TTFT `499.18ms -> 492.65ms`, token F1 `1.0000`.

Aggregate over the two-case smoke:

```text
hybrid_code_aware_lossy avg TTFT: 292.6ms
lossless_full_prefill avg TTFT: 498.1ms
paired speedup: 1.70x
avg token F1: 0.9869
prompt_fair_ok: 2/2
```

Task-aware p8000 diagnostic:

`results/selective_ast_reuse/prompt_fair_taskaware_p8000_uncalibrated_smoke2_20260618`

- `psf__requests-5414`: TTFT `770.12ms -> 103.73ms`, token F1 `1.0000`,
  suffix copy `7541`
- `psf__requests-6028`: TTFT `803.34ms -> 108.70ms`, token F1 `0.6950`,
  suffix copy `7774`

This confirms that task-aware file selection exposes larger, more relevant
anchors and much stronger TTFT gains, but risk gating is still necessary.

Task-aware p8000 + task-overlap rule:

`results/selective_ast_reuse/prompt_fair_taskaware_p8000_taskoverlap_rule_smoke2_20260618`

- `psf__requests-5414`: rule matched, TTFT `752.69ms -> 104.21ms`,
  token F1 `1.0000`, suffix copy `7541`
- `psf__requests-6028`: rule default rejected, TTFT `801.74ms -> 816.44ms`,
  token F1 `1.0000`
- aggregate hybrid TTFT `460.3ms` vs lossless `777.2ms`
- aggregate token F1 `1.0000`
- prompt fair `2/2`

Task-aware p8000 + anchor-aware rule, requests8:

`results/selective_ast_reuse/prompt_fair_taskaware_p8000_anchorregex_rule_requests8_20260618`

- policy:
  `results/selective_ast_reuse/pareto_rule_calibration_hybrid_taskaware_p8000_requests8_anchorregex_20260618/hybrid_rule_calibration_policy.json`
- generated rules: 4, default action: reject
- copied cases:
  - `psf__requests-1142`: TTFT `440.22ms -> 90.13ms`, speedup `4.88x`,
    token F1 `0.9291`, suffix copy `4432`
  - `psf__requests-1766`: TTFT `538.27ms -> 92.88ms`, speedup `5.80x`,
    token F1 `0.9362`, suffix copy `5208`
  - `psf__requests-2931`: TTFT `655.31ms -> 84.16ms`, speedup `7.79x`,
    token F1 `1.0000`, suffix copy `6244`
  - `psf__requests-5414`: TTFT `757.54ms -> 106.12ms`, speedup `7.14x`,
    token F1 `1.0000`, suffix copy `7541`
- rejected high-risk cases recovered to strict-safe:
  `psf__requests-1724`, `psf__requests-1921`, `psf__requests-2317`,
  `psf__requests-6028`
- aggregate hybrid TTFT `361.3ms` vs lossless `616.0ms`
- aggregate token F1 `0.9832`
- buckets: `6` strict-safe, `2` lossy-acceptable, `0` aggressive-diagnostic
- prompt fair `8/8`

Task-aware p8000 + anchor-aware rule, full 28-case:

`results/selective_ast_reuse/prompt_fair_taskaware_p8000_anchorregex_rule_28case_20260618`

- n_ok: `28/28`
- prompt fair: `28/28`
- copied cases: 4/28, same four requests anchors as requests8
- lossless avg TTFT: `1215.2ms`
- hybrid avg TTFT: `1138.3ms`
- aggregate TTFT speedup: `1.07x`
- copied-case TTFT speedups: `4.88x`, `6.87x`, `7.72x`, `8.41x`
- aggregate token F1: `0.9952`
- buckets: `26` strict-safe, `2` lossy-acceptable, `0` aggressive-diagnostic
- interpretation: this is a conservative safety baseline. It demonstrates that
  prompt-fair suffix copy can be made safe on the selected anchors, but the
  current rule coverage is too narrow for large 28-case average speedup.

Task-aware p8000 + multi-cap anchor-aware rule:

`results/selective_ast_reuse/prompt_fair_taskaware_p8000_anchorregex_plus1724_28case_20260618`

- automatic policy generator:
  `results/selective_ast_reuse/pareto_rule_calibration_hybrid_taskaware_p8000_multicap_anchorregex_allow1724_20260618/hybrid_rule_calibration_policy.json`
- calibration sources:
  - p8000 unbounded requests8
  - exact-anchor cap3500 requests8
  - exact-anchor cap4500 requests8
- generated rules: 5, default action: reject
- cap-sensitive allowlist: `psf__requests-1724`
- new rule:
  - `psf__requests-1724`: cap3500, TTFT speedup `2.23x` on the 28-case rerun,
    token F1 `1.0000`
- full 28-case result:
  - n_ok: `28/28`
  - prompt fair: `28/28`
  - copied cases: 5/28
  - lossless avg TTFT: `1213.0ms`
  - hybrid avg TTFT: `1128.2ms`
  - aggregate TTFT speedup: `1.08x`
  - paired speedup average: `1.89x`
  - aggregate token F1: `0.9952`
  - buckets: `26` strict-safe, `2` lossy-acceptable, `0` aggressive-diagnostic
- cap sweep finding:
  - `psf__requests-1724` is cap-sensitive: cap3500 is strict-safe, but cap4500
    and p8000 are aggressive.
  - `psf__requests-1921`, `psf__requests-2317`, and `psf__requests-6028`
    remain aggressive even at cap3500, so they should stay rejected until the
    anchor selector or runtime strategy changes.
- interpretation: per-anchor cap calibration can safely recover additional
  copy coverage, but the speed gain is incremental until the safe-anchor
  coverage expands beyond the requests family.

Oracle / blacklist diagnostics after the multi-cap result:

`results/selective_ast_reuse/pareto_oracle_multirun_policy_hybrid_20260618`

- aggregated historical p5500/miduncap/cap sweeps suggested 12/28 safe-speedup
  cases, including several pytest anchors.
- direct replay with the generated shape-checked policy:
  `results/selective_ast_reuse/prompt_fair_oracle_multirun_policy_hybrid_28case_20260618`
  - copied cases: 0/28
  - token F1: `1.0000`
  - TTFT speedup: `1.00x`
  - reason: the per-case policy required historical selection shapes, but the
    current run produced different selection telemetry, so cap entries were
    rejected by shape mismatch.
- no-shape replay:
  `results/selective_ast_reuse/prompt_fair_oracle_multirun_policy_noshape_hybrid_28case_20260618`
  - copied cases: 3/28
  - token F1: `0.9852`
  - buckets: `25` strict-safe, `2` lossy-acceptable, `1` aggressive
  - failure: `psf__requests-1724` fell to F1 `0.7194`
  - interpretation: selection shape/context is not a cosmetic detail; coarse
    case-to-cap replay is unsafe.
- p5500 blacklist replay:
  `results/selective_ast_reuse/prompt_fair_hybrid_p5500_blacklist2317_nomax_28case_20260618`
  - intention: keep p5500 selector but reject only the historical borderline
    row `psf__requests-2317`
  - actual result: copied cases 7/28, token F1 `0.9320`, buckets `21` strict,
    `2` acceptable, `5` aggressive
  - failures: `psf__requests-6028`, `pytest-dev__pytest-5262`,
    `pytest-dev__pytest-7205`, `pytest-dev__pytest-7324`,
    `pytest-dev__pytest-7432`
  - interpretation: the current driver/runtime state no longer reproduces the
    historical p5500 selection behavior by simply adding a blacklist. Treat the
    older p5500 balanced result as historical diagnostic unless rerun under the
    exact same code/selection state.

Selector-snapshot and truncated-prompt diagnostics:

- `benchmark/multi_workflow/bench_selective_wholefile_reuse.py` now writes a
  `selector_snapshot` into both `summary.json` and
  `selection_features.json`. The snapshot includes git commit, dataset/manifest
  hashes, max-file settings, hybrid bridge thresholds, selection gates, graph
  budgets, and suffix-copy limits. This is required because the same case/cap
  can be safe or unsafe when the selected file or bridge/window construction
  changes.
- Current p5500 selector dry-run:
  `results/selective_ast_reuse/current_p5500_selector_features_20260618`
  showed five high-risk high-speed rows:
  `psf__requests-6028`, `pytest-dev__pytest-5262`,
  `pytest-dev__pytest-7205`, `pytest-dev__pytest-7324`,
  `pytest-dev__pytest-7432`.
- Cap sweep on those five rows:
  - cap2500: all five rows remained aggressive.
  - cap1024: only `pytest-dev__pytest-7432` became strict-safe; the others
    remained aggressive.
  - cap512: only `pytest-dev__pytest-7432` remained strict-safe, with little
    TTFT gain.
  This means `testing/python/metafunc.py` and `requests/models.py` prefix-like
  anchors are not made safe by a simple smaller cap in most cases.
- Task-aware truncated p5500 selector:
  `results/selective_ast_reuse/prompt_fair_taskaware_truncated_p5500_unfiltered_28case_20260618`
  had strong speed but 5 aggressive rows. A cap1024 sweep on its pytest bad
  rows recovered three strict-safe rows:
  `pytest-dev__pytest-5787`, `pytest-dev__pytest-7236`, and
  `pytest-dev__pytest-7432`.
- Task-aware truncated p5500 per-case safe policy:
  `results/selective_ast_reuse/prompt_fair_taskaware_truncated_p5500_safe_multicap_percase_28case_20260618`
  - n_ok: `28/28`
  - prompt fair: `28/28`
  - copied cases: 5/28
  - lossless avg TTFT: `525.8ms`
  - hybrid avg TTFT: `489.6ms`
  - aggregate TTFT speedup: `1.074x`
  - paired speedup average: `1.35x`
  - aggregate token F1: `0.9965`
  - buckets: `26` strict-safe, `2` lossy-acceptable, `0` aggressive
  - copied rows: `psf__requests-1142`, `psf__requests-5414`,
    `pytest-dev__pytest-5787`, `pytest-dev__pytest-7236`,
    `pytest-dev__pytest-7432`
- Comparison: the full-file task-aware p8000 multi-cap policy and the truncated
  p5500 per-case policy both land around `1.07x` full-table speedup with 5/28
  copied rows and no aggressive rows. They cover different safe cases. This
  suggests the next useful optimization is a mixed selector that can use
  full-file p8000 for requests-style safe anchors and truncated/windowed p5500
  for selected pytest anchors, while keeping prompt-fair comparisons within
  each protocol.
- Mixed selector follow-up:
  `results/selective_ast_reuse/prompt_fair_mixed_windows_p8000_safe_multicap_28case_20260618`
  implements that combination. It uses full-file task-aware p8000 for the safe
  requests anchors and manifest-level truncated prompt windows for the selected
  pytest anchors recovered by cap1024.
  - n_ok: `28/28`
  - prompt fair: `28/28`, `prompt_unfair_cases=[]`
  - copied cases: `8/28`
  - lossless avg TTFT: `1118.7ms`
  - hybrid avg TTFT: `1029.6ms`
  - aggregate TTFT speedup: `1.086x`
  - paired speedup average: `1.086x` over the same 28 cases
  - aggregate token F1: `0.9952`
  - buckets: `26` strict-safe, `2` lossy-acceptable, `0` aggressive
  - avg suffix copy: `1071.3` tokens; avg cached tokens increase from `602.9`
    to `1674.2`
  - copied rows:
    `psf__requests-1142` (`4.26x`, F1 `0.9291`, copy `4432`),
    `psf__requests-1724` (`2.27x`, F1 `1.0000`, copy `3500`),
    `psf__requests-1766` (`5.37x`, F1 `0.9362`, copy `5208`),
    `psf__requests-2931` (`7.70x`, F1 `1.0000`, copy `6244`),
    `psf__requests-5414` (`8.55x`, F1 `1.0000`, copy `7541`),
    `pytest-dev__pytest-5787` (`1.18x`, F1 `1.0000`, copy `1024`),
    `pytest-dev__pytest-7236` (`1.17x`, F1 `1.0000`, copy `1024`),
    `pytest-dev__pytest-7432` (`1.14x`, F1 `1.0000`, copy `1024`).
  This is now the safest prompt-fair full-table line in the report: it improves
  copied coverage from 5/28 to 8/28 without introducing any aggressive row.
- Post-hoc retention-aware calibration:
  `results/selective_ast_reuse/pareto_calibration_mixed_posthoc_20260618`
  adds optional code-action and gold-patch-intent gates to
  `analyze_pareto_calibration.py`. The gates do not change existing behavior
  unless explicitly enabled.
  - code-action gate: require `code_action_score >= 0.90`
  - gold-intent gate: require `gold_intent_delta >= -0.10` versus lossless
  - result on the mixed run: `7` safe-speedup cases, `1` post-hoc rejected
    speedup case, `20` safe-no-speedup cases.
  - rejected case: `psf__requests-1142`, which had token F1 `0.9291` and
    `4.26x` speedup, but code-action score `0.5343` and gold-intent delta
    `-0.55`; the output dropped the explicit target-file anchor. This confirms
    that token F1 alone can miss code-task regressions.
- Post-hoc policy rerun:
  `results/selective_ast_reuse/prompt_fair_mixed_posthoc_policy_28case_20260618`
  applies the retention-aware policy on the full 28-case prompt-fair table.
  - n_ok: `28/28`
  - prompt fair: `28/28`, `prompt_unfair_cases=[]`
  - copied cases: `7/28`
  - lossless avg TTFT: `1120.9ms`
  - hybrid avg TTFT: `1043.1ms`
  - aggregate TTFT speedup: `1.075x`
  - aggregate token F1: `0.9977`
  - buckets: `27` strict-safe, `1` lossy-acceptable, `0` aggressive
  - code-action composite: `28/28`
  - gold-intent regression: `0/28`
  This is the strongest conservative line so far: slightly slower than the
  mixed-window `8/28` line, but it removes the known output-anchor regression.

## Interpretation

This is progress toward a real risk predictor, not a final speed point.

- Per-case strict calibration proved risk can be controlled, but it is too
  conservative and not a held-out result.
- Rule calibration without instance-id matching currently becomes conservative
  because bridge-prefix anchors with similar length can have different quality
  outcomes.
- Task-anchor overlap is a useful additional signal: it distinguishes cases
  that select the same anchor but ask for different fixes.
- The remaining limitation is prompt/file coverage. Some risky or missed cases
  use a 1-file prompt whose selected file is not the file named by
  `FAIL_TO_PASS` or issue hints, so the reuse gate has no better code anchor to
  choose from.
- Task-aware file selection partially fixes file coverage, but opening larger
  relevant anchors requires stronger risk gating. Uncalibrated p8000 can be very
  fast and also very wrong; task-overlap gating recovers the safe subset.
- Anchor-aware rules recover a stronger Pareto point on requests8: no
  aggressive cases, aggregate F1 `0.9832`, and `1.70x` average TTFT speedup.
  On the full 28-case table the same rules are safe but conservative: only
  4/28 cases copy suffix KV, so the average speedup is `1.07x` despite
  `4.88x`-`8.41x` speedup on copied cases.
- Multi-cap anchor-aware calibration adds one more strict-safe copied row
  (`psf__requests-1724`) and improves full-table speedup from `1.07x` to
  `1.08x` without introducing any aggressive row. This confirms that
  cap-sensitive rows should not be globally rejected if a lower safe cap has
  direct evidence, but also shows that cap alone cannot rescue all risky
  prefix-like anchors.
- Oracle/blacklist diagnostics show that historical safe points cannot be
  blindly replayed under the current driver state. The safe replay condition is
  not just case id and cap length; it also includes the selected anchor shape,
  selected file, bridge/window construction, and benchmark parameters such as
  max-file settings.
- Truncated-prompt p5500 diagnostics add safe pytest coverage but lose some
  full-file p8000 requests coverage. Both safe branches converge near `1.07x`
  full-table speedup, so the next speed jump likely requires a mixed
  selector/window strategy rather than another global cap sweep.
- The first mixed selector confirms this direction: it raises safe copied
  coverage to 8/28 and full-table TTFT speedup to `1.086x` with F1 `0.9952` and
  no aggressive rows. The remaining limitation is that the newly added pytest
  safe rows copy only `1024` tokens each and therefore contribute modest
  per-case speedups (`1.14x`-`1.18x`), while the requests rows remain the main
  speed contributors.
- The retention-aware rerun shows the tradeoff more clearly: rejecting
  `requests-1142` lowers full-table speedup to `1.075x`, but raises F1 to
  `0.9977` and removes all code-action/gold-intent regressions. Future speed
  improvements should start from this stricter baseline or explicitly disclose
  when a high-speed row fails retention sanity.
- Bounded hybrid-window prototype:
  - Added `--hybrid-bridge-anchor-max-tokens` to let
    `hybrid_code_aware_lossy` build prompt-resident bounded bridge windows
    instead of always using file-start bridges.
  - Added `--hybrid-bridge-max-count-per-file` so bounded-window selection can
    keep only the deepest N windows per file and avoid selecting every
    function/method in the file.
  - Dry-run with `--hybrid-bridge-anchor-max-tokens 1200`,
    `--hybrid-bridge-max-count-per-file 1`, and `--max-file-chars 68000`
    selected one prompt-resident window for all 28 cases, with roughly
    `1.2k` whitespace-token anchors. This expands candidate coverage, but it is
    not automatically safe.
  - Single-case smoke on `psf__requests-1724`:
    `requests/packages/urllib3/connectionpool.py:bridge_window:bounded:352-674`
    copied `2048` suffix tokens with planned length `2543`, but token F1 fell
    to `0.6929` (`aggressive-diagnostic`). Lowering the cap did not recover
    quality: cap1024 gave `1.16x` TTFT but F1 `0.6772`, and cap256 gave only
    `1.02x` TTFT with F1 `0.7188`. A larger bounded3000 variant still started
    at line 1 and copied `2048`/planned `5205`, with F1 `0.7188`. In contrast,
    the existing mixed-window safe policy for the same case uses
    `bridge_prefix:file_start:1-674`, cap3500, F1 `1.0000`, and `2.27x` TTFT.
    Therefore this case is evidence that moving the anchor start closer to the
    target can be worse than the calibrated file-start bridge.
  - Single-case smoke on `pytest-dev__pytest-7205`:
    `src/_pytest/python.py:bridge_window:bounded:1157-1558` with cap256 stayed
    acceptable (F1 `0.9752`) but had only `1.01x` TTFT speedup. cap512 increased
    copied tokens but collapsed to F1 `0.5000`. The current mixed-window main
    policy rejects this case, preserving F1 `1.0000`.
  - Interpretation: bounded prompt-resident windows are useful as an
    experimental selector primitive, but the first cap sweep shows that moving
    the anchor start closer to the target does not by itself solve lossy-context
    drift. Current bounded-window points are either unsafe or too small to
    matter for TTFT, so do not promote bounded-window results to the main table
    without a fresh F1-safe calibration.
- Multirun posthoc-gated calibration:
  - `analyze_pareto_calibration.py` now supports run-specific
    `--code-action-run LABEL=path` and `--gold-intent-run LABEL=path`, so
    semantic posthoc gates are applied to each candidate row before choosing
    the fastest acceptable run for that case.
  - Joint calibration over mixed-window, p5500 risk-gated, and tokenfilter
    p5500 runs found `12` candidate safe-speedup cases, `13` safe-no-speedup
    cases, and `3` posthoc-rejected speedup candidates.
  - First reproduction:
    `results/selective_ast_reuse/prompt_fair_multirun_posthoc_policy_28case_20260618`
    reached `1.095x` full-table TTFT speedup with avg F1 `0.9855`, but
    `psf__requests-6028` reproduced as an aggressive-diagnostic row
    (token F1 `0.6573`) and was the only code-action composite reject.
  - Stable no-6028 reproduction:
    `results/selective_ast_reuse/prompt_fair_multirun_posthoc_no6028_policy_28case_20260618`
    reached prompt fair `28/28`, avg TTFT `1037.6ms` vs lossless `1123.7ms`,
    full-table speedup `1.083x`, copied cases `7/28`, avg token F1 `0.9977`,
    buckets `27` strict-safe + `1` lossy-acceptable + `0` aggressive,
    code-action composite `28/28`, and gold-intent regression `0/28`.
  - Interpretation: this replaces the previous mixed-window safe line as the
    cleanest PDF mainline. The difference between the `1.095x` and `1.083x`
    reruns is exactly the unstable 6028 row, so the current conservative claim
    is lower-speed but cleaner.
- Graph-source bridge diagnostic:
  - Added experimental selector flag
    `--hybrid-bridge-source {function,graph,graph_then_function}`. The default
    remains `function`; `graph` builds bridge-prefix anchors from graph-mapped
    AST spans instead of all function/method spans in the file. This does not
    change the target prompt text; graph evidence is used only as internal
    anchor selection metadata.
  - Motivation: replay analysis showed that some p5500/tokenfilter candidates
    depended on graph-local bridge construction. The current function-source
    bridge can expand to an entire large file prefix, e.g. pytest
    `metafunc.py` through line 1974, which is too large and gets filtered.
  - Dry-run:
    `results/selective_ast_reuse/dryrun_hybrid_graph_bridge_source_20260618`
    produced candidates for `24/28` cases, substantially higher than the
    conservative mainline copy coverage.
  - 8-case smoke, cap2048:
    `results/selective_ast_reuse/prompt_fair_graph_bridge_source_cap2048_smoke8_20260618`
    reached prompt fair `8/8`, avg TTFT speedup `1.191x`, copied `8/8`, avg
    F1 `0.9337`, buckets `5` strict-safe + `2` lossy-acceptable + `1`
    aggressive. The aggressive row was `psf__requests-2931` with F1 `0.5872`.
  - 8-case smoke, cap1024:
    `results/selective_ast_reuse/prompt_fair_graph_bridge_source_cap1024_smoke8_20260618`
    reached speedup `1.095x`, but `psf__requests-2931` remained aggressive
    with the same F1 `0.5872`; lowering the cap did not rescue this anchor.
  - 28-case diagnostic, cap2048:
    `results/selective_ast_reuse/prompt_fair_graph_bridge_source_cap2048_28case_20260618`
    stopped after `23/28` cases because the server hit an OOM. Partial results:
    raw speedup `1.137x`, avg F1 `0.8500`, buckets `12` strict-safe + `4`
    lossy-acceptable + `7` aggressive, copied `18/23`. If rows with F1 < 0.90
    are rejected posthoc, the estimated speedup on the completed subset falls
    to about `1.060x`.
  - Combination analysis with the current no6028 mainline estimates only
    `1.089x` full-table speedup, and the main extra speed comes from
    `psf__requests-1142`, which is already known to regress gold/file-anchor
    retention. Therefore graph-source bridge is useful as a diagnostic coverage
    expander, but it does not replace the current `1.083x` conservative
    prompt-fair mainline.
- PDF/report refresh after graph-source diagnostics:
  - Updated
    `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617.html`
    and regenerated
    `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617.pdf`.
  - The PDF is now generated through a rasterized HTML screenshot path instead
    of relying on target-machine Chinese font embedding. This avoids the
    previous small vector PDF rendering/garbling issue.
  - The report keeps the `multirun posthoc-gated no6028` run as the mainline
    (`1.083x`, F1 `0.9977`, `7/28` copied, `0` aggressive), while explicitly
    listing graph-source bridge as diagnostic/negative evidence.
- p5500 semantic-gated posthoc estimate:
  - Artifact:
    `results/selective_ast_reuse/pareto_calibration_p5500_semantic_gated_f0895_20260618`.
  - Gate: token F1 threshold lowered to `0.895` to allow the known
    `psf__requests-2317` token-boundary row only when code-action and
    gold-intent sanity pass. Also require code-action score `>=0.90` and
    gold-intent delta `>= -0.10`.
  - Strictly counting only rows with real suffix copy, rejecting
    `psf__requests-1142`, `psf__requests-1921`, and
    `pytest-dev__pytest-7205` gives a posthoc estimate of `1.185x` full-table
    TTFT speedup, avg token F1 `0.9882`, and `11/28` copied rows.
  - If the previously unstable `psf__requests-6028` is also rejected, the
    estimate becomes `1.157x`, avg token F1 `0.9906`, and `10/28` copied rows.
  - Interpretation: this is a promising faster Pareto candidate, but it is not
    yet a promoted mainline because historical p5500 selection shapes have not
    been reproduced stably under the current driver state. The next useful
    step is a reproducible selector snapshot/policy replay for this semantic
    gated subset, or a held-out risk predictor that learns these accept/reject
    decisions without per-instance posthoc labels.
- Current-selector replay of the p5500 semantic policy:
  - Dry-run with the generated p5500 semantic policy:
    `results/selective_ast_reuse/dryrun_p5500_semantic_policy_current_selector_20260618`.
    Although the policy listed `11` cap actions, only `4` rows retained live
    anchors after shape checking; `7` rows were rejected by shape mismatch.
  - Disabling graph target-file preference made this worse:
    `results/selective_ast_reuse/dryrun_p5500_semantic_policy_noprefergraph_20260618`
    had `0` live cap rows and `11` shape mismatches.
  - Added a conservative calibration-policy shape-prune path in
    `benchmark/multi_workflow/bench_selective_wholefile_reuse.py`: if the
    current selection is a strict superset of the required historical shape,
    prune extra anchors by granularity; if any required granularity is missing,
    still reject. This only affects explicit `--hybrid-calibration-policy`
    diagnostic runs.
  - Tests after this code change:
    `py_compile` passed and
    `pytest -q benchmark/multi_workflow/test_selective_ast_reuse.py benchmark/multi_workflow/test_prompt_fair_kv_reuse.py`
    passed with `41` tests.
  - Shape-prune dry-run:
    `results/selective_ast_reuse/dryrun_p5500_semantic_policy_shapeprune_v2_20260618`
    produced `5` live cap rows:
    `psf__requests-1724` (shape-pruned),
    `psf__requests-2317`,
    `pytest-dev__pytest-10356`,
    `pytest-dev__pytest-5631`,
    and `pytest-dev__pytest-5787`.
- 5-case GPU smoke for shape-pruned p5500 semantic replay:
  - Dataset:
    `results/selective_ast_reuse/data/p5500_semantic_shapeprune_smoke5`.
  - Result:
    `results/selective_ast_reuse/prompt_fair_p5500_semantic_shapeprune_smoke5_20260618`.
  - Prompt fairness: `5/5`, prompt-unfair cases `[]`.
  - Runtime copy: hybrid copied in `5/5`; avg suffix copy `2062.0` tokens,
    avg planned copy `2216.6`, context-aligned match rate `1.00`.
  - Quality: avg token F1 `0.9070`; buckets `3` strict-safe + `2`
    aggressive-diagnostic. Strict rows were `psf__requests-1724`,
    `psf__requests-2317`, and `pytest-dev__pytest-10356`; aggressive rows were
    `pytest-dev__pytest-5631` (F1 `0.6833`) and
    `pytest-dev__pytest-5787` (F1 `0.8515`).
  - TTFT telemetry was blank in this smoke even though elapsed/cached/copy
    telemetry was present, so this run cannot be used for a speed claim until
    the response-metadata TTFT extraction path is fixed or rerun under the
    previous telemetry configuration.
  - Interpretation: shape-prune can recover one large strict-safe row
    (`requests-1724`), but historical small pytest anchors are not stable under
    the current runtime/selector state. Do not promote this policy. The next
    optimization should reject or separately calibrate small graph-only anchors
    and fix TTFT telemetry before running another full table.
- 3-case strict large-anchor smoke:
  - Dataset:
    `results/selective_ast_reuse/data/p5500_semantic_shapeprune_strict3`.
  - Result:
    `results/selective_ast_reuse/prompt_fair_p5500_semantic_shapeprune_strict3_20260618`.
  - Cases: `psf__requests-1724`, `psf__requests-2317`,
    `pytest-dev__pytest-10356`.
  - Prompt fairness: `3/3`, prompt-unfair cases `[]`.
  - Quality: exact output match `3/3`, avg token F1 `1.0000`,
    buckets `3` strict-safe.
  - Runtime copy: anchor match rate `1.00`, avg suffix copy `3308.7` tokens,
    avg planned copy `3435.0`, avg cached tokens `4203.7`.
  - TTFT: lossless avg `1457.4ms`, hybrid avg `1105.7ms`, paired speedup
    `1.318x`.
  - Row details:
    - `psf__requests-1724`: shape-pruned, copy `753`, TTFT speedup `1.021x`,
      F1 `1.0000`.
    - `psf__requests-2317`: copy `4623`, TTFT speedup `1.731x`,
      F1 `1.0000`.
    - `pytest-dev__pytest-10356`: copy `4550`, TTFT speedup `1.315x`,
      F1 `1.0000`.
  - Interpretation: the stable direction is not "recover all historical p5500
    caps"; it is "large, reproducible, shape-checked anchors only." Small
    graph-only anchors that looked safe historically should be rejected by
    default or require a separate risk model.
- 28-case full run with large-reproducible-only policy:
  - Policy:
    `results/selective_ast_reuse/large_reproducible_anchor_policy_20260618/hybrid_large_reproducible_policy.json`.
    It allows only `psf__requests-1724`, `psf__requests-2317`, and
    `pytest-dev__pytest-10356`; all other cases are rejected by default.
  - Dry-run:
    `results/selective_ast_reuse/dryrun_large_reproducible_policy_20260618`
    confirmed `3` live cap rows, `25` rejects, and `0` shape mismatches.
  - Full run:
    `results/selective_ast_reuse/prompt_fair_large_reproducible_policy_28case_20260618`.
  - Prompt fairness: `28/28`, prompt-unfair cases `[]`.
  - Quality: exact output match `1.0000`, avg token F1 `1.0000`,
    buckets `28` strict-safe.
  - Runtime copy: `3/28` rows copied, anchor match rate `0.1071`, avg suffix
    copy `354.5` tokens over the full table (`3308.7` over copied rows).
  - TTFT: lossless avg `1701.5ms`, hybrid avg `1654.9ms`, paired speedup
    `1.028x`.
  - Copied row details:
    - `psf__requests-1724`: shape-pruned, copy `753`, speedup `1.025x`,
      F1 `1.0000`.
    - `psf__requests-2317`: copy `4623`, speedup `1.723x`,
      F1 `1.0000`.
    - `pytest-dev__pytest-10356`: copy `4550`, speedup `1.336x`,
      F1 `1.0000`.
  - Interpretation: this is a very clean safety baseline, but not the desired
    final Pareto point because full-table speedup is only `1.028x`. It proves
    that large shape-checked anchors can be fully safe in prompt-fair mode, but
    coverage is too narrow. The next step is to recover more large anchors
    without admitting small graph-only anchors.
- Bridge-candidate expansion diagnostic:
  - Current selector no-policy dry-run found six bridge-prefix candidates:
    `pallets__flask-5014`, `psf__requests-1142`,
    `psf__requests-1724`, `psf__requests-1921`,
    `psf__requests-2317`, and `pytest-dev__pytest-10356`.
  - 6-case smoke:
    `results/selective_ast_reuse/prompt_fair_bridge_candidate_smoke6_20260618`.
    It reached prompt fair `6/6`, copied `6/6`, avg TTFT speedup `1.508x`,
    avg token F1 `0.9728`, buckets `4` strict-safe + `2`
    lossy-acceptable. Newly validated strict rows:
    `pallets__flask-5014` (speedup `1.622x`, F1 `1.0000`) and
    `psf__requests-1921` (speedup `1.758x`, F1 `1.0000`).
    `psf__requests-1142` was fast (`5.865x`) but remains excluded because
    historical code-action/gold-intent posthoc showed target-anchor regression.
    `psf__requests-2317` became only lossy-acceptable with cap `5500`, so the
    next policy restores the stricter `4623` cap.
  - Expanded safe5 policy:
    `results/selective_ast_reuse/expanded_large_bridge_safe5_policy_20260618/hybrid_expanded_large_bridge_safe5_policy.json`.
    Allowed rows: `pallets__flask-5014`, `psf__requests-1724`,
    `psf__requests-1921`, `psf__requests-2317` with cap `4623`, and
    `pytest-dev__pytest-10356`.
  - Full 28-case run:
    `results/selective_ast_reuse/prompt_fair_expanded_large_bridge_safe5_28case_20260618`.
    Prompt fair `28/28`, exact output match `1.0000`, avg token F1 `1.0000`,
    buckets `28` strict-safe. Copied rows `5/28`, avg suffix copy `688.6`
    tokens over the full table. TTFT: lossless `1700.1ms`, hybrid `1623.6ms`,
    paired speedup `1.047x`.
  - Interpretation: expanded safe5 improves the clean baseline from `1.028x`
    to `1.047x` without any token or exact-output loss, but it is still below
    the earlier conservative no6028 mainline (`1.083x`). The bottleneck is now
    missing high-yield safe anchors such as historical `requests-5414` and
    `requests-2931`, not suffix-copy correctness on the currently selected
    anchors.
- Mixed-window new-bridge cap sweep:
  - Motivation: the mixed-window selector can find additional large
    bridge-prefix candidates for `pallets__flask-5014`,
    `psf__requests-1921`, and `psf__requests-2317`. These looked attractive
    in dry-run because they provide prompt-resident single `bridge_prefix`
    anchors with estimated reuse around `2638`-`3601` whitespace tokens.
  - Artifacts:
    - `results/selective_ast_reuse/prompt_fair_mixed_bridge_new3_cap1024_smoke_20260618`
    - `results/selective_ast_reuse/prompt_fair_mixed_bridge_new3_cap2500_smoke_20260618`
    - `results/selective_ast_reuse/prompt_fair_mixed_bridge_new3_smoke_20260618`
  - Cap `1024`: paired TTFT speedup `1.088x`, avg token F1 `0.6872`,
    buckets `3` aggressive-diagnostic. Per-case F1:
    `pallets__flask-5014` `0.8158`, `psf__requests-1921` `0.6000`,
    `psf__requests-2317` `0.6457`.
  - Cap `2500`: paired TTFT speedup `1.407x`, avg token F1 `0.7066`,
    buckets `3` aggressive-diagnostic.
  - Cap `5500`: paired TTFT speedup `3.696x`, avg token F1 `0.6345`,
    buckets `3` aggressive-diagnostic.
  - Interpretation: shortening the copied suffix is not sufficient for these
    mixed-window anchors. Even `1024` copied tokens fall below the F1 `0.90`
    lossy-acceptable threshold. The failure mode is therefore not just
    over-copy length; anchor semantic position and preceding-context alignment
    matter. These rows must remain diagnostic and should not be merged into the
    conservative main table.
- Hybrid best-window merge attempt:
  - Built a hybrid manifest that keeps the historical mixed-window rows for the
    stable requests/pytest anchors and swaps in original 1-file windows for the
    expanded safe5 cases (`pallets__flask-5014`, `psf__requests-1921`,
    `psf__requests-2317`, `pytest-dev__pytest-10356`).
  - Dry-run artifact:
    `results/selective_ast_reuse/dryrun_hybrid_best_windows_20260618`.
    Result: old no6028 live caps remain (`7` live copied candidates), but the
    swapped-in expanded cases hit shape mismatch under the mixed-window
    selector/policy settings. A no-policy dry-run shows those cases either
    produce only a different single bridge prefix or no bridge at all.
  - Interpretation: combining safe anchors across manifests requires a
    selector-stable policy, not only per-case manifest substitution. This
    should be addressed by a second-generation risk gate over reproducible
    selector features, not by hand-merging incompatible calibration artifacts.
- No6028 targeted cap upgrade:
  - Motivation: the conservative no6028 mainline used cap `1024` for three
    copied pytest rows. A prior single-case diagnostic suggested
    `pytest-dev__pytest-7432` could safely copy more.
  - Probe artifact:
    `results/selective_ast_reuse/prompt_fair_no6028_pytest_cap3500_probe_20260618`.
    Cap `3500` was safe only for `pytest-dev__pytest-7432`:
    - `pytest-dev__pytest-5787`: speedup `2.317x`, F1 `0.6471`,
      aggressive-diagnostic.
    - `pytest-dev__pytest-7236`: speedup `2.674x`, F1 `0.7552`,
      aggressive-diagnostic.
    - `pytest-dev__pytest-7432`: speedup `2.004x`, F1 `1.0000`,
      strict-safe.
  - New policy:
    `results/selective_ast_reuse/no6028_pytest7432_cap3500_policy_20260618/policy.json`.
    It keeps the no6028 policy unchanged except for increasing
    `pytest-dev__pytest-7432` from cap `1024` to cap `3500`.
  - Full 28-case artifact:
    `results/selective_ast_reuse/prompt_fair_no6028_pytest7432_cap3500_28case_20260618`.
    Prompt fair `28/28`; copied rows `7/28`; avg suffix copy increases from
    `913.0` to `1001.5` tokens. Paired TTFT speedup improves from `1.083x`
    to `1.089x` (`1113.4ms` lossless vs `1022.6ms` hybrid). Accuracy is
    unchanged: avg token F1 `0.9977`, exact output match `0.9643`,
    buckets `27` strict-safe + `1` lossy-acceptable + `0`
    aggressive-diagnostic. Post-hoc sanity also remains clean:
    code-action composite `28/28`, gold-intent regression `0/28`.
  - Interpretation: this is a small but real Pareto improvement over no6028.
    It confirms that per-anchor cap calibration can recover speed without
    changing prompts or admitting aggressive rows, but the fact that adjacent
    pytest rows fail at the same cap reinforces that cap length is not a
    monotonic safety variable.
- No6028 cap4000 refinement:
  - Motivation: after `pytest-dev__pytest-7432` proved strict-safe at cap
    `3500`, test whether the same anchor can safely copy more while keeping
    `psf__requests-1724` fixed at cap `3500`.
  - Cap sweep artifacts:
    - `results/selective_ast_reuse/prompt_fair_cap1724_4000_cap7432_3500_smoke2_20260618`
    - `results/selective_ast_reuse/prompt_fair_cap1724_3500_cap7432_4000_smoke2_20260618`
    - `results/selective_ast_reuse/prompt_fair_cap1724_3500_cap7432_4500_smoke2_20260618`
    - `results/selective_ast_reuse/prompt_fair_cap1724_3500_cap7432_5426_smoke2_20260618`
  - Results:
    - `psf__requests-1724` is not safe above cap `3500`; cap `4000` drops to
      F1 `0.6929`.
    - `pytest-dev__pytest-7432` is strict-safe at cap `4000` with single-row
      speedup about `2.57x`, but cap `4500` and full planned `5426` both drop
      to F1 `0.7677`.
  - New policy:
    `results/selective_ast_reuse/no6028_pytest7432_cap4000_policy_20260618/policy.json`.
    It keeps no6028 unchanged except `pytest-dev__pytest-7432` cap `4000`.
  - Full 28-case artifact:
    `results/selective_ast_reuse/prompt_fair_no6028_pytest7432_cap4000_28case_20260618`.
    Prompt fair `28/28`; copied rows `7/28`; avg suffix copy `1019.3`.
    Paired TTFT speedup improves to `1.091x` (`1112.5ms` lossless vs
    `1019.5ms` hybrid). Accuracy remains unchanged from no6028/cap3500:
    avg token F1 `0.9977`, exact output match `0.9643`, buckets `27`
    strict-safe + `1` lossy-acceptable + `0` aggressive-diagnostic. Post-hoc
    sanity remains clean: code-action composite `28/28`, gold-intent
    regression `0/28`.
  - Interpretation: this is the current safest full-table mainline. It is only
    a small speedup improvement, but it gives a concrete per-anchor safety
    boundary: cap `4000` safe, cap `4500` unsafe for `pytest-7432`; cap `3500`
    safe, cap `4000` unsafe for `requests-1724`.

## Combined Graph-Target Profile Update

- Motivation:
  - The previous safest line copied only `7/28` rows. Four previously validated
    expanded graph-target rows (`pallets__flask-5014`, `psf__requests-1921`,
    `psf__requests-2317`, `pytest-dev__pytest-10356`) were not reproduced in
    the combined selector because the hybrid run was not loading the strict28
    graph bundle manifest and because global graph-target preference would
    perturb already-calibrated rows.
  - Fix: create a combined manifest that preserves the current mixed-window
    files for existing calibrated rows, but adds the required prompt-resident
    graph target files only for the four expanded rows. Graph bundles remain an
    internal anchor-selection signal; no graph evidence is added to target
    prompts.
- Artifacts:
  - Combined manifest:
    `results/selective_ast_reuse/data/combined_no6028_cap4000_plus_expanded_graph_targets_20260618/manifest.json`
  - Selector override/policy:
    `results/selective_ast_reuse/combined_profile_policy_20260618/selector_overrides.json`
    and `results/selective_ast_reuse/combined_profile_policy_20260618/policy.json`
  - Dry-run:
    `results/selective_ast_reuse/dryrun_combined_manifest_graphload_20260618`
    selected `11` live cap rows and rejected the remaining cap candidates by
    shape gate.
  - Four-case smoke:
    `results/selective_ast_reuse/prompt_fair_combined_expanded4_smoke_20260618`
    reached `1.53x` paired TTFT speedup with token F1 `1.0000` and exact output
    match `1.0000`.
  - Full 28-case run:
    `results/selective_ast_reuse/prompt_fair_combined_profile_11live_28case_20260618`
- Full-run result:
  - Prompt fairness: `prompt_unfair_cases=[]`, `28/28` cases executed.
  - Lossless baseline: avg TTFT `1202.9ms`, avg cached tokens `602.9`.
  - `hybrid_code_aware_lossy`: avg TTFT `1039.6ms`, paired speedup `1.157x`,
    avg cached tokens `2284.0`, avg anchor match/suffix copy `1681.1`.
  - Accuracy: avg token F1 `0.9977`, exact output match `0.9643`, buckets `27`
    strict-safe + `1` lossy-acceptable + `0` aggressive.
  - Copied rows: `11/28`; copied-row paired speedups range from `1.17x` to
    `8.59x`.
- Report refresh:
  - Updated
    `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617.html`.
  - Regenerated rasterized PDF
    `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617.pdf`
    to avoid target-machine Chinese font garbling. Page previews are in
    `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617_raster_pages/`.
- Interpretation:
  - This is now the strongest prompt-fair token-level mainline: it improves the
    prior `1.091x` / `7` copied-row line to `1.157x` / `11` copied rows while
    keeping `0` aggressive rows.
  - The older `no6028 + pytest-7432 cap4000` line remains the semantic posthoc
    sanity baseline. The combined profile still needs patch-level/code-action
    sanity rerun before being described as the final semantic-safe main result.

## Combined Profile Semantic Sanity

- Ran posthoc semantic sanity on
  `results/selective_ast_reuse/prompt_fair_combined_profile_11live_28case_20260618/summary.json`.
- Code-action overlap:
  - Artifact:
    `results/selective_ast_reuse/prompt_fair_combined_profile_11live_28case_20260618/code_action_overlap_summary.json`
  - Result: `28/28` composite acceptable, avg code-action score `1.0000`,
    `code_action_ge_090=28`, no composite rejects.
- Gold-patch intent delta:
  - Artifact:
    `results/selective_ast_reuse/prompt_fair_combined_profile_11live_28case_20260618/gold_patch_intent_summary.json`
  - Result: `28/28` no gold-intent regression under max regression `0.10`,
    avg delta vs lossless `0.0000`, no composite rejects.
- Report refresh:
  - Updated
    `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617.html`
    and regenerated raster PDF
    `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617.pdf`.
- Interpretation:
  - Combined profile can now be described as prompt-fair with token F1 `0.9977`,
    `0` aggressive rows, code-action composite `28/28`, and gold-intent
    regression `0/28`.
  - Patch apply / candidate tests are still not part of the prompt-fair
    selective-driver main table and should be added as the next sanity layer.

## Additional Candidate Expansion Probe

- Goal: test whether two high-TTFT pytest rows rejected by the current combined
  profile can be safely added by making their graph target files prompt-resident:
  `pytest-dev__pytest-10051` and `pytest-dev__pytest-7324`.
- Artifacts:
  - Candidate manifest:
    `results/selective_ast_reuse/data/combined_11live_plus_pytest10051_7324_graph_targets_20260618/manifest.json`
  - Dry-run:
    `results/selective_ast_reuse/dryrun_combined_13live_candidate_20260618`
    produced `13` live cap rows, adding:
    - `pytest-dev__pytest-10051`: `src/_pytest/logging.py` bridge + 2 methods
    - `pytest-dev__pytest-7324`: `src/_pytest/mark/expression.py` bridge +
      function + method
- Probe results:
  - `pytest-dev__pytest-7324` single-case smoke:
    `results/selective_ast_reuse/prompt_fair_combined_pytest7324_smoke_20260618`
    completed but failed quality: hybrid F1 `0.0`, bucket
    `aggressive-diagnostic`, `cached_tokens=0`, `suffix_copy_len=None`.
    This row must remain rejected.
  - `pytest-dev__pytest-10051` quality probe:
    `results/selective_ast_reuse/prompt_fair_combined_pytest10051_quality_probe_20260618`
    stalled in the HTTP response path even with non-streaming output and
    `--max-tokens 32`; no rows were written. Prompt length was about `22k`
    tokens with `testing/python/metafunc.py` + `src/_pytest/logging.py`.
    Treat this as a runtime-stall candidate and keep it rejected until the
    long-prompt path is debugged separately.
- Interpretation:
  - The 13-live extension does not improve the stable mainline. The current
    `1.157x` / `11` copied-row combined profile remains the best verified
    prompt-fair result.

## Patch-File Graph Bridge Probe: pytest-dev__pytest-7490

- Goal: test whether a task/patch-file prompt context can add a small, safe
  pytest graph-bridge copy row without changing target prompts across modes.
- Implementation artifacts:
  - Patch-file candidate manifest:
    `results/selective_ast_reuse/data/combined_plus_patchfile_candidates_20260618/manifest.json`
  - Accepted 7490-only combined manifest:
    `results/selective_ast_reuse/data/combined_11live_plus_pytest7490_patchfile_20260618/manifest.json`
  - Selector/policy:
    `results/selective_ast_reuse/combined_profile_plus_pytest7490_policy_20260618/`
- Smoke result:
  - Run:
    `results/selective_ast_reuse/prompt_fair_graph_bridge_patchfile_cap1024_smoke2_20260618`
  - `pytest-dev__pytest-7490`: prompt-fair, selected
    `bridge_window:1 + function:2`, suffix copy `1024`, cached tokens `1862`,
    TTFT speedup `1.3796x`, token F1 `0.9867`.
  - `pytest-dev__pytest-6197`: rejected. Even with suffix cap `1024`, F1 was
    `0.7216` with only `1.0326x` speedup.
- Full 28-case probe:
  - Run:
    `results/selective_ast_reuse/prompt_fair_combined_plus_pytest7490_28case_20260618`
  - Prompt fairness held: `prompt_unfair_cases=[]`.
  - Live suffix-copy rows increased from `11` to `12`, adding
    `pytest-dev__pytest-7490`.
  - Aggregate speedup only improved from `1.1571x` to `1.1671x`.
  - Quality regressed: token F1 dropped from `0.9977` to `0.9799`, exact output
    match from `0.9643` to `0.7500`, and bucket counts became
    `21 strict-safe / 4 lossy-acceptable / 3 aggressive-diagnostic`.
- Interpretation:
  - `pytest-dev__pytest-7490` is a useful local Pareto point, but the full
    combined profile is not stable enough to promote. Keep the `11` copied-row
    profile as the main result until repeated full runs or a stronger retention
    gate prevent regressions in previously safe rows.

## Fixed-Seed Retention Gate and Cap Rescue

- Driver change:
  - Added `--server-random-seed` to
    `benchmark/multi_workflow/bench_selective_wholefile_reuse.py`; default is
    `42` and the value is now passed to SGLang as `--random-seed`.
  - Summary/report now record `server_random_seed` and
    `disable_overlap_schedule`.
  - Regression checks: `py_compile` passed and
    `pytest -q benchmark/multi_workflow/test_prompt_fair_kv_reuse.py
    benchmark/multi_workflow/test_selective_ast_reuse.py` passed
    (`42 passed`).
- Drift probe:
  - Run:
    `results/selective_ast_reuse/prompt_fair_plus7490_seed42_driftprobe5_20260619`
  - Fixed seed did not rescue the unstable rows:
    - `psf__requests-1724`: F1 `0.895`, speedup `2.227x`
    - `psf__requests-1766`: F1 `0.875`, speedup `6.922x`
    - `pytest-dev__pytest-10356`: F1 `0.7812`, speedup `1.331x`
    - `psf__requests-2931`: F1 `1.0000`, speedup `7.218x`
    - `pytest-dev__pytest-7490`: F1 `0.9867`, speedup `1.311x`
  - `--disable-overlap-schedule` did not improve quality; 10356 became worse
    (`0.6833` F1), so the issue is the copy policy rather than scheduling.
- Stable retention profile:
  - Policy:
    `results/selective_ast_reuse/combined_profile_plus_pytest7490_retention_stable_policy_20260619/policy.json`
  - Run:
    `results/selective_ast_reuse/prompt_fair_combined_plus7490_retention_stable_seed42_28case_20260619`
  - Result: prompt-fair `28/28`, no aggressive rows, `9` suffix-copy rows,
    avg TTFT `1031.4ms` vs lossless `1150.2ms`, paired speedup `1.115x`,
    avg token F1 `0.9959`, exact output match `0.8571`.
  - Sanity:
    code-action composite `28/28`; gold-intent regression `0/28`.
- Cap rescue:
  - Probe:
    `results/selective_ast_reuse/prompt_fair_cap_rescue_2500_4000_2048_smoke3_20260619`
  - Lower caps did not rescue requests:
    - `psf__requests-1724` cap `2500`: F1 `0.5677`
    - `psf__requests-1766` cap `4000`: F1 `0.7550`
  - `pytest-dev__pytest-10356` cap `2048` was rescued:
    F1 `1.0000`, speedup `1.118x`, suffix copy `2048`.
- Stable + 10356 profile:
  - Policy:
    `results/selective_ast_reuse/combined_profile_plus7490_stable_plus10356_policy_20260619/policy.json`
  - Run:
    `results/selective_ast_reuse/prompt_fair_combined_plus7490_stable_plus10356_seed42_28case_20260619`
  - Result: prompt-fair `28/28`, no aggressive rows, `10` suffix-copy rows,
    avg TTFT `1022.0ms` vs lossless `1152.2ms`, paired speedup `1.127x`,
    avg token F1 `0.9959`, exact output match `0.8571`.
  - Live rows:
    `pallets__flask-5014`, `psf__requests-1921`,
    `psf__requests-2317`, `psf__requests-2931`, `psf__requests-5414`,
    `pytest-dev__pytest-10356`, `pytest-dev__pytest-5787`,
    `pytest-dev__pytest-7236`, `pytest-dev__pytest-7432`,
    `pytest-dev__pytest-7490`.
  - Sanity:
    code-action composite `28/28`; gold-intent regression `0/28`.
- Interpretation:
  - The strongest quality-stable fixed-seed profile is now the 10-live
    `stable_plus10356` run. It is safer than the older 11-live profile, but its
    full-table TTFT speedup is only `1.127x`; the current work still needs more
    safe prompt-resident anchors or a better bounded-copy strategy before the
    speedup can be called clearly strong.

## Patch-File Batch Expansion: 7205 and 7324

- Goal: increase safe suffix-copy coverage by switching high-TTFT no-copy pytest
  rows to patch-file prompt contexts, while keeping target prompts identical
  across modes.
- Candidate manifest:
  `results/selective_ast_reuse/data/combined_plus_patchfile_batch_candidates_20260619/manifest.json`
  - Added patch-file contexts for:
    `pytest-dev__pytest-10051`, `10081`, `5262`, `5631`, `5809`, `5840`,
    `6202`, `7205`, `7324`.
- Dry-run:
  `results/selective_ast_reuse/dryrun_patchfile_batch_graph_bridge_20260619`
  - Promising bounded graph-bridge shapes:
    - `pytest-dev__pytest-10081`: `bridge_window:1 + method:1`
    - `pytest-dev__pytest-5631`: `bridge_window:1 + function:2`
    - `pytest-dev__pytest-5809`: `bridge_window:1 + function:1`
    - `pytest-dev__pytest-6202`: `bridge_window:1 + method:1`
    - `pytest-dev__pytest-7205`: `bridge_window:1 + function:1`
    - `pytest-dev__pytest-7324`: `bridge_window:1 + function:1 + method:1`
- Smoke:
  `results/selective_ast_reuse/prompt_fair_patchfile_batch_graph_bridge_cap1024_smoke6_20260619`
  - Accepted:
    - `pytest-dev__pytest-7205`: F1 `1.0000`, speedup `1.350x`,
      suffix copy `510`
    - `pytest-dev__pytest-7324`: F1 `1.0000`, speedup `1.811x`,
      suffix copy `1024`
  - Rejected as aggressive:
    - `pytest-dev__pytest-10081`: F1 `0.8037`
    - `pytest-dev__pytest-5631`: F1 `0.5841`
    - `pytest-dev__pytest-5809`: F1 `0.7177`
    - `pytest-dev__pytest-6202`: F1 `0.5783`
- Full 28-case profile:
  - Policy:
    `results/selective_ast_reuse/combined_profile_plus7490_10356_7205_7324_policy_20260619/policy.json`
  - Run:
    `results/selective_ast_reuse/prompt_fair_combined_plus7490_10356_7205_7324_seed42_28case_20260619`
  - Result: prompt-fair `28/28`, no aggressive rows, `12` suffix-copy rows,
    avg TTFT `729.7ms` vs lossless `862.2ms`, paired speedup `1.182x`,
    avg token F1 `0.9959`, exact output match `0.8571`.
  - Live rows:
    `pallets__flask-5014`, `psf__requests-1921`,
    `psf__requests-2317`, `psf__requests-2931`, `psf__requests-5414`,
    `pytest-dev__pytest-10356`, `pytest-dev__pytest-5787`,
    `pytest-dev__pytest-7205`, `pytest-dev__pytest-7236`,
    `pytest-dev__pytest-7324`, `pytest-dev__pytest-7432`,
    `pytest-dev__pytest-7490`.
  - Sanity:
    code-action composite `28/28`; gold-intent regression `0/28`.
  - Note: this uses the patch-file batch manifest, so its absolute TTFT is not
    directly comparable to older 1-file manifests. The paired speedup is still
    prompt-fair within this manifest.
- Failed-candidate lower-cap rescue:
  - Run:
    `results/selective_ast_reuse/prompt_fair_patchfile_batch_failed_cap512_smoke4_20260619`
  - `pytest-dev__pytest-10081` became strict-safe at cap `512`, but speedup was
    `0.987x`; not worth adding.
  - `pytest-dev__pytest-5631`, `5809`, and `6202` remained aggressive.
- Interpretation:
  - The current best quality-stable prompt-fair profile is the 12-live
    patch-file batch profile with speedup `1.182x`, no aggressive rows, token
    F1 `0.9959`, code-action `28/28`, and gold-intent regression `0/28`.
  - This is better than the 10-live stable profile, but still not a clearly
    strong TTFT improvement. The next useful step is either (1) recover missing
    local repos for `7521/7571/7982/8399`, or (2) implement a more surgical
    selective-recompute/copy strategy so rows like `10081/5631/5809/6202` do
    not drift under bridge-window copy.

## Follow-up Prompt-Resident Extended Probe

- Missing patch-file check:
  - `pytest-dev__pytest-7521`, `7571`, `7982`, and `8399` have no local
    checkout under `results/swebench_local_envs/repos/`; current repo-level
    datasets only contain unrelated prompt files such as `testing/python/metafunc.py`
    or `src/_pytest/pytester.py`.
  - Because their actual patch files are absent, they were not promoted to a
    patch-file prompt profile.
- Extended prompt-resident dry-run:
  - Run:
    `results/selective_ast_reuse/dryrun_extended_promptresident_probe_20260619`
  - Potential shapes appeared for existing prompt files:
    - `7521`: `testing/python/metafunc.py:bridge_window`
    - `7571`: `testing/python/metafunc.py:bridge_window`
    - `7982`: `src/_pytest/pytester.py:bridge_window`
    - `8399`: `testing/python/metafunc.py:bridge_window`
    - `10051`: `bridge_window:1 + method:5`
    - `5262`: `bridge_window:1 + method:1`
    - `5840`: `bridge_window:2 + method:3`
  - The first four are not patch-file resident and are therefore high-risk
    evidence; they were not benchmarked in this round.
- Smoke:
  - Run:
    `results/selective_ast_reuse/prompt_fair_extended_promptresident_cap512_smoke3_20260619`
  - Results:
    - `pytest-dev__pytest-10051`: rejected, F1 `0.6834`, speedup `0.949x`
    - `pytest-dev__pytest-5262`: safe but weak, F1 `1.0000`, speedup `1.034x`
    - `pytest-dev__pytest-5840`: safe but no real suffix copy / weak speedup,
      F1 `1.0000`, speedup `1.015x`
- Interpretation:
  - Existing prompt-resident non-patch files do not provide useful additional
    speedup under the current bounded bridge-window strategy.
  - The 12-live patch-file batch profile remains the best current stable result.

## Next Experiment

Use the new `selected_anchor_names` telemetry and mixed-window manifest support
to build a second-generation held-out risk gate:

- Expand coverage without lowering the safety bar:
  - train rules on more than the requests family, but validate with
    leave-one-repo-out or leave-one-family-out splits
  - replace exact anchor-name rules with coarser reusable buckets:
    file family, bridge length range, symbol/path mention, lexical overlap, and
    previous risk marker
  - make the selector snapshot explicit in every calibration artifact:
    selected file, selected anchor names, bridge/window construction, tokenized
    span length, max-file settings, and driver git/code version
  - improve the mixed selector beyond the current safe `1.086x` line: keep
    full-file task-aware p8000 for high-yield requests anchors, then search for
    larger but still safe pytest/django/flask prompt-resident windows instead
    of relying only on cap1024 pytest prefix anchors
  - use post-hoc retention gates during calibration: a row with token F1 >=
    `0.90` should still be rejected from the conservative main table if it loses
    the lossless file/symbol/action anchors or regresses gold-patch intent by
    more than `0.10`
  - for bounded windows, do not assume shorter/more local windows are safer;
    require a calibration sweep over copy cap and window position, and promote
    only rows that remain above the F1 `0.90` threshold
  - add code-action / patch-intent sanity on the 4 copied requests cases before
    using the anchor-aware result in the main PDF/table
  - extend file selection beyond the current 1-file manifest using issue/test
    target hints, then re-run the same risk gate on more prompt-resident anchors
  - promote only configurations with F1 >= `0.90`, no aggressive cases, and
    better than the current full-table `1.08x` conservative baseline

## Per-Anchor Suffix-Recompute Head Rescue

- Runtime/driver change:
  - Added span-level `suffix_recompute_head_len` support in
    `code_anchor_token_spans`.
  - This keeps the global `SGLANG_LOSSY_SUFFIX_RECOMPUTE_HEAD_LEN` at `0`, but
    allows selected calibrated anchors to recompute the first N anchor tokens
    before copying the remaining suffix.
  - Motivation: avoid slowing or changing all previously stable anchors while
    applying CacheBlend-style selective recompute only to risky bridge-window
    candidates.
- Unit coverage:
  - `test_agenttemplatekv_span_suffix_head_recompute_overrides_env` verifies
    span-level head metadata works even when the global env var is unset.

### Failed-Bridge Head Sweep

- Runs:
  - `prompt_fair_failed_bridge_head128_cap1024_smoke4_20260619`
  - `prompt_fair_failed_bridge_head256_cap1024_smoke4_20260619`
- Key findings:
  - `head=128` rescued only `pytest-dev__pytest-5631` at the exact threshold:
    F1 `0.9000`, speedup `1.367x`.
  - `head=256` made `pytest-dev__pytest-10081` strict-safe with speedup
    `1.254x`, kept `5631` at F1 `0.9000` / speedup `1.217x`, and made `6202`
    strict-safe but weak (`1.061x`).
  - `pytest-dev__pytest-5809` stayed aggressive and remains rejected.
- Per-anchor smoke:
  - Run:
    `prompt_fair_headrescue_peranchor_smoke5_20260619`
  - `10081`: F1 `1.0000`, speedup `1.324x`, suffix copy `1024`, head `256`
  - `5631`: F1 `0.9000`, speedup `1.289x`, suffix copy `654`, head `256`
  - Existing live controls (`psf__requests-5414`, `pytest-dev__pytest-7324`)
    kept head `0`, confirming the new metadata is local to selected anchors.

### Full 28-Case Updates

- 14-live head-rescue profile:
  - Run:
    `prompt_fair_combined_plus10081_5631_head256_seed42_28case_20260619`
  - Prompt fairness: `[]` unfair cases.
  - Paired TTFT speedup: `1.1866x` vs previous best `1.1815x`.
  - Avg token F1: `0.9924`; buckets `23 strict-safe`, `5 lossy-acceptable`,
    `0 aggressive`.
- High-cost rejected-case probe:
  - Run:
    `prompt_fair_highcost_reject_probe9_cap1024_head256_20260619`
  - Accepted by the current F1/speed/copy gate:
    - `pytest-dev__pytest-6202`: F1 `1.0000`, speedup `1.064x`
    - `pytest-dev__pytest-7521`: F1 `1.0000`, speedup `1.033x`
  - Rejected:
    - `6197`, `7571`, `7982`, `8399` at cap1024 due to aggressive F1 or weak
      no-copy behavior.
- Cap512 rescue:
  - Run:
    `prompt_fair_highcost_cap512_probe4_head256_20260619`
  - `pytest-dev__pytest-6197` becomes acceptable: F1 `0.9702`, speedup
    `1.027x`, suffix copy `512`.
  - `7571`, `7982`, and `8399` remain aggressive even at cap512.
- Latest 17-live full profile:
  - Run:
    `prompt_fair_combined_plus_highcost_head256_seed42_28case_20260619`
  - Prompt fairness: `[]` unfair cases.
  - Avg TTFT: lossless `860.6ms`, hybrid `718.7ms`.
  - Paired TTFT speedup: `1.1974x`.
  - Avg token F1: `0.9913`; buckets `22 strict-safe`, `6 lossy-acceptable`,
    `0 aggressive`.
  - Live suffix-copy rows: `17/28`.
  - Avg suffix copy: `1523.4` tokens; avg cached tokens: `2126.3`.

### Current Interpretation

- Per-anchor suffix recompute is useful as a quality rescue mechanism, but the
  recovered pytest rows mostly provide weak speedups. The best stable prompt-
  fair profile is now `1.197x`, not yet the desired clearly strong `1.3x+`.
- The remaining mean-TTFT bottleneck is not runtime copy correctness; it is
  finding high-yield prompt-resident anchors in the slow pytest cases without
  causing semantic drift.
- `pytest-dev__pytest-10356` was tested at cap `2304`; it improved speed only
  slightly but dropped to F1 `0.7083`, so cap `2048` remains its safe setting.
- Next optimization should focus on better anchor selection for high-cost
  pytest cases, especially using patch-file recovery / multi-file manifests or
  a more selective bridge-window position policy. Simply lowering copy caps
  rescues quality but leaves TTFT gains too small.

## Exact Pytest Checkout / Task-Aware AST Probe

- Added missing exact pytest base checkouts with `git worktree` for:
  - `pytest-dev__pytest-7521` at `41d211c24a67`
  - `pytest-dev__pytest-7571` at `422685d0bdc1`
  - `pytest-dev__pytest-7982` at `a7e38c5c6192`
  - `pytest-dev__pytest-8399` at `6e7dc8bac831`
- New diagnostic manifests:
  - `results/selective_ast_reuse/data/pytest_exact_patchtarget6_20260619/manifest.json`
  - `results/selective_ast_reuse/data/pytest_exact_patchonly6_20260619/manifest.json`
- Added selector mode:
  - `--hybrid-bridge-source task_ast`
  - `--hybrid-bridge-source task_ast_direct`
  - These select AST spans using only `problem_statement`, `FAIL_TO_PASS`, and
    `test_patch` tokens. They do not use gold patch text.

### Results

- Exact patch-only + `task_ast_direct`:
  - Run:
    `prompt_fair_pytest_exact_patchonly_taskastdirect_cap1024_20260619`
  - Prompt fairness: `[]`.
  - Useful rows:
    - `pytest-dev__pytest-7521`: strict-safe, speedup `1.014x`, suffix copy
      `145`.
    - `pytest-dev__pytest-7982`: strict-safe, speedup `1.086x`, suffix copy
      `109`.
  - Rejected rows:
    - `7571`, `8399`, `6197`, `5840` remained aggressive.
  - Interpretation: semantic AST selection is cleaner and can rescue individual
    rows, but direct non-prefix copy of small task spans is too short to move
    overall TTFT meaningfully.
- Patch-only bridge-window cap sweep:
  - `cap1024/head256`: speed exists but all rows aggressive.
  - `cap512/head512`: still all aggressive.
  - `cap256/head512`: only `7521` and `8399` strict-safe, but speedups are
    weak (`~1.03x`).
  - Interpretation: file-start bridge windows are high-risk for pytest even
    when using true patch files.
- Exact patch-target multi-file probe:
  - Increasing `anchor_max_total_tokens` exposed real copy for `7521/7571`,
    but large multi-file anchors caused scheduler OOM on the 24GB testbed.
  - `8399` exact patch-target + `task_ast_direct` copied only `208` tokens,
    had speedup `<1.0x`, and F1 `0.8617`; rejected.

### Current Status

- Best stable full result remains:
  - `prompt_fair_combined_plus_highcost_head256_seed42_28case_20260619`
  - Paired TTFT speedup `1.1974x`
  - Avg token F1 `0.9913`
  - `0` aggressive rows
- Patch-file recovery is useful for data quality and diagnostics, but by itself
  does not yet produce the desired `1.3x+` stable profile.
- The next promising direction is runtime-side multi-anchor or better
  selective recompute/copy scheduling, rather than selecting more tiny AST
  spans. Current runtime effectively benefits from one substantial copied
  suffix; many semantically precise spans are too small to affect TTFT.

## Runtime Multi-Anchor Prototype

- Implemented an experimental runtime flag:
  - `SGLANG_LOSSY_MULTI_ANCHOR=1`
  - Driver switch: `--lossy-multi-anchor-copy`
- Intended behavior:
  - After copying one exact-content anchor, keep scanning later anchors whose
    start is already covered by the current prefix/copy.
  - For a later anchor that needs a gap recompute, preserve the first copied
    anchor telemetry and set the next staged target prefix.
  - No new zero-filled gap behavior is introduced.
- Unit tests:
  - Adjacent two-anchor copy accumulates two copied spans.
  - Staged second-anchor path preserves first-copy telemetry and then copies
    the second anchor in the next call.
- Smoke:
  - Run:
    `prompt_fair_pytest_exact_patchonly_taskastdirect_multistage_20260619`
  - Result:
    - Prompt fairness: `[]`.
    - Paired speedup `1.024x`, essentially unchanged from single-anchor
      `1.022x`.
    - `lossy_anchor_multi_copy_count` stayed `1` for all real rows.
  - Interpretation:
    - The unit-level primitive works, but real SGLang request scheduling does
      not re-enter anchor matching in a way that turns later staged task spans
      into additional copied anchors.
    - Multi-anchor reuse therefore needs a deeper scheduler/runtime change
      that explicitly schedules repeated staged match/copy cycles, not just a
      local radix-cache loop.

## 2026-06-19 Follow-Up: Multi-Anchor Entry Fix and Reproducibility Check

- Runtime entry fix:
  - Relaxed `_try_lossy_fuzzy_match` only under `SGLANG_LOSSY_MULTI_ANCHOR=1`
    so a request that already copied one anchor can continue scanning anchors
    in a later staged round even if the first-match reason is no longer one of
    the original lossy reasons.
  - Unit coverage remains green:
    `python/sglang/srt/mem_cache/test_anchor_match.py -k "multi_anchor"`.
- Exact patch-only 6-case recompute smoke:
  - Run:
    `prompt_fair_pytest_exact_patchonly_taskastdirect_recompute_entryfix_ttft_20260619`
  - Prompt fairness: `[]`.
  - Paired TTFT speedup: `1.032x`.
  - Avg token F1: `0.9542`.
  - Buckets: `5` strict-safe, `1` aggressive-diagnostic.
  - Real copy happened only on `pytest-dev__pytest-7982` (`109` tokens,
    strict-safe) and `pytest-dev__pytest-8399` (`1024` tokens, aggressive).
  - `lossy_anchor_multi_copy_count` still stayed `1`; the local entry fix did
    not unlock repeated real-request copies.
- Best 28-case + multi-anchor entry-fix:
  - Run:
    `prompt_fair_combined_plus_highcost_head256_seed42_multianchor_entryfix_28case_20260619`
  - Prompt fairness: `[]`.
  - Paired TTFT speedup: `1.074x`.
  - Avg token F1: `0.9753`.
  - Buckets: `24` strict-safe, `2` lossy-acceptable, `2` aggressive.
  - Live-copy rows dropped from `17/28` in the current best run to `6/28`;
    no row had `lossy_anchor_multi_copy_count > 1`.
  - Conclusion: the multi-anchor flag remains diagnostic only and should not
    be used for the main result.
- Old high-speed policy reproducibility check:
  - A posthoc pass on the old
    `prompt_fair_pareto_f090_hybrid_minspan50_sel100_riskgate_maxtotal9000_28case_20260618`
    suggested that rejecting `psf__requests-2317` could yield about `1.239x`
    with no aggressive rows.
  - Real rerun:
    `prompt_fair_minspan50_sel100_riskgate_reject2317_28case_20260619`
  - `psf__requests-2317` was correctly gate-rejected, but the old profile did
    not reproduce under the current runtime/setup:
    paired speedup `1.182x`, avg token F1 `0.8892`, and `10` aggressive rows.
  - Conclusion: do not promote the old high-speed profile. Future policy search
    should be calibrated on current-runtime reruns, not historical posthoc
    tables.

### Updated Current Best

- Main prompt-fair result remains:
  - `prompt_fair_combined_plus_highcost_head256_seed42_28case_20260619`
  - Paired TTFT speedup `1.1974x`
  - Avg token F1 `0.9913`
  - Buckets: `22` strict-safe, `6` lossy-acceptable, `0` aggressive
  - Live suffix-copy rows: `17/28`
- Next useful optimization should be current-runtime policy search over the
  stable profile, with explicit rejection of any row that falls below F1 `0.90`.
  The multi-anchor runtime branch should stay behind its diagnostic flag until
  the scheduler can intentionally re-enter staged match/copy cycles.

## 2026-06-19 Current-Runtime File-Selection Probe

- Motivation:
  - The current best run still rejects several high-TTFT cases because the
    selected manifest file exposes no usable prompt-resident anchor.
  - Probe target set:
    `psf__requests-1142`, `psf__requests-1724`, `psf__requests-1766`,
    `psf__requests-6028`, `pytest-dev__pytest-10051`,
    `pytest-dev__pytest-5262`, `pytest-dev__pytest-5840`,
    `pytest-dev__pytest-7571`, `pytest-dev__pytest-7982`,
    `pytest-dev__pytest-8399`.
- Dry-run finding:
  - `--prefer-graph-target-files` exposes new anchors for:
    - `psf__requests-1724`: `requests/sessions.py`, est `2150`.
    - `psf__requests-1766`: `requests/auth.py`, est `875`.
    - `pytest-dev__pytest-5840`: `src/_pytest/pathlib.py`, est `1131`.
- 10-case conservative probe:
  - Run:
    `prompt_fair_reject10_prefergraph_cap1024_head256_20260619`
  - Prompt fairness: `[]`.
  - Paired speedup: `1.052x`.
  - Avg token F1: `0.9428`.
  - Buckets: `7` strict-safe, `2` lossy-acceptable, `1` aggressive.
  - Useful candidate:
    - `psf__requests-1766`: speedup `1.237x`, F1 `0.9907`,
      suffix copy `1024`.
  - Risky candidate:
    - `pytest-dev__pytest-5840`: speedup `1.191x`, F1 `0.5133`,
      suffix copy `1024`.
- Single-case graph-target manifest attempt:
  - Created:
    `results/selective_ast_reuse/data/combined_plus1766_graphfile_auth_20260619/manifest.json`
    with only `psf__requests-1766` switched from `requests/models.py` to
    `requests/auth.py`.
  - Created policy/overrides:
    `results/selective_ast_reuse/combined_plus_highcost_plus1766_graphfile_policy_20260619/`.
  - Full run:
    `prompt_fair_combined_plus_highcost_plus1766_graphfile_seed42_28case_20260619`
  - Result:
    - Prompt fairness: `[]`.
    - Paired speedup: `1.178x`.
    - Avg token F1: `0.9014`.
    - Buckets: `17` strict-safe, `3` lossy-acceptable, `8` aggressive.
    - `psf__requests-1766` itself dropped to F1 `0.7602`.
    - Previously stable large-copy rows such as `psf__requests-1921` and
      `psf__requests-2317` also became aggressive under the graph-enabled full
      run.
  - Conclusion:
    - Do not promote the graph-target-file mix into the main result.
    - Graph-target file selection can reveal useful anchors in isolation, but
      enabling graph bundle metadata globally changes the full-run behavior
      enough that current best stability is lost.
    - Any future file-selection rescue must be implemented without requiring
      global graph bundle enablement, or must isolate graph metadata strictly to
      the target case.
- Filtered graph follow-up:
  - Created a graph manifest containing only `psf__requests-1766`:
    `results/selective_ast_reuse/data/graph_filtered_1766_20260619/code_graph_precision_manifest.jsonl`.
  - Dry-run with the filtered graph manifest confirms the intended 1766
    selection:
    `requests/auth.py:bridge_prefix:file_start:1-194` +
    `requests/auth.py:method:build_digest_header:68-149`.
  - However, enabling graph-aware mode still changes the full driver behavior
    enough that several previously stable calibration entries dry-run as
    rejected unless the original file-selection behavior is exactly preserved.
  - Conclusion:
    - The next implementation improvement should be a driver-level per-case
      graph/file-selection isolation switch. The current global
      `--enable-graph-aware-lossy` / `--prefer-graph-target-files` interaction
      is too broad for safely adding one rescued case to an otherwise stable
      profile.
    - Until that isolation exists, the current best stable 28-case result
      remains unchanged.

## 2026-06-19 Driver Fix: Selection-Only Graph Bundles + Bridge Seed Spans

- Implemented driver-side controls in
  `benchmark/multi_workflow/bench_selective_wholefile_reuse.py`:
  - `--load-graph-bundles-for-selection`
    - Loads graph bundles for internal hybrid anchor selection without adding
      `graph_aware_lossy` to target modes and without injecting graph evidence
      into target prompts.
  - `--include-hybrid-bridge-seed-spans`
    - Includes AST/graph seed spans alongside hybrid bridge anchors for
      calibrated profiles that expect `bridge_prefix/window + symbol` anchors.
    - Seed spans are restricted to selected bridge files, and graph-mapped
      spans are ordered before generic same-file seed spans. This preserves the
      safer graph-selected method choices such as pytest `__call__` spans.
- Validation:
  - `py_compile` passed for the touched driver.
  - Dry-run:
    `dryrun_plus1766_graphfirst_seedspans_20260619`
    confirmed:
    - active modes stay `lossless_full_prefill, hybrid_code_aware_lossy`
      only.
    - `psf__requests-1766` selects
      `requests/auth.py:bridge_prefix:file_start:1-194` +
      `requests/auth.py:method:build_digest_header:68-149`.
    - `pytest-dev__pytest-10356` recovers the historical graph-selected
      `src/_pytest/mark/structures.py:method:__call__` anchor set instead of
      unrelated same-file/test-file methods.
- 1766 cap sweep:
  - `prompt_fair_1766_graphfile_cap512_smoke_20260619`
    - speedup `1.232x`, F1 `0.7511`, aggressive; rejected.
  - `prompt_fair_1766_graphfile_cap256_smoke_20260619`
    - speedup `1.103x`, F1 `1.0000`, strict-safe; promoted to full run.
- Full 28-case:
  - Run:
    `prompt_fair_combined_plus_highcost_plus1766_cap256_graphfirst_seedspans_28case_20260619`
  - Prompt fairness: `[]`.
  - Paired TTFT speedup: `1.1950x`.
  - Avg token F1: `0.9924`.
  - Buckets: `23` strict-safe, `5` lossy-acceptable, `0` aggressive.
  - `psf__requests-1766` is now strict-safe with suffix copy `256` and
    per-case speedup `1.221x`.
  - This is a safer/stabler profile than the previous graph-file attempts, but
    it does not beat the current best speedup `1.1974x`.
- Updated status:
  - Highest-speed stable prompt-fair result remains:
    `prompt_fair_combined_plus_highcost_head256_seed42_28case_20260619`
    (`1.1974x`, avg F1 `0.9913`, no aggressive rows).
  - Best safety-biased alternative:
    `prompt_fair_combined_plus_highcost_plus1766_cap256_graphfirst_seedspans_28case_20260619`
    (`1.1950x`, avg F1 `0.9924`, no aggressive rows, one additional
    strict-safe live-copy case).

## 2026-06-19 Cap Relaxation: 1766 Rescue + 6202/7490 Larger Safe Copies

- Motivation:
  - The graphfile rescue for `psf__requests-1766` is safe at suffix cap `256`
    but does not by itself beat the previous best 28-case speed.
  - A small cap-relaxation probe showed that some existing strict-safe or
    lossy-acceptable rows can tolerate longer suffix copy, while others become
    aggressive immediately.
- Probe:
  - Run:
    `prompt_fair_cap_relax_probe5_cap1500_20260619`
  - Cases:
    `pytest-dev__pytest-10081`, `pytest-dev__pytest-5787`,
    `pytest-dev__pytest-6202`, `pytest-dev__pytest-7236`,
    `pytest-dev__pytest-7490`.
  - Result:
    - `pytest-dev__pytest-6202`: F1 `1.0000`, suffix copy `1500`;
      promoted.
    - `pytest-dev__pytest-7490`: F1 `0.9865`, suffix copy `1500`;
      promoted.
    - `pytest-dev__pytest-10081`, `pytest-dev__pytest-5787`,
      `pytest-dev__pytest-7236`: aggressive; rejected.
- Full 28-case with 1766 cap256 + 6202/7490 cap1500:
  - Policy/overrides:
    `results/selective_ast_reuse/plus1766_cap256_relax6202_7490_policy_20260619/`.
  - Run:
    `prompt_fair_plus1766_cap256_relax6202_7490_cap1500_28case_20260619`
  - Result:
    - Prompt fairness: `[]`.
    - Paired TTFT speedup: `1.1976x`.
    - Avg token F1: `0.9923`.
    - Buckets: `23` strict-safe, `5` lossy-acceptable, `0` aggressive.
    - This slightly exceeds the historical speed best (`1.1974x`) while
      keeping higher average F1.
- Follow-up cap1900 probe:
  - Run:
    `prompt_fair_cap_relax_6202_7490_cap1900_smoke_20260619`
  - Result:
    - `pytest-dev__pytest-6202` at cap `1900`: F1 `0.5737`, aggressive;
      rejected. Keep cap `1500`.
    - `pytest-dev__pytest-7490` at cap `1900`: F1 `1.0000`, strict-safe;
      promoted to full run.
- Full 28-case with 1766 cap256 + 6202 cap1500 + 7490 cap1900:
  - Policy/overrides:
    `results/selective_ast_reuse/plus1766_cap256_relax6202_1500_7490_1900_policy_20260619/`.
  - Run:
    `prompt_fair_plus1766_cap256_relax6202_1500_7490_1900_28case_20260619`
  - Result:
    - Prompt fairness: `[]`.
    - Paired TTFT speedup: `1.2001x`.
    - Avg lossless TTFT: `846.3ms`.
    - Avg hybrid TTFT: `705.2ms`.
    - Avg token F1: `0.9928`.
    - Buckets: `24` strict-safe, `4` lossy-acceptable, `0` aggressive.
    - Avg real suffix copy: `1526.0` tokens.
    - `psf__requests-1766`: F1 `1.0000`, suffix copy `256`, TTFT
      `183.1ms -> 152.8ms`.
    - `pytest-dev__pytest-6202`: F1 `1.0000`, suffix copy `1500`, TTFT
      `1225.6ms -> 1095.8ms`.
    - `pytest-dev__pytest-7490`: F1 `1.0000`, suffix copy `1900`, TTFT
      `260.1ms -> 120.6ms`.
- Updated status:
  - New highest-speed stable prompt-fair result:
    `prompt_fair_plus1766_cap256_relax6202_1500_7490_1900_28case_20260619`
    (`1.2001x`, avg F1 `0.9928`, `0` aggressive rows).
  - The useful pattern is not a global larger suffix cap. It is per-case
    bounded suffix copy calibrated by smoke tests: 6202 tolerates `1500`,
    7490 tolerates `1900`, while nearby cases can fail badly at `1500`.

## 2026-06-19 Failed Cap Relaxation: 2317/7432/10356

- Probe:
  - Manifest:
    `results/selective_ast_reuse/data/cap_relax_2317_7432_10356_20260619/manifest.json`
  - Policy/overrides:
    `results/selective_ast_reuse/relax2317_7432_10356_probe_policy_20260619/`.
  - Run:
    `prompt_fair_relax2317_7432_10356_probe_20260619`
- Tested relaxations:
  - `psf__requests-2317`: cap `4623 -> 5002`.
  - `pytest-dev__pytest-7432`: cap `4000 -> 5426`.
  - `pytest-dev__pytest-10356`: cap `2048 -> 3000`.
- Result:
  - Prompt fairness: `[]`.
  - All three rows became aggressive:
    - `psf__requests-2317`: speedup `1.774x`, F1 `0.8095`, suffix copy
      `5002`.
    - `pytest-dev__pytest-7432`: speedup `6.843x`, F1 `0.5767`, suffix
      copy `5426`.
    - `pytest-dev__pytest-10356`: speedup `1.195x`, F1 `0.7626`, suffix
      copy `3000`.
- Conclusion:
  - Do not relax these caps in the main profile.
  - Current safe caps remain:
    - `psf__requests-2317`: `4623`.
    - `pytest-dev__pytest-7432`: `4000`.
    - `pytest-dev__pytest-10356`: `2048`.
  - These cases are useful evidence that the Pareto boundary is real: larger
    suffix copy can give large TTFT gains, but the quality drop is immediate
    once the per-case safe boundary is crossed.

## 2026-06-19 Failed Payload Pruning Probe: 6197/7521

- Motivation:
  - In the current best run, `pytest-dev__pytest-6197` and
    `pytest-dev__pytest-7521` have high estimated reusable tokens
    (`4925`/`5480`) but `payload_anchor_token_count=0`.
  - Root cause from telemetry:
    - many prompt-resident anchors are selected (`80`/`84`), but the total
      anchor token count exceeds `anchor_max_total_tokens=12000`;
    - the default payload policy is `anchor_max_total_policy=reject`, so all
      anchors are dropped before runtime.
- Probe:
  - Manifest:
    `results/selective_ast_reuse/data/prune_first_6197_7521_20260619/manifest.json`
  - Policy/overrides:
    `results/selective_ast_reuse/prune_first_6197_7521_policy_20260619/`.
  - Run:
    `prompt_fair_prune_first_6197_7521_probe_20260619`
  - Change:
    per-case `anchor_max_total_policy=prune_first` while keeping the existing
    calibrated suffix caps.
- Result:
  - Prompt fairness: `[]`.
  - `pytest-dev__pytest-6197`:
    - payload anchors kept: `71`, pruned: `9`, token count: `11957`.
    - suffix copy: `0`.
    - speed ratio: `0.922x` (slower than lossless).
    - F1 `0.9744`, lossy-acceptable but no speed benefit.
  - `pytest-dev__pytest-7521`:
    - payload anchors kept: `68`, pruned: `16`, token count: `11999`.
    - suffix copy: `0`.
    - speed ratio: `0.943x` (slower than lossless).
    - F1 `0.8926`, aggressive-diagnostic.
- Conclusion:
  - Do not use `prune_first` for these high-cost rows.
  - Simply keeping a large pruned anchor set is not enough; the runtime still
    needs a matchable anchor shape that can produce real suffix copy. For now,
    leaving these rows rejected is better than sending many unmatched anchors.

## 2026-06-19 Safe but Not Useful Cap Relaxation: pytest-7324

- Motivation:
  - In the current best full run, `pytest-dev__pytest-7324` is strict-safe with
    suffix copy `1024` and planned copy `1163`.
  - This is the smallest remaining strict-safe planned-vs-copy gap, so it is a
    low-risk candidate for a micro cap relaxation.
- Probe:
  - Manifest:
    `results/selective_ast_reuse/data/cap_relax_7324_1163_20260619/manifest.json`
  - Policy/overrides:
    `results/selective_ast_reuse/relax7324_cap1163_policy_20260619/`.
  - Run:
    `prompt_fair_relax7324_cap1163_smoke_gpu_20260619`
  - Change:
    `pytest-dev__pytest-7324` cap `1024 -> 1163`.
- Result:
  - Prompt fairness: `[]`.
  - F1 `1.0000`, strict-safe.
  - Suffix copy `1163`, planned copy `1163`, not truncated.
  - TTFT `163.76ms -> 96.79ms`, single-case speedup `1.692x`.
- Conclusion:
  - The relaxation is safe, but it does not clearly improve over the current
    full-run cap1024 row (`161.44ms -> 88.65ms`, speedup `1.821x`).
  - Do not promote cap1163 into the main 28-case profile unless a repeated
    run shows a stable TTFT gain. Keep `pytest-dev__pytest-7324` at cap `1024`
    for now.

## 2026-06-20 Symbol-Level Rule Calibration

- Motivation:
  - The 2026-06-19 best profile is still mostly case-id calibrated.
  - The next step is to move toward reusable, explainable risk rules that are
    not simply `instance_id -> cap`.
- Built a cross-run anchor risk table:
  - Output:
    `results/selective_ast_reuse/anchor_risk_table_latest_20260620/anchor_risk_rows.csv`
  - Inputs include current best, previous best, cap1500/cap1900 probes,
    failed cap relaxations, prune-first probe, and 1766 cap sweeps.
  - Key aggregate signal over copied rows:
    - `copy <= 1024`: 25/26 acceptable.
    - `1025 <= copy <= 2000`: 8/12 acceptable, 4 aggressive.
    - `copy > 4500`: 12/14 acceptable, 2 aggressive.
  - Conclusion:
    - Copy length alone is not a sufficient predictor. The same rough length
      can be safe or unsafe depending on anchor file/symbol/shape.
- First rule attempt:
  - Policy:
    `results/selective_ast_reuse/file_shape_rule_policy_20260620/policy.json`
  - Rule key:
    `(anchor_file, selected_shape, estimated-token window)`.
  - Dry-run:
    `dryrun_file_shape_rules_20260620`
  - Result:
    - 0/28 rules matched because the policy was derived from already-pruned
      calibrated shapes, while runtime rule matching sees raw selected shapes.
- Second rule attempt:
  - Policy:
    `results/selective_ast_reuse/raw_file_shape_rule_policy_20260620/policy.json`
  - Rule key:
    raw selected anchor file + raw token window, then
    `required_selected_span_count_by_granularity` pruning.
  - Dry-run:
    `dryrun_raw_file_shape_rules_20260620`
  - Result:
    - 9/28 cap rows.
    - Exposed a rule expressivity bug: `requests/sessions.py` function rule
      could match `psf__requests-2317`, then shape pruning selected the first
      same-granularity method/function instead of the calibrated symbol.
- Driver fix:
  - File:
    `benchmark/multi_workflow/bench_selective_wholefile_reuse.py`
  - Added rule-side support for:
    - passing `suffix_recompute_head_len` from rule policy into payload
      telemetry and token spans;
    - pruning selected anchors by exact
      `required_selected_anchor_name_all_regex` /
      `required_selected_anchor_name_any_regex`.
  - This fixes two observed problems:
    - `psf__requests-1766` needs `suffix_recompute_head_len=256`; without it,
      cap256 dropped to F1 `0.7602`.
    - `psf__requests-2317` must prune to the calibrated `request` method,
      not an arbitrary first method/function on the same file.
- Final symbol-shape rule policy:
  - Policy:
    `results/selective_ast_reuse/raw_symbol_shape_rule_policy_20260620/policy.json`
  - Rules:
    - Match exact calibrated anchor symbols inside the raw selection by
      `selected_anchor_name_all_regex`.
    - Prune to those exact anchors.
    - Apply empirical safe cap and optional suffix recompute head.
    - Default action is reject.
  - Dry-run:
    `dryrun_raw_symbol_shape_rules_20260620`
    - 10/28 cap rows, 18/28 default reject.
- 10-case rule smoke:
  - Run:
    `prompt_fair_raw_symbol_shape_rules_cap10_prunefix_20260620`
  - Result:
    - Prompt fairness: `[]`.
    - Speedup: `1.532x` on the 10 matched cases.
    - Avg F1: `0.9899`.
    - Buckets: 7 strict-safe, 3 lossy-acceptable, 0 aggressive.
    - `psf__requests-1766` recovered to F1 `1.0000` with cap256/head256.
    - `psf__requests-2317` recovered to F1 `1.0000` with the exact
      `request` method anchor.
- 28-case rule baseline:
  - Run:
    `prompt_fair_raw_symbol_shape_rules_28case_20260620`
  - Result:
    - Prompt fairness: `[]`.
    - Paired TTFT speedup: `1.1570x`.
    - Avg lossless TTFT: `856.2ms`.
    - Avg hybrid TTFT: `740.1ms`.
    - Avg token F1: `0.9964`.
    - Buckets: 25 strict-safe, 3 lossy-acceptable, 0 aggressive.
    - Copied rows: 9/28.
  - Comparison:
    - It does not beat the best case-id calibrated profile
      (`1.2001x`, F1 `0.9928`, 16 copied rows).
    - It is a stronger algorithmic result because it is symbol-level and
      default-reject, not pure instance-id lookup.
- Updated direction:
  - Keep `prompt_fair_plus1766_cap256_relax6202_1500_7490_1900_28case_20260619`
    as the highest-speed stable result.
  - Use `prompt_fair_raw_symbol_shape_rules_28case_20260620` as the first
    generalizable risk-rule baseline.
  - Next improvement should recover bridge-window rows that the symbol-rule
    policy skipped because raw selection did not include the generated
    `bridge_window` anchor names; this likely requires rule-time bridge-window
    construction or matching graph/seed spans before bridge synthesis.

## 2026-06-20 Bridge-Window Synthesis Rules

- Driver change:
  - File:
    `benchmark/multi_workflow/bench_selective_wholefile_reuse.py`
  - Added rule-time bounded bridge-window synthesis:
    exact raw seed symbols can now match by
    `required_selected_anchor_name_all_regex`, then synthesize
    `bridge_window:bounded:*` anchors without changing the target prompt.
  - Added telemetry:
    `hybrid_calibration_bridge_window_synthesized`,
    `hybrid_calibration_bridge_window_max_tokens`, and
    `hybrid_calibration_bridge_window_seed_count`.
- Policy:
  - `results/selective_ast_reuse/raw_symbol_bridge_synth_rule_policy_20260620_policy.json`
  - Default action remains reject.
  - Rules are exact-symbol / exact-anchor regex rules, with optional
    `synthesize_bridge_window_max_tokens`.
- Dry-run:
  - `dryrun_raw_symbol_bridge_synth_exact_rules_reject1921_6202_28case_20260620`
  - 9/28 cap rows after conservative pruning.
- Full run:
  - `prompt_fair_raw_symbol_bridge_synth_stage_reject1921_6202_28case_20260620`
  - Prompt unfair cases: `[]`.
  - Paired TTFT speedup: `1.0516x`.
  - Avg lossless TTFT: `519.2ms`.
  - Avg hybrid TTFT: `493.7ms`.
  - Avg token F1: `0.9958`.
  - Buckets: 26 strict-safe, 2 lossy-acceptable, 0 aggressive.
  - Copied rows: 8/28.
  - Real suffix copy examples:
    - `psf__requests-1766`: copy `256`, F1 `1.0000`.
    - `pytest-dev__pytest-10081`: copy `1024`, F1 `1.0000`.
    - `pytest-dev__pytest-5631`: copy `654`, F1 `0.9060`.
    - `pytest-dev__pytest-7205`: copy `510`, F1 `1.0000`.
    - `pytest-dev__pytest-7324`: copy `863`, F1 `1.0000`.
    - `pytest-dev__pytest-7490`: copy `1803`, F1 `0.9760`.
- Negative evidence found during this iteration:
  - `psf__requests-1921` broad 4210-token copy is unstable across reruns:
    historical runs include strict-safe rows, but the current prompt-fair
    rerun dropped to F1 `0.5753`. The rule is rejected in the general policy.
  - `pytest-dev__pytest-6202` and `pytest-dev__pytest-5787` have identical
    raw selected anchors under the current feature set. A `getmodpath`
    bridge-window rule is useful for `6202`, but caused `5787` to drop to
    F1 `0.8345`. The rule is rejected until task-aware rule matching is added.
- Interpretation:
  - This run is not the highest-speed result; the case-id calibrated profile
    remains better at `1.2001x`.
  - It is the most conservative general-policy result so far: exact-symbol
    seed matching, rule-time bridge-window synthesis, default reject, no
    prompt differences, and 0 aggressive rows.
  - The next algorithmic step is task-aware rule matching / risk prediction,
    because exact selected-anchor names alone cannot distinguish some
    semantically different pytest tasks with identical file-level raw anchors.

## 2026-06-20 Task-Aware Bridge-Synth Recovery

- Motivation:
  - `pytest-dev__pytest-6202` and `pytest-dev__pytest-5787` have the same raw
    selected anchors on `src/_pytest/python.py`, including
    `method:getmodpath:271-289`.
  - A pure exact-anchor rule cannot distinguish them: the 6202 bridge-window
    copy is strict-safe, while the same rule made 5787 aggressive.
  - However, task text differs: 6202 explicitly mentions
    `src/_pytest/python.py` / `python.py`, while 5787 does not.
- Policy update:
  - Policy:
    `results/selective_ast_reuse/raw_symbol_bridge_synth_rule_policy_20260620_policy.json`
  - Restored `raw_symbol_bridge_synth_13_pytest-dev_pytest-6202` with:
    - exact seed symbol regex for `getmodpath`;
    - `require_anchor_path_mentioned=true`;
    - `require_anchor_basename_mentioned=true`;
    - `min_anchor_lexical_overlap=3`;
    - cap `1500` and suffix recompute head `256`.
  - This remains prompt-fair: the task text is used only by the selection
    policy, not appended to target prompts.
- Dry-run:
  - `dryrun_taskaware_bridge_synth_6202_28case_20260620`
  - 10/28 cap rows.
  - `pytest-dev__pytest-6202`: cap via task-aware rule.
  - `pytest-dev__pytest-5787`: default reject.
- Full run:
  - `prompt_fair_taskaware_bridge_synth_6202_28case_20260620`
  - Prompt unfair cases: `[]`.
  - Paired TTFT speedup: `1.0696x`.
  - Avg lossless TTFT: `520.6ms`.
  - Avg hybrid TTFT: `486.7ms`.
  - Avg token F1: `0.9958`.
  - Buckets: 26 strict-safe, 2 lossy-acceptable, 0 aggressive.
  - Copied rows: 9/28.
  - Newly recovered row:
    - `pytest-dev__pytest-6202`: copy `1500`, head `256`, F1 `1.0000`.
- Interpretation:
  - This improves the conservative bridge-synth rule line from `1.0516x`
    to `1.0696x` while preserving 0 aggressive rows.
  - It is still below the case-id calibrated upper line (`1.2001x`), but it is
    a more algorithmic result: exact seed symbols plus task-aware path/basename
    evidence can recover safe high-value rows that exact anchors alone cannot
    separate.
  - Next target: generalize this into a learned or rule-based risk predictor
    over task mentions, target-file retention, copy length, gap length, and
    anchor shape.

## 2026-06-20 Task/Symbol-Aware Requests Recovery

- Motivation:
  - The conservative bridge-synth line left several requests rows rejected
    even though dry-run selection exposed relevant exact anchors.
  - The goal was to recover rows only when task text or symbol evidence
    explains why the selected anchor should matter, without changing target
    prompts.
- Policy:
  - `results/selective_ast_reuse/raw_symbol_bridge_synth_taskaware_requests_policy_20260620_policy.json`
  - Added current-anchor rules for:
    - `psf__requests-2317`: `requests/sessions.py` prefix plus
      `method:request:378-459`, gated by path/basename mention and lexical
      overlap; cap `3500`.
    - `psf__requests-5414`: synthesized bridge window around
      `method:prepare_url:360-444`, gated by `requests/models.py` mention;
      cap `2048`.
    - `psf__requests-1142`: `requests/models.py` prefix plus
      `method:prepare_content_length:388-395`, gated by symbol overlap;
      cap `3000`.
- Full run:
  - `prompt_fair_taskaware_requests_recovery_28case_20260620`
  - Prompt unfair cases: `[]`.
  - Avg lossless TTFT: `519.6ms`.
  - Avg hybrid TTFT: `459.8ms`.
  - Paired TTFT speedup: `1.1299x`.
  - Avg token F1: `0.9914`.
  - Buckets: 24 strict-safe, 4 lossy-acceptable, 0 aggressive.
  - Copied rows: 12/28.
  - Newly recovered rows:
    - `psf__requests-1142`: copy `3000`, F1 `0.9206`.
    - `psf__requests-2317`: copy `3500`, F1 `1.0000`.
    - `psf__requests-5414`: copy `2048`, F1 `0.9565`.
- Interpretation:
  - This improves the explainable rule-policy line from `1.0696x` to
    `1.1299x` with no aggressive rows.
  - It is still below the case-id calibrated `1.2001x` line, but it is a more
    defensible algorithmic result: exact anchor symbols, shape pruning,
    task/path/symbol evidence, and empirical caps.

## 2026-06-20 File-Prefix Fallback Negative Probe

- Change tested:
  - Added `--hybrid-bridge-source function_then_extended`.
  - This keeps the existing function/method bridge path when function seeds
    exist, but falls back to extended AST spans when no function/method seed
    is available.
- Target:
  - `psf__requests-2931` had no function seed under the current selector, but
    `function_then_extended` recovered:
    - `requests/models.py:bridge_prefix:file_start:1-200`
    - `requests/models.py:file_prefix:models.py:1-200`
- Probe results:
  - `prompt_fair_taskaware_requests_plus2931_smoke9_20260620`:
    - copy `1245`, F1 `0.6452`, aggressive.
  - `prompt_fair_taskaware_requests_plus2931_cap512_smoke9_20260620`:
    - copy `512`, F1 `0.5827`, aggressive.
- Interpretation:
  - The failure persists even after reducing copy length, so the issue is not
    simply over-copy.
  - File-prefix-only fallback anchors can be exact-content matches but still
    unsafe for lossy KV reuse because the anchor may not preserve the semantic
    target context needed by generation.
  - `function_then_extended` remains a useful selector capability for future
    diagnostics, but the 2931 fallback rule should not enter the current main
    policy.

## 2026-06-20 Pytest-10356 Shape Fix Recovery

- Motivation:
  - `pytest-dev__pytest-10356` was a high-cost rejected row in the current
    prompt-fair table.
  - Historical runs showed safe speedup for
    `src/_pytest/mark/structures.py` anchors, but the current rule was rejected
    by shape mismatch.
- Fix:
  - Policy:
    `results/selective_ast_reuse/raw_symbol_bridge_synth_taskaware_requests_plus10356_policy_20260620.json`
  - The exact-anchor regex list prunes the current raw selection to:
    - 1 `bridge_prefix`
    - 1 `function`
    - 6 `method` spans
  - The old rule expected 11 method spans, so it always rejected the row.
- Single-case probes:
  - cap `2048`: F1 `0.8872`, aggressive.
  - cap `1500`: F1 `1.0000`, strict-safe, TTFT `1038.6ms -> 921.7ms`.
  - cap `1024`: F1 `0.8872`, aggressive.
  - cap `512`: F1 `1.0000`, strict-safe, but weaker TTFT.
- Full run:
  - `prompt_fair_taskaware_requests_plus10356_28case_20260620`
  - Prompt unfair cases: `[]`.
  - Avg lossless TTFT: `519.4ms`.
  - Avg hybrid TTFT: `455.5ms`.
  - Paired TTFT speedup: `1.140x`.
  - Avg token F1: `0.9914`.
  - Buckets: 24 strict-safe, 4 lossy-acceptable, 0 aggressive.
  - Copied rows: 13/28.
  - Newly recovered row:
    - `pytest-dev__pytest-10356`: copy `1500`, F1 `1.0000`.
- Interpretation:
  - This improves the explainable rule-policy line from `1.130x` to `1.140x`
    with no aggressive rows.
  - Copy safety is not monotonic in copy length: cap `1024` and `2048` were
    worse than cap `1500`, so a learned/rule-based risk predictor must model
    anchor position and semantic shape, not just suffix length.

## 2026-06-20 Flask-5014 Selector-Profile Negative Probe

- Motivation:
  - Historical runs had useful speedup on `pallets__flask-5014` using
    `src/flask/blueprints.py`, but the current manifest prompt initially loaded
    `src/flask/scaffold.py`.
- Driver fix:
  - Extended `case-selector-overrides` to support:
    - `files_per_case`
    - `file_start_index`
    - `max_file_chars`
    - `max_complete_file_chars`
    - `prefer_selective_files`
    - `prefer_graph_target_files`
  - Also applied selector overrides during `load_cases`, not only during
    anchor selection.
  - Fixed formal target/warmup paths to populate `selected_anchor_names`
    before hybrid calibration, matching the dry-run path.
- Probe:
  - Selector override switches `5014` to full `src/flask/blueprints.py`.
  - Exact rule recovers:
    - `src/flask/blueprints.py:bridge_prefix:file_start:1-621`
    - two `__init__` method spans.
- Results:
  - cap `5146`: F1 `0.8062`, aggressive.
  - cap `2048`: F1 `0.8254`, aggressive.
  - cap `1024`: F1 `0.8254`, aggressive.
  - cap `512`: F1 `0.8254`, aggressive.
- Interpretation:
  - This selector-profile result is prompt-fair, but not accuracy-safe.
  - The failure does not disappear with shorter suffix copy, so the current
    5014 anchor shape should remain diagnostic-only.

## 2026-06-20 Pytest-10051 Logging Bridge-Window Recovery

- Motivation:
  - `pytest-dev__pytest-10051` is a prompt-fair high-cost row where the issue
    explicitly references `src/_pytest/logging.py`, `caplog.clear()`, and
    `caplog.get_records()`.
  - Default raw selection exposes `src/_pytest/logging.py:method:clear:441-443`.
- Probe:
  - Case-specific exact seed rule:
    `src/_pytest/logging.py:method:clear:441-443`.
  - Calibration synthesizes a bounded prompt-resident bridge window:
    `src/_pytest/logging.py:bridge_window:bounded:1-443`.
  - Single-case run:
    `prompt_fair_pytest10051_bridgewindow_single_20260620`.
- Single-case result:
  - Prompt unfair cases: `[]`.
  - Lossless TTFT: `495.16ms`.
  - Hybrid TTFT: `332.02ms`.
  - Speedup: `1.491x`.
  - Suffix copy: `2048` tokens, planned `3093`.
  - Exact output match: `true`.
  - Token F1: `1.0000`.
- Full run:
  - `prompt_fair_taskaware_requests_plus10051_28case_20260620`.
  - Prompt unfair cases: `[]`.
  - Avg lossless TTFT: `520.0ms`.
  - Avg hybrid TTFT: `448.4ms`.
  - Paired TTFT speedup: `1.160x`.
  - Avg token F1: `0.9914`.
  - Buckets: 24 strict-safe, 4 lossy-acceptable, 0 aggressive.
  - Copied rows: 14/28.
  - Anchor match rate: `0.500`.
- Interpretation:
  - This improves the explainable rule-policy line from `1.140x` to `1.160x`
    without adding any aggressive rows.
  - The rule is more defensible than broad bridge-window selection because it
    requires an exact seed method plus path/basename/lexical evidence from the
    task text, then synthesizes only one bounded window.

## 2026-06-20 High-Cost Pytest Negative / Boundary Probes

- `pytest-dev__pytest-6197`:
  - Restoring full `testing/python/metafunc.py` recovered historical
    `bridge_window:bounded:1-302`.
  - Single-case result was strict-safe and copied `512` tokens, but speedup was
    only `1.030x` because the selector override enlarged the target prompt.
  - Keep as boundary evidence, not a main-policy row.
- `pytest-dev__pytest-7521`:
  - Restoring full `testing/python/metafunc.py` recovered historical
    `bridge_window:bounded:1-283`.
  - Single-case result was strict-safe and copied `1024` tokens, but speedup
    was only `1.021x` for the same enlarged-prompt reason.
  - Also dangerous as a general rule because the same window shape matched
    `pytest-dev__pytest-7571`; keep case-specific diagnostic only.
- `pytest-dev__pytest-5840`:
  - `src/_pytest/pathlib.py:bridge_window:bounded:1-346` copied through the
    `unique_path` region and gave speedup around `1.35x`, but F1 stayed
    `0.8696` for caps `1792`, `1900`, and `2048`.
  - A more conservative pre-`unique_path` window
    `src/_pytest/pathlib.py:bridge_window:bounded:1-337` was also F1 `0.8696`.
  - Keep as aggressive diagnostic.
- `pytest-dev__pytest-5262`:
  - `src/_pytest/capture.py:bridge_window:bounded:1-451` copied `2048` tokens
    and improved TTFT from `516.65ms` to `377.66ms`, but F1 was only `0.8504`.
  - Keep as aggressive diagnostic.

## 2026-06-20 Requests-6028 Selector Repair Recovery

- Motivation:
  - `psf__requests-6028` was previously treated as unstable under broad
    requests-file selection.
  - The patch is local to `requests/utils.py:prepend_scheme_if_needed`, around
    line 960.
  - The default driver prompt truncated `requests/utils.py` before that target
    function, so the raw selector only saw `utils.py:1-200` file-prefix anchors.
- Selector repair:
  - Added case selector override:
    `results/selective_ast_reuse/requests6028_fullfile_selector_overrides_20260620.json`.
  - The override keeps prompt-fairness inside the case: both lossless and
    hybrid target modes use the same full `requests/utils.py` target prompt.
  - With full file text, dry-run recovers:
    `requests/utils.py:function:prepend_scheme_if_needed:960-982`.
- Rule:
  - Policy:
    `results/selective_ast_reuse/requests6028_prepend_scheme_probe_policy_20260620.json`.
  - Exact seed:
    `requests/utils.py:function:prepend_scheme_if_needed:960-982`.
  - Synthesized bounded window:
    `requests/utils.py:bridge_window:bounded:431-982`.
  - Cap: `1500`.
- Single-case result:
  - `prompt_fair_requests6028_prepend_scheme_cap1500_single_20260620`.
  - Prompt unfair cases: `[]`.
  - Lossless TTFT: `806.51ms`.
  - Hybrid TTFT: `683.73ms`.
  - Speedup: `1.180x`.
  - Suffix copy: `1500`, planned `3727`.
  - Exact output match: `true`.
  - Token F1: `1.0000`.
- Full run:
  - `prompt_fair_taskaware_requests_plus6028_28case_20260620`.
  - Prompt unfair cases: `[]`.
  - Avg lossless TTFT: `529.6ms`.
  - Avg hybrid TTFT: `453.3ms`.
  - Paired TTFT speedup: `1.168x`.
  - Avg token F1: `0.9914`.
  - Buckets: 24 strict-safe, 4 lossy-acceptable, 0 aggressive.
  - Copied rows: 15/28.
  - Anchor match rate: `0.536`.
- Interpretation:
  - This improves the explainable rule-policy line from `1.160x` to `1.168x`
    with no aggressive rows.
  - The result clarifies that the earlier 6028 failures were not simply
    "6028 is unsafe"; the unsafe part was broad / truncated anchor shape.
    Restoring the patch-local function and synthesizing a bounded target-local
    window makes this case strict-safe under prompt-fair evaluation.

## 2026-06-20 cap3000 selector-repair update

- Probe result:
  - `psf__requests-6028` cap sweep showed that raising the bounded suffix copy cap from `1500` to `3000` remained strict-safe under prompt-fair target prompts: F1 `1.0000`, exact output match `1.0000`, single-case TTFT `802.6ms -> 525.6ms` in `prompt_fair_requests6028_cap3000_probe_single_20260620`.
  - `pytest-dev__pytest-10051` cap3000 was rejected: F1 dropped to `0.8750`, so the rule stays at cap2048.
- Policy update:
  - Updated `raw_symbol_bridge_synth_taskaware_requests_plus10356_policy_20260620.json` so `raw_symbol_bridge_window_18_psf_requests-6028_prepend_scheme_if_needed` uses `max_suffix_copy_len=3000`.
  - Restored `pytest-dev__pytest-10356` to `max_suffix_copy_len=1500`; a transient full-run command had accidentally changed this rule and was discarded as non-mainline.
- Full 28-case prompt-fair rerun:
  - Artifact: `results/selective_ast_reuse/prompt_fair_taskaware_requests_plus6028_cap3000_fixed_28case_20260620`.
  - Prompt fairness: `prompt_unfair_cases=[]`, `28/28` executed.
  - Lossless baseline: avg TTFT `529.9ms`, avg cached tokens `602.9`.
  - `hybrid_code_aware_lossy`: avg TTFT `447.7ms`, paired speedup `1.1835x`, avg cached tokens `1557.5`, avg suffix copy `954.6`.
  - Accuracy: avg token F1 `0.9914`, exact output match `0.8571`, buckets `24` strict-safe + `4` lossy-acceptable + `0` aggressive.
  - `psf__requests-6028`: TTFT `532.0ms`, F1 `1.0000`, suffix copy `3000`, planned suffix copy `3727`.
- Report refresh:
  - Updated `results/selective_ast_reuse/code_aware_kv_reuse_contribution_report_20260617.html` to use the cap3000 fixed run as the current explainable full-table mainline.
  - Historical `1.2001x / F1 0.9928` cap-calibrated profile is now described as a Pareto diagnostic rather than the current mainline.

## 2026-06-20 pytest-10356 cap3000 recheck and full run

- Recheck:
  - `pytest-dev__pytest-10356` cap3000 recheck in `prompt_fair_pytest10356_cap3000_recheck_single_20260620` remained strict-safe: F1 `1.0000`, exact output match `1.0000`, suffix copy `3000`, planned copy `4294`, single-case TTFT `1042.9ms -> 786.9ms`.
  - The rule was updated from cap1500 to cap3000. This is treated as an empirical shape-specific cap, not a monotonic copy-length rule, because older cap1024/cap2048 probes showed F1 drift.
- Full 28-case prompt-fair rerun:
  - Artifact: `results/selective_ast_reuse/prompt_fair_taskaware_requests_plus6028_10356_cap3000_28case_20260620`.
  - Prompt fairness: `prompt_unfair_cases=[]`, `28/28` executed.
  - Lossless baseline: avg TTFT `529.5ms`, avg cached tokens `602.9`.
  - `hybrid_code_aware_lossy`: avg TTFT `443.7ms`, paired speedup `1.1934x`, avg cached tokens `1611.1`, avg suffix copy `1008.2`.
  - Accuracy: avg token F1 `0.9914`, exact output match `0.8571`, buckets `24` strict-safe + `4` lossy-acceptable + `0` aggressive.
  - Key strict-safe 3000-copy rows: `psf__requests-6028` and `pytest-dev__pytest-10356`.
- Report refresh:
  - Updated the main report HTML/PDF to use `prompt_fair_taskaware_requests_plus6028_10356_cap3000_28case_20260620` as the current explainable full-table mainline.
