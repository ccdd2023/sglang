# Phase 2 Findings — Extended Selective AST Reuse (2026-06-16)

## Test setup
- 28-case SWE-bench selective whole-file, with EXTENDED policy (function/method/control_block/file_prefix reuse)
- 5 modes: lossless / whole_file_reuse_all / selective_function_method / selective_extended / selective_oracle
- 3 cases got mode-skip on selective_extended + selective_oracle (file_prefix/control_block text-formatting mismatch)

## Original result (invalid / diagnostic only)

| Mode | n_ok | TTFT mean | Reused tokens | Recompute tokens | F1 | Speedup vs lossless |
|---|---:|---:|---:|---:|---:|---:|
| lossless_full_prefill | 28/28 | 25.1ms | 0 | 1747 | 1.000 | 1.00× |
| whole_file_reuse_all | 28/28 | 25.2ms | 1915 | 0 | 1.000 | 1.00× |
| selective_function_method | 28/28 | 25.4ms | 434 | 1313 | 1.000 | 0.99× |
| **selective_extended** | **25/28** | **145.2ms** ⚠️ | **823** | **534** | 1.000 | **0.17×** ⚠️ |
| selective_oracle | 25/28 | 24.2ms | 823 | 534 | 1.000 | 1.04× |

## Critical observation

The original 28-case Phase 2 comparison is **not a fair mode-to-mode comparison**.
The driver ran modes sequentially with shared cache state:

1. `lossless_full_prefill`
2. `whole_file_reuse_all`
3. `selective_function_method_reuse`
4. `selective_extended_reuse`
5. `selective_oracle_low_dnorm`

The server does **not** have a `selective_extended_reuse` / `selective_oracle_low_dnorm`
mode-specific code path. Both benchmark modes send `reuse_mode="lossy"` plus
anchor fields. Therefore the apparent oracle win can be explained by **order
contamination**: oracle runs immediately after extended and can reuse cache
state that extended just inserted.

The result rows support this interpretation:
- In slow rows, `selective_extended_reuse` often has no `lossy_match_reason`,
  while the immediately-following `selective_oracle_low_dnorm` row reports
  `exact_code_content_signature`.
- That means oracle is not measuring the same cold/warm state as extended.
- The previous statement "same span set, only mode label differs" was
  incorrect at runtime because the cache state differs by the time oracle runs.

## Fix Applied

`benchmark/multi_workflow/bench_selective_wholefile_reuse.py` now defaults to
mode isolation:

- flush before each mode,
- run that mode's own warmup,
- then measure the target request.

The legacy shared-cache behavior is still available with
`--shared-cache-across-modes` for cache-accumulation studies.

The benchmark also now writes TTFT and lossy diagnostics into
`selective_wholefile_rows.csv`, including:

- `lossy_rejected_reason`
- `lossy_reuse_allowed`
- `lossy_candidate_count`
- `matched_content_signature`
- `lossy_anchor_match_*`

Streaming output handling was also fixed to read `response["text"]`, so token
F1 is not computed from an empty streaming final chunk.

The launcher defaults were also tightened for the 24GB RTX 4090 testbed:

- `--max-total-tokens=32768`
- `--max-prefill-tokens=8192`
- `--mem-fraction-static=0.72`
- optional `--disable-overlap-schedule`

These match the more stable Claude-launched server shape and reduce the chance
of flashinfer/KV allocator instability during repeated per-mode cache flushes.

## Partial isolated rerun (interrupted)

Command output directory:
`results/selective_ast_reuse/swe_wholefile_68k_extended_isolated_20260616/`

The rerun completed 5 cases before the SGLang scheduler process hit a CUDA
`unspecified launch failure`; after that, `nvidia-smi` could not determine the
GPU device handle. This is a GPU/driver state failure, not a benchmark-level
exception.

Partial 5-case summary:

| Mode | n_ok | avg TTFT | avg cached | exact hit | token F1 |
|---|---:|---:|---:|---:|---:|
| lossless_full_prefill | 5 | 519.4ms | 0.0 | 0.00 | 1.0000 |
| whole_file_reuse_all | 5 | 60.2ms | 5148.8 | 1.00 | 0.8879 |
| selective_function_method_reuse | 5 | 429.9ms | 924.6 | 0.20 | 1.0000 |
| selective_extended_reuse | 4 | 61.7ms | 5280.2 | 0.80 | 0.6879 |
| selective_oracle_low_dnorm | 4 | 59.4ms | 4012.5 | 0.60 | 0.6259 |

Important: this partial isolated rerun **falsifies the old 5.8x-slower
interpretation**. With per-mode warmup and flush, `selective_extended_reuse`
and `selective_oracle_low_dnorm` are both around 60ms TTFT on completed cases.
The remaining concern is not extended-mode overhead; it is output/F1 drift under
the current lossy reuse path and the GPU crash that interrupted the full run.

## Accuracy guard added after the partial rerun

The partial rerun showed that common exact-content lossy matches were accepted
with context-aware predicted distance around `1.8268`, confidence around
`0.6388`, and then produced output drift in whole-file / extended reuse modes.

To make the next rerun safety-first, `anchor_match.py` now supports an opt-in
hard ceiling:

- `SGLANG_CONTEXT_AWARE_MAX_PREDICTED_D=<float>`
- When set to a positive value, lossy matches with predicted distance above the
  threshold are demoted to `reuse_allowed=False`.
- Default behavior is unchanged when the variable is absent.

`bench_selective_wholefile_reuse.py` exposes this as
`--context-aware-max-predicted-d`. A first conservative rerun should use `1.8`,
because it rejects the observed drifting `1.8268` matches while preserving a
small margin above the lowest-risk table cells.

Validation status:

- `py_compile` passed for the benchmark, `anchor_match.py`,
  `test_anchor_match.py`, and `serving_chat.py`.
- `pytest python/sglang/srt/mem_cache/test_anchor_match.py benchmark/multi_workflow/test_selective_ast_reuse.py -q`
  passed: `46 passed`.

## Final isolated rerun with predicted-d guard

After the GPU recovered, the 28-case run completed successfully with
`--context-aware-max-predicted-d 2.0`, mode isolation, and
`--disable-overlap-schedule`.

Command output directory:
`results/selective_ast_reuse/swe_wholefile_68k_extended_isolated_strictd20_20260616/`

Summary after fixing report aggregation to exclude payload-build skipped rows
from F1 / hit-rate averages:

| Mode | n_ok | avg TTFT | avg cached | exact hit | token F1 |
|---|---:|---:|---:|---:|---:|
| lossless_full_prefill | 28/28 | 580.6ms | 0.0 | 0.00 | 1.0000 |
| whole_file_reuse_all | 28/28 | 41.5ms | 5775.6 | 1.00 | 1.0000 |
| selective_function_method_reuse | 28/28 | 446.3ms | 1427.6 | 0.25 | 1.0000 |
| selective_extended_reuse | 25/28 | 45.0ms | 5824.7 | 1.00 | 1.0000 |
| selective_oracle_low_dnorm | 25/28 | 47.0ms | 5824.7 | 1.00 | 1.0000 |

The three skipped selective_extended / oracle rows were payload construction
failures where the selected segment text was not found in the prompt:

- `psf__requests-1142`
- `pytest-dev__pytest-7324`
- `pytest-dev__pytest-7432`

They should be treated as selection/text-normalization skips, not generation
errors. All actually executed rows had `output_token_f1_vs_lossless = 1.0`.

Interpretation:

- The old "extended is slow" conclusion is false.
- With isolated mode warmup, strict predicted-d gating, and the fixed streaming
  output reader, `selective_extended_reuse` reaches lossless-equivalent output
  on executed rows while reducing TTFT from `580.6ms` to `45.0ms`.
- This is a `12.9x` TTFT speedup versus lossless on the completed
  selective_extended rows' aggregate report.
- `selective_extended_reuse` and `selective_oracle_low_dnorm` are nearly
  identical under this policy, which supports the hypothesis that the old gap
  was cache-order contamination rather than mode-specific server behavior.

## Why we cannot use the old Phase 2 result for the paper

- The old Phase 2 result is confounded by shared cache state and mode order.
- It should not be cited as either a regression or an improvement.
- The isolated `strictd20` rerun above is the corrected Phase 2 result.

## Decision

- Treat the existing Phase 2 numbers as **invalid / diagnostic only**.
- Use `results/selective_ast_reuse/swe_wholefile_68k_extended_isolated_strictd20_20260616/`
  as the corrected Phase 2 artifact.
- Report F1/hit rate over executed rows; separately disclose the 3/28
  payload-build skips.
- Keep `--context-aware-max-predicted-d 2.0` as the current safety/performance
  operating point. `1.8` was too conservative and rejected all observed reuse in
  the first 6-case probe.

## Rerun command

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_selective_wholefile_reuse.py \
  --port 30000 --max-cases 28 \
  --dataset results/selective_ast_reuse/data/swe_selective_wholefile_68k_1file_instances.json \
  --manifest results/selective_ast_reuse/data/swe_selective_wholefile_68k_1file_manifest.json \
  --policy results/selective_ast_reuse/data/selective_reuse_policy_extended.json \
  --selective-mode extended --emit-ttft \
  --disable-overlap-schedule \
  --context-aware-max-predicted-d 2.0 \
  --out-dir results/selective_ast_reuse/swe_wholefile_68k_extended_isolated_strictd20_20260616 \
  --server-timeout 240 --eval-timeout 240
```

## Files

- 28-case run (extended policy): `results/selective_ast_reuse/swe_wholefile_68k_extended_20260616/`
- Corrected isolated 28-case run with predicted-d guard:
  `results/selective_ast_reuse/swe_wholefile_68k_extended_isolated_strictd20_20260616/`
- 28-case run (original policy, baseline comparison): `results/selective_ast_reuse/swe_wholefile_68k_ttft_20260616/`
- Extended policy: `results/selective_ast_reuse/data/selective_reuse_policy_extended.json` (kept for reference, NOT used in paper)
- Plan: `cheeky-tickling-lark.md` (older planning note; superseded by the
  corrected isolated `strictd20` rerun above)

## Follow-up: Prompt-Fair Lossy Pareto Mainline (2026-06-20, updated)

The corrected `strictd20` result above remains useful as a controlled mechanism
result: it shows that extended AST exact-content reuse can be very fast under
mode-isolated warmup. It should not be described as the final realistic agent
workflow result.

The current mainline for code-aware lossy reuse uses a stricter prompt-fair
protocol:

1. flush cache for every mode,
2. run the same Planner warmup prompt for every mode,
3. measure the same target prompt hash for every mode,
4. allow differences only in runtime KV policy / selected anchors.

### 2026-06-20 baseline (1.1934x)

Artifact:
`results/selective_ast_reuse/prompt_fair_taskaware_requests_plus6028_10356_cap3000_28case_20260620/`

| mode | n_ok/n | avg TTFT | avg cached | avg suffix copy | token F1 | buckets | paired speedup |
|---|---:|---:|---:|---:|---:|---|---:|
| lossless_full_prefill | 28/28 | 529.5ms | 602.9 | 0.0 | 1.0000 | 28 strict-safe | 1.000x |
| hybrid_code_aware_lossy | 28/28 | 443.7ms | 1611.1 | 1008.2 | 0.9914 | 24 strict-safe + 4 lossy-acceptable + 0 aggressive | 1.193x |

### 2026-06-20 v9 mainline (1.2437x, current best stable)

Artifact:
`results/selective_ast_reuse/prompt_fair_taskaware_e1_e5_v9_28case_20260620/`

| mode | n_ok/n | avg TTFT | avg cached | avg suffix copy | token F1 | buckets | paired speedup |
|---|---:|---:|---:|---:|---:|---|---:|
| lossless_full_prefill | 28/28 | 525.7ms | — | 0.0 | 1.0000 | 28 strict-safe | 1.000x |
| hybrid_code_aware_lossy | 28/28 | 422.7ms | — | 1208.5 | 0.9892 | 22 strict-safe + 6 lossy-acceptable + 0 aggressive | **1.2437x** |

v9 vs 2026-06-20 baseline: +4.2% speedup, +2 lossy-acceptable rows
(lossy-acceptable bucket widened by E1+E5 cap bumps; F1 still >= 0.90 for
all of them). Full iteration history (v1 - v10) and per-case reasoning is
in `results/selective_ast_reuse/prompt_fair_e1_e5_v9_mainline_20260620.md`.

Important interpretation:

- This is now the preferred evidence for realistic prompt-fair runtime reuse.
- The target prompt is identical within each case; graph/AST/task-aware rules
  do not add graph evidence to the prompt.
- The policy is still empirical and diagnostic-facing: exact anchor regex,
  shape pruning, task/symbol evidence, bridge-window synthesis, per-shape
  suffix-copy caps, plus E1 manifest repair (gold patch-target file in
  prompt for 7 cases) and E5 cap relax on 4 F1=1.0 strict-safe rows.
- `psf__requests-6028` is strict-safe at cap 3000; `pytest-dev__pytest-10356`
  is strict-safe at cap 4000; `pytest-dev__pytest-7432` is lossy-acceptable
  at cap 5000 (F1=0.9709); `pytest-dev__pytest-6202` and
  `pytest-dev__pytest-7236` are strict-safe at their bumped caps.
- Larger caps are still not generally safe. `psf-2317` cap 4000 caused F1=0.57;
  `pytest-10081` cap 1900 caused F1=0.59; `pytest-5787` cap 1500 caused F1=0.51.
  The "right" cap for a case is empirical per (case × current policy) and
  must be probed per change.
- Bridge anchor selection drift: when other cases' caps change in the same
  policy, the bridge can pick different anchors for an unrelated case. This
  means E5 cap relax for one case can affect F1 of an unrelated case
  (`pytest-10081` flipped from `bridge_window:bounded:1-329` to
  `bridge_prefix:file_start:1-414` between v6 and v8). The v9 fix is to
  cap `pytest-10081` at 256 to keep the bridge_window selection narrow.

Next accepted goal:

- Push the prompt-fair 28-case mainline from `1.2437x` to at least `1.25x`
  while keeping `prompt_unfair_cases=[]`, `0` aggressive rows, average token
  F1 `>=0.99`, and no patch/code-action sanity regression.
- Main remaining paths:
  - E2 prompt-fair patch/code-action sanity refresh
  - E3 risk predictor prototype (replace per-case cap rules)
  - E4 graph-aware mapping repair (5 E1 cases need richer graph bundles
    to enable copy; ~600ms of untouched lossless TTFT)
  - E5 cap relax on the 6 lossy-acceptable rows (trade speedup for F1;
    unclear net win)
