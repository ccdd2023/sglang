# Direction A Equal-Budget Ablation: Node-Kind Interface-Recompute (2026-07-10)

> Decisive test of whether **code structure decides what to recompute** (deck slide 18,
> direction A) buys accuracy over uniform/position recompute at equal recompute budget B.
> Addresses R34's dismissal: R34 lacked an equal-budget ablation, so its effect was
> written off as a global FRAC bump. This run fixes that.

## TL;DR

<!-- Results filled 2026-07-10 after the 8-config run. -->
- **Question**: at equal recompute budget B, does setting the head-recompute boundary K
  from the AST interface (signature + docstring) per chunk beat uniform `frac × chunk_len`
  (R32) or position-stratified (R38b) on failure-type agreement (type_match)?
- **Verdict: FALSIFICATION (tentative - consistent direction, within CI noise).**
  nodekind (interface) scores 6.6% at c2_reuse=362; R32 at the same budget (~362, between
  f026@380 and f030@345) scores 9.8% - 3.3pp lower. nodekind_sig (signature-only, 3.3%)
  is worse. Both dominated by R32 on the Pareto front. Code structure (node-kind) does NOT
  buy accuracy over position (uniform head) at equal B.
- **Pareto frontier**: nodekind_sig, R32_f015, R32_f026, R32_f045 (nodekind dominated by
  R32_f026). Secondary: R32_f045 (11.5%) ~ lossless (10.7%) at 1.43x speedup - better
  operating point than R32@0.30 (9.8%).
- **Fire-rate guard**: node-kind fires on 100% of pool chunks (120/120 have a non-zero
  interface boundary) — NOT an R34-style no-op.

## 1. Hypothesis

A function's **interface** (signature + docstring) is the cross-context-sensitive region
(callers depend on it; it sits at the chunk-prefix boundary where HKVD deviation is
highest). The **body** is byte-stable implementation detail, safe to copy lossy. So
recomputing up to each chunk's *actual interface boundary* (not a uniform fraction) should
recover accuracy at lower recompute cost — i.e., at equal B, node-kind ≥ R32 type_match.

If the per-chunk adaptive boundary beats uniform fraction at equal B, code structure
buys accuracy (direction A confirmed). If it ties/loses, the contiguous-interface
form of direction A is falsified.

## 2. Why contiguous, not scattered (the contiguity constraint)

Deck slide 18's full direction A ("recompute signature + scattered control-flow tokens,
copy docstring + boilerplate") requires **non-contiguous** recompute (control-flow is in
the body). Exploration found sglang's prefill enforces a **contiguous cached prefix**
(`radix_cache.py:2432-2437`: a non-contiguous copy inflates `device_indices` past
`input_len` → negative `num_extend` → flashinfer crash). Scattered recompute would need
multi-round gap staging (`recompute_gap_chunk`), adding prefill rounds that eat TTFT —
self-defeating for a TTFT paper. The existing 2-round inter-chunk gap-prefill still hits
1.38×, but many intra-chunk control-flow gaps would not.

So direction A is landed as **contiguous interface-recompute**: recompute the interface
prefix `[chunk_start, interface_end)`, copy the body `[interface_end, chunk_end)`. This
fits the existing head-recompute executor, fires on every function/class chunk, and is a
clean (if weaker) test of "code structure decides the recompute boundary."

## 3. Implementation (P1)

- **`ast_chunker.py`**: `ChunkSpan` gains `signature_end_byte` / `interface_end_byte`
  (same byte coordinate system as `byte_start/byte_end`). `signature_end_byte` = start of
  the first body statement (end of the def/class header); `interface_end_byte` extends it
  across a leading docstring (`ast.Expr` → `ast.Constant` str) if present. Verified on
  sample functions; 100% of the 120 pool chunks have a non-zero boundary.
- **`radix_cache.py`**: new env `SGLANG_CHUNK_HEAD_RECOMPUTE_NODE_KIND=1` +
  `SGLANG_CHUNK_NODE_KIND_BOUNDARY={interface|signature}`. In `_build_chunk_plan`, when
  enabled, K = interface token count (via the per-span byte→token map) instead of
  `frac × chunk_len`. Default OFF; falls back to the frac path when the boundary is 0 or
  the map is missing. New counter `placeholder_chunk_pool_node_kind_k_count`.
- **`byte_to_tok` bug fix (pre-existing, uncovered)**: `_build_byte_to_token_map` silently
  returned `None` at runtime — sglang's HF `Qwen2Tokenizer` returns `tokenizers.Encoding`
  objects, the tuple unpack raised, the bare `except` swallowed it. So R32/R38b used the
  O(chunks×text) per-chunk re-encode fallback. Fixed by unwrapping `Encoding`. Verified
  `bisect_right(ends, byte)` == `len(encode(text[:byte]))` on chunk-start boundaries
  (0/120 diff) → R32/R38b offsets/accuracy unchanged, only faster. This fix is also why
  node-kind can fire (it needs the map); without it node-kind would silently fall back to
  frac = the R34 no-op trap.
- **Telemetry**: the new counter is wired through `scheduler_output_processor_mixin.py`
  AND `serving_chat.py` (two emission lists) AND the bench client — three places, mirroring
  `placeholder_chunk_pool_total_tokens_dense`. (Hit the "two emission paths" gotcha:
  adding to one place alone yields 0 in rows.csv.)
- **Budget** (`results/compute_nodekind_budget.py`): over the 120-chunk pool,
  `frac*_interface = 0.261`, `frac*_signature = 0.106` (the uniform FRAC that matches
  node-kind's total B). R32@0.26 is the equal-budget vertical-slice point.

## 4. Experimental setup

- **Configs** (n=15 diverse pandas cases × 5 agents, verdict mode, same pool
  `pandas_15case_v1`, same fixed `byte_to_tok` path):
  | config | head-recompute rule |
  |---|---|
  | lossless | no reuse (baseline accuracy + slowest TTFT) |
  | R32_f015 / f026 / f030 / f045 | uniform `frac × chunk_len` (Pareto sweep + frac*=0.26) |
  | R38b | position-stratified EARLY=0.60 / LATE=0.15 |
  | nodekind | K = signature + docstring (interface) |
  | nodekind_sig | K = signature only |
- **Metrics**: type_match (outputs.jsonl `output_text` via `score_r38.py`, fixed /n
  denominator — NOT /FAIL_rows); TTFT (rows.csv `ttft_ms`); reuse budget proxy =
  `c2_chunk_reused_tokens` (per-request; B = total − c2_reused and total is constant
  across configs → equal c2_reused ⇔ equal B); node-kind fire rate.
- **Fairness**: common-complete-cases subset (5 FAIL agents in all configs, same-pass) to
  cancel OOM row-drop differences. `placeholder_chunk_pool_total_tokens_dense` is
  cumulative + counts decisions not tokens, so unusable for per-request budget — c2_reused
  is the correct proxy.
- **Hard constraints honored**: `--disable-overlap-schedule --max-running-requests 1`
  (>3 cases); output to `results/`; F1 from `outputs.jsonl`; L3 k-NN OFF; reuse real
  (`codeaware_reused_tokens > 0`).

## 5. Results

<!-- Filled 2026-07-10. -->
Per-config (n=15, 5 agents, verdict; rows dropped to 60-61/75 by OOM on long cases):

| config | n | type_match | /n% | CI95 | TTFT | c2_reuse | fire |
|---|---|---|---|---|---|---|---|
| lossless | 75 | 8 | 10.7% | [3.6,29.1] | 1028 | 0 | - |
| R32_f015 | 61 | 4 | 6.6% | [0.0,18.2] | 707 | 465 | - |
| R32_f026 | 61 | 6 | 9.8% | [0.0,23.6] | 713 | 380 | - |
| R32_f030 | 61 | 6 | 9.8% | [0.0,26.0] | 715 | 345 | - |
| R32_f045 | 61 | 7 | 11.5% | [0.0,29.1] | 720 | 268 | - |
| R38b | 60 | 4 | 6.7% | [0.0,30.0] | 721 | 283 | - |
| nodekind | 61 | 4 | 6.6% | [0.0,15.0] | 715 | 362 | fires |
| nodekind_sig | 61 | 2 | 3.3% | [0.0,9.1] | 701 | 523 | fires |

- **R32 sweep is cleanly monotonic**: more recompute (lower c2_reuse) -> higher type_match
  (6.6% @465 -> 9.8% @380 -> 9.8% @345 -> 11.5% @268). Validates the budget->accuracy
  relationship and the c2_reused budget proxy.
- **Pareto frontier** (maximize reuse AND type_match): `nodekind_sig, R32_f015, R32_f026,
  R32_f045`. `nodekind` is dominated by `R32_f026` (more reuse 380>362 AND higher 9.8%>6.6%).
- **Vertical slice** (nodekind vs R32 at matched reuse = equal B): nodekind reuse=362,
  type_match=6.6%; R32 at reuse~362 is ~9.8% (between f026@380=9.8% and f030@345=9.8%).
  Delta = **-3.3pp**.
- **Common-complete-cases subset = 0**: no case has 5 FAIL agents in ALL 8 configs (OOM
  drops differ + some cases yield PASS/UNK). Cannot tighten via subset; fixed-/n% is the
  comparison. CIs overlap, so the delta is consistent in direction but not statistically
  decisive at n=15.

## 6. Verdict

<!-- Filled 2026-07-10. -->
**FALSIFICATION (tentative): contiguous node-kind interface-recompute does NOT beat
uniform R32 at equal recompute budget.** nodekind (6.6%) and nodekind_sig (3.3%) both sit
below the R32 sweep curve at their budgets; nodekind is Pareto-dominated by R32_f026. The
hypothesis (interface is accuracy-critical, body safe to copy) is **not supported** - the
data implies the opposite: the body (recomputed by R32's uniform head) is more
accuracy-critical than the docstring (recomputed by nodekind). Position-driven head
recompute beats code-structure-driven interface recompute at equal B.

Caveats: n=15 + OOM gives wide CIs (the -3.3pp delta is within CI overlap); the
common-complete-cases subset is empty (0 cases with 5 FAIL agents in all 8 configs). But
the R32 sweep monotonicity + nodekind consistently below the curve support the direction.

Per the plan's decision rule (ties/loses -> falsification, return to slide 18):
- Direction A (contiguous form) is falsified; the strong scattered form is TTFT-impractical
  (contiguity constraint, §2). So direction A as specified does not deliver an
  accuracy-preserving code-structure lever.
- This **reinforces CLAUDE.md §2a**: the method is a speed optimization (Pareto trade), not
  accuracy-preserving - code-structure-driven recompute selection does not beat position-driven.
- Next levers (slide 18): **dataflow (B)** - recompute only tokens referencing an
  upstream-changed symbol (true novelty, needs cross-chunk dependency analysis); or
  **task-cycle (C)** - AST-diff across agent iterations. Both harder than contiguous.
- **Secondary positive finding**: R32_f045 (FRAC=0.45) reaches 11.5% ~ lossless (10.7%) at
  1.43x speedup - a better operating point than R32@0.30 (9.8%). With the byte_to_tok fix,
  R32 is also faster (715ms vs old 745ms). Worth a follow-up confirmation run.

## 7. R34 lessons (all addressed)

1. Gate on AST node kind (always present), not type annotation (rare) — ✓ node-kind fires
   on 100% of pool chunks.
2. Equal-budget ablation (R34 lacked it → dismissed as global bump) — ✓ this run.
3. Signal verification folded into the decisive test (node-kind == uniform @ equal B is
   itself the falsification) — ✓ (user chose to skip the P0 proxy pre-check).

## 8. Limitations

- **Contiguous only**: the strong "recompute scattered control-flow" claim is
  TTFT-impractical (contiguity constraint). This tests the weaker "code structure decides
  the boundary" claim.
- **OOM**: long cases crash (rc=-9, system RAM), dropping to ~60-61/75 rows per config;
  the common-complete-cases subset mitigates but reduces n.
- **Wide CIs**: n=15 with OOM gives noisy type_match (CIs span ~15-26pp); the equal-B
  delta must be read against this noise.
- **verdict accuracy ≠ task capability** (per CLAUDE.md): type_match measures failure-type
  agreement on verdict tasks, not code-gen correctness.

## 9. Files

- Code: `ast_chunker.py`, `radix_cache.py`, `serving_chat.py`,
  `scheduler_output_processor_mixin.py`, `bench_kvcomm_ttft_stress.py`
- Budget: `results/compute_nodekind_budget.py` + `results/codebase_kv/pandas_15case_v1/nodekind_budget.json`
- Launchers: `results/scale15_5x5/launchers/run_{nodekind,nodekind_sig,r32_frac}.sh`,
  `results/scale15_5x5/run_all_ablation.sh`
- Analyzer: `results/scale15_5x5/analyze_ablation_nodekind.py`
- Outputs: `results/scale15_5x5/{lossless,r32,r32_f015,r32_f026,r32_f045,r38b,nodekind,nodekind_sig}/`
