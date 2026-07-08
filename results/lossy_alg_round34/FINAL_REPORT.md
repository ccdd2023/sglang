# Round 34 — Type-Annotation-Aware Chunk Prefill (RETIRED 2026-07-08)

## Hypothesis (Direction B from R33 follow-up)

When QUERY chunk has Python type annotations (`def foo(x: pd.DataFrame) -> bool`),
raise `SGLANG_CHUNK_HEAD_RECOMPUTE_FRAC` to a sig-gated value (default 0.50).
Rationale: typed chunks have STABLE interface but the BODY is what is reused —
and the body's cross-context attention likely carries the most stale-KV risk.
Heavy-head-recompute recovers accuracy for these. Untyped chunks keep base FRAC
(R32 default 0.30).

**This was the only "use coding structure" direction that does NOT modify the
agent's prompt** — pure reuse decision policy.

## Result: NEGATIVE / NULL

| Metric | lossless | R34 control (=R32) | **R34 sig_gated** | Δ vs control |
|---|---|---|---|---|
| **Failure-type agreement** | 38.5% | **41.7%** | **33.3%** | **-8.4 pp (regressed)** |
| Verdict FAIL accuracy | 52.0% | 48.0% | 48.0% | 0 |
| PASS% | 48% | 28% | 28% | 0 |
| avg TTFT (reusers, ms) | ~954 | 711.6 | 734.4 | +22.8 ms (+3.2%) |
| p50 TTFT (ms) | ~955 | 755.1 | 758.0 | +3 ms |
| avg codeaware_reused | 0 | 333.7 | **294.0** | -39.7 (-12%) |
| avg radix_prefix | ~109 | 172.6 | 228.2 | +55.6 (+32%) |
| Fair A/B speedup | 1.000× | ~1.28× | **0.992×** | regressed (parity violated) |
| Fair A/B parity | OK | OK | **VIOLATED** (radix_delta=69) | — |

## Why it failed (honest analysis)

1. **pandas 5-case is untyped code**. The precomputed pool covers
   `pandas/core/interchange/{buffer,column,dataframe}.py` — pandas 0.x-era
   interchange protocol code, which has **no Python type annotations on
   any function**. The helper `_extract_type_signature_string` returns
   `""` for every chunk in the pool that grep could find, so the gate
   effectively never fires on the relevant candidates. The "treatment"
   config therefore devolves to a global FRAC bump from 0.30 → 0.50
   (because the few chunks that DO have `-> None` return annotations
   still trigger the gate, and the cumulative effect is too many chunks
   getting 0.50 vs only a subset getting 0.30).

2. **Higher FRAC regresses accuracy when applied to too many chunks**.
   R32 measured FRAC=0.15 → 31.2%, FRAC=0.30 → 41.7%. Going beyond 0.30
   starts to hurt because too much of the chunk becomes "fresh prefill"
   instead of "reuse with cross-context attention", which means the
   model sees less of the cached chunk's content and more arbitrary
   in-context rewrites. The Pareto point is ~0.30, not 0.50.

3. **TTFT +3.2% for +0% accuracy** — strictly worse than R32.

4. **Fair A/B parity violated**: radix_prefix jump from 172.6 → 228.2
   (+55 tokens) between control and treatment is large enough that
   12.5% of per-agent radix prefixes exceed the 15% delta threshold.
   This is NOT an algorithm effect — it's the radix cache picking up
   extra prefill state because head-recompute changes alignment. But it
   means the speedup claim isn't trustworthy even if we'd seen better
   accuracy (which we didn't).

## What the helper still gives us

- `_extract_type_signature_string` in `ast_chunker.py` is **correct and
  tested**: it extracts `def name(arg: type, ...) -> return_type:` for
  FunctionDef/AsyncFunctionDef and `class Name(base1, base2)` for
  ClassDef, returning "" on parse failure or missing annotations.
- For codebases with FULL annotations (modern pandas 2.x, pydantic
  models, FastAPI handlers, mypy strict projects), this gate WOULD
  fire, but the empirically-best FRAC remains at ~0.30 (R32 Pareto)
  — so the gate would still need a more sophisticated threshold
  (e.g., FRAC × function_complexity, not FRAC × annotation_count).

## Decision: RETIRED

- Helper kept in `ast_chunker.py` for future R35+ iterations on
  annotated codebases (e.g. SWE-bench fix-mode on django/astropy, which
  have more annotations).
- Invocation site retired in `radix_cache.py` with a comment.
- Launchers kept disabled for reproducibility re-check.

## Verdict scoring

```
config                      n  pass%  fail%  FAIL_acc%  type_agree%
lossless                   25  48.0%  52.0%      52.0%        38.5%
r34_control                25  28.0%  48.0%      48.0%        41.7%
r34_sig_gated              25  28.0%  48.0%      48.0%        33.3%   ← regressed -8.4 pp
```

## Files

| Artifact | Path |
|---|---|
| Helper (kept) | `python/sglang/srt/mem_cache/ast_chunker.py::_extract_type_signature_string` |
| Invocation (RETIRED) | `python/sglang/srt/mem_cache/radix_cache.py:_build_chunk_plan` |
| Baseline output (= R32) | `results/lossy_alg_round34/r34_control_verdict/` |
| Treatment output | `results/lossy_alg_round34/r34_sig_gated_recompute_verdict/` |
| Lossless reference | `results/lossy_alg_round34/lossless_verdict/` |
| Fair A/B | `results/lossy_alg_round34/FAIR_AB_REPORT.md` |
| Launchers (disabled) | `results/lossy_alg_round34/launchers/run_r34_*.sh` |
| Verdict scorer | `results/lossy_alg_round34/scripts/score_r34.py` |
