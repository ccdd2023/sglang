# T1 Pilot Overhead Report

**Generated**: analyze_t1_pilot.py
**Gate (formal)**: per-minipre p95 <= 8.0 ms
**Gate (practical)**: TTFT delta p50 <= 30.0 ms (T1-active rows)

## Inputs

- T1 rows: `results/scale15_5x5/t1_pilot/rows.csv`
- Baseline rows: `results/scale15_5x5/r32/rows.csv`

## Metrics

| Metric | Value |
|---|---|
| Paired rows | 9 |
| Rows with minipre launches > 0 | 6 |
| Total minipre launches (unique, cached) | 576 |
| Total positions emitted (raw, inflated) | 112320 |
| Avg launches/req | 96.0 |
| TTFT delta p50 (all) (ms) | 1110.2 |
| TTFT delta p95 (all) (ms) | 1188.1 |
| TTFT delta p50 (T1-active) (ms) | 1129.3 |
| TTFT delta p95 (T1-active) (ms) | 1188.1 |
| per-minipre ms p50 | 9.28 |
| per-minipre ms p95 | 18.06 |
| per-minipre ms p99 | 18.06 |

## Verdict

**FAIL**

### Reason

- per-minipre p95 (18.06 ms) > gate (8.0 ms)., TTFT p50 regressed 1129 ms (> 30 ms practical gate).

### Action

- Write ABLATION_TRUE_CACHEBLEND.md with full NEGATIVE report
- Update CLAUDE.md §6 P3' to `FALSIFIED at policy layer (4th falsification)`
- Add memory pointer: `true-cacheblend-phase-t1-fail-2026-07-11.md`
- Decision: Path B (5-8 days) OR retire P3' entirely
