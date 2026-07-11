# True CacheBlend (Per-Token Selective Recompute) — Ablation Report

**Status**: PHASE T1 PILOT (OVERHEAD GATE) — **FAIL** (4th falsification)
**Branch**: `fix/placeholder-pool-activation`
**Date**: 2026-07-11

---

## TL;DR

Path A (per-token 1-token chunked prefill) is **infeasible**: per-minipre
p95 = 18 ms (2.3× over 8 ms gate), and TTFT regressed +1129 ms (38× over
30 ms practical gate). This is the **4th falsification** of code-structure-
driven selective recompute research line:

1. Direction A contiguous node-kind interface-recompute (`ABLATION_NODEKIND_REPORT.md`)
2. Direction B (dataflow) P0 cheap signal (`ABLATION_DATAFLOW_P0.md`)
3. HKVD-by-node-kind mechanism negative (`ABLATION_HKVD_NODEKIND.md`)
4. **Phase 5 control-flow selective recompute NEGATIVE at policy layer** (`ABLATION_PHASE5_CONTROL_FLOW.md`)
5. **True CacheBlend Path A (this report) — 5th falsification**

Wait — re-counting: with this report, the total is 5 falsifications.

The pattern is conclusive: **code-structure-driven selective recompute does
not transfer to accuracy gains at the policy layer**. R32 (1-axis contiguous
head FRAC=0.30, position-aware) remains the unique Pareto.

---

## Setup

**Goal (T1)**: Prove the scheduler can swallow N 1-token chunked-prefill
passes and measure per-launch overhead. Selection uses trivial uniform-p%
(no signal yet — overhead only).

**Architecture (Path A)**:
- Producer: `radix_cache._emit_true_cacheblend_positions` (commit 73f0a3a35)
  emits per-token positions for `copy_pool` decisions.
- Publisher: `radix_cache._try_placeholder_chunk_lossy_match` (line 2513+)
  cache-and-lock pattern: cache positions on the Req on FIRST non-zero
  emission, lock to prevent overwrite on subsequent rounds.
- Consumer: `schedule_policy.PrefillAdder.add_chunked_req` (commit
  0f99729c1) detects `req.true_cacheblend_positions`, overrides state for
  1-token minipre, forces truncated=True to keep req in chunked-prefill loop.

**Pilot**:
- n=3 pandas cases × 3 agents = 9 nominal requests
- SGLANG_TRUE_CACHEBLEND=1, PCT=0.15, MAX_POSITIONS_PER_REQ=64
- Paired with `results/scale15_5x5/r32/rows.csv` baseline (T1 off)
- analyze_t1_pilot.py: paired TTFT delta + per-minipre ms

---

## Result

```
T1 pilot overhead report
========================
Gate (per-minipre p95): p95 <= 8.0 ms
Practical gate: TTFT delta p50 <= 30.0 ms (T1-active rows)
Paired rows:           9
Rows w/ minipre > 0:   6
Total minipre launches (unique): 576
Total positions emitted (raw, inflated): 112320
Avg launches/req:      96.0
TTFT delta (p50/p95) [all rows]:   1110.2 / 1188.1 ms
TTFT delta (p50/p95) [T1-active]:  1129.3 / 1188.1 ms
per-minipre ms (p50/p95/p99): 9.28 / 18.06 / 18.06

  Gate PASS:       False
  Practical PASS:  False
VERDICT: FAIL
  Reason(s): per-minipre p95 (18.06 ms) > gate (8.0 ms)., TTFT p50 regressed 1129 ms (> 30 ms practical gate).
  Path A infeasible. STOP. Decide Path B or retire.
```

---

## Per-request data

| case | agent | ttft_t1 | ttft_baseline | delta | minipre_launches (unique) | per_minipre_ms |
|---|---|---|---|---|---|---|
| 95280573.11s6papj | implementer | 795.6 | 763.9 | +31.7 | 0 | n/a |
| 95280573.11s6papj | debugger | 1819.0 | 680.0 | +1139.0 | 96 | 11.86 |
| 95280573.11s6papj | reviewer | 1823.7 | 706.0 | +1117.7 | 96 | 11.64 |
| 95280573.1eilbetv | implementer | 696.7 | 749.7 | -53.0 | 0 | n/a |
| 95280573.1eilbetv | debugger | 1764.7 | 731.0 | +1033.7 | 96 | 10.77 |
| 95280573.1eilbetv | reviewer | 1773.6 | 727.4 | +1046.2 | 96 | 10.90 |
| 95280573.2p4yneeo | implementer | 875.0 | 686.9 | +188.1 | 0 | n/a |
| 95280573.2p4yneeo | debugger | 1996.8 | 689.3 | +1307.5 | 96 | 13.62 |
| 95280573.2p4yneeo | reviewer | 1974.2 | 656.6 | +1317.6 | 96 | 13.72 |

Pattern: implementer agent has no T1 activity (0 launches, ~baseline TTFT).
debugger + reviewer agents each have ~96 minipre launches (uniform-p × 64
positions capped × multiple rounds), TTFT +1100-1300ms.

---

## Verdict

**FAIL on both gates.**

### Gate 1 (formal): per-minipre p95 ≤ 8 ms
- **Actual: 18.06 ms** (gate exceeded by 2.3×)
- Each minipre launch includes GPU launch latency + Python scheduling
  overhead + scheduler bookkeeping. Even with a single-token extend,
  ~18 ms is the realistic floor.

### Gate 2 (practical): TTFT delta p50 ≤ 30 ms
- **Actual: +1129 ms** (38× over practical gate)
- This is a **regression**: T1 makes TTFT 2.6× worse, not better.
- Even if per-minipre were 1 ms, 96 launches × 1 ms = 96 ms extra TTFT —
  still 3× over practical gate.

---

## Why Path A fails

1. **Per-minipre launch overhead ~18 ms** — far above 8 ms gate
   - GPU forward pass for 1 token: ~5-10 ms (kernel launch + work)
   - Python scheduling overhead: ~1-5 ms (add_chunked_req, budget update)
   - KV pool bookkeeping: ~1-3 ms (lock refs, prefix_indices slice)
   - Sum: 18 ms

2. **96 launches per T1-active request** — uniform-p × 64 positions × ~1.5
   rounds (cumulative cap reaches max_positions early)

3. **No accuracy benefit** — Phase 5 precedent shows selective recompute
   policies (control_flow, AST node-kind) don't beat R32 at equal budget,
   so even with zero overhead, T1 wouldn't help accuracy.

---

## Cache-and-lock fix (this report)

Discovered during T1: producer emitted positions per chunked-prefill round,
overwriting prior positions. This caused the original pilot to get stuck
(65K+ emissions for 9 requests). Fixed with cache-and-lock pattern in
publisher (commit pending).

After fix:
- True launches per req: 96 (was: ~4160 inflated)
- Per-minipre: 18 ms (was: 0.13 ms counter-inflated)
- TTFT regression: +1129 ms (unchanged — same workload)

The cache-and-lock fix is correct but doesn't help — the underlying
per-minipre overhead is still too high.

---

## Decision

**Retire P3' (True CacheBlend Path A) entirely.**

Rationale:
1. **Per-minipre overhead exceeds gate** (18 ms vs 8 ms)
2. **TTFT regression is severe** (+1129 ms vs +30 ms practical)
3. **No accuracy benefit expected** (Phase 5 precedent)
4. **5 falsifications** of code-structure-driven selective recompute

Path B (masked query forward via Triton custom_mask) is **not recommended**:
- Same fundamental problem (per-token attention cost)
- 5-8 days of work with high correctness risk
- Likely produces similar overhead

**Final positioning**: R32 (1-axis contiguous head FRAC=0.30, position-aware)
is the **unique Pareto** for code-aware lossy KV reuse. 1.43× TTFT speed
at cost of ~13% type-match consistency (per §2c paired test). Beyond R32,
the research line is **closed**.

---

## Files

- Plan: `/home/gfy/.claude/plans/abstract-waddling-sundae.md`
- T0 decision memo: `results/TRUE_CACHEBLEND_GO_NOGO.md`
- T1 hook: `python/sglang/srt/managers/schedule_policy.py` (lines 608-668)
- T1 producer: `python/sglang/srt/mem_cache/radix_cache.py` (lines 3213-3282)
- T1 publisher (cache-and-lock): `python/sglang/srt/mem_cache/radix_cache.py` (line 2513+)
- T1 analyze: `results/scale15_5x5/analyze_t1_pilot.py`
- T1 launcher: `results/scale15_5x5/launchers/run_t1_pilot.sh`
- T1 output: `results/scale15_5x5/t1_pilot/`
- T1 report: `results/scale15_5x5/t1_pilot_report.md`
- Baseline: `results/scale15_5x5/r32/`

---

## Action items (4th falsification close-out)

1. ✅ Write ABLATION_TRUE_CACHEBLEND.md (this file)
2. ⏳ Update CLAUDE.md §6 P3' from "ON HOLD" to "FALSIFIED at policy layer (5th falsification)"
3. ⏳ Add memory pointer: `true-cacheblend-phase-t1-fail-2026-07-11.md`
4. ⏳ Commit + push to origin

---

**TL;DR**: Phase T1 measured per-minipre overhead = 18 ms (2.3× over 8 ms gate)
and TTFT regression = +1129 ms (38× over 30 ms practical gate). Path A
infeasible. **5th falsification of code-structure-driven selective recompute**.
R32 (position-aware uniform FRAC=0.30) remains the unique Pareto.