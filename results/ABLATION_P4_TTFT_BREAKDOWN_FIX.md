# P4 - R40 TTFT-breakdown zmq pickle boundary FIX

**Date**: 2026-07-10
**Status**: ✅ **FIXED - all 6 previously-blocked TTFT-breakdown fields now cross the scheduler→tokenizer boundary**
**Files changed**:
- `python/sglang/srt/observability/req_time_stats.py` (`SchedulerReqTimeStats.__getstate__`)
- `python/sglang/srt/mem_cache/radix_cache.py` (NameError in head_recompute setters)
**Verification**: `results/verify_p4_getstate_fix.py` (5 unit tests PASS) + 3-case bench (rows.csv 6/6 fields non-zero)

---

## TL;DR

R40 Phase 1 wired 8 TTFT-breakdown timing fields through the request lifecycle. Only `tokenize_ms` worked end-to-end; the other 6 (radix_prefix_ms, chunk_plan_ms, copy_ms, gap_prefill_ms, head_recompute_early_ms, head_recompute_late_ms) stayed 0.0 in rows.csv. Two compounding bugs:

1. **`SchedulerReqTimeStats.__getstate__` allowlist** (req_time_stats.py:642-655): an explicit field allowlist omitted the 6 R40 fields, AND a `if not self.enable_metrics: return {}` gate dropped ALL fields (bench runs without `--enable-metrics`). So zero fields crossed the zmq pickle boundary. `tokenize_ms` worked only because it is derived from timestamps that live entirely in the API-server process and never cross.

2. **`NameError: name 'plan' is not defined`** (radix_cache.py:3963): the head_recompute_early/late setters referenced `plan.copy_count`, but `plan` is not a parameter or local of `_try_placeholder_chunk_lossy_match`. The `except Exception: pass` swallowed the NameError, so set_head_recompute_early/late_ms never ran (set_copy_ms/set_gap_prefill_ms ran first and survived, which is why those 4 appeared to work after fix #1 alone).

**Post-fix (3-case R32_f015 bench, 15 rows)**:

| field | pre-fix (61 rows) | post-fix (15 rows) |
|---|---|---|
| ttft_tokenize_ms | 15/15 (always worked) | 15/15 |
| ttft_radix_prefix_ms | **0/61 BLOCKED** | **15/15 ✓** |
| ttft_chunk_plan_ms | **0/61 BLOCKED** | **12/15 ✓** |
| ttft_copy_ms | **0/61 BLOCKED** | **10/15 ✓** |
| ttft_gap_prefill_ms | **0/61 BLOCKED** | **10/15 ✓** |
| ttft_head_recompute_early_ms | **0/61 BLOCKED** | **10/15 ✓** |
| ttft_head_recompute_late_ms | **0/61 BLOCKED** | **3/15 ✓** (late=0 when early_share=1.0, correct) |

(12/15, 10/15, 3/15 < 15 because not every request triggers chunk_plan/copy/head_recompute - e.g. lossless prefix-only requests have no chunk plan. This is correct behavior, not a bug.)

---

## 1. Bug #1: `__getstate__` allowlist + enable_metrics gate

### Diagnosis

`SchedulerReqTimeStats.__getstate__` (req_time_stats.py:642-655) controls what crosses the scheduler→detokenizer→tokenizer zmq pickle boundary:

```python
def __getstate__(self) -> object:
    if not self.enable_metrics:        # Gate 1: bench default = False -> return {}
        return {}
    state = {                          # Gate 2: allowlist of 5 pre-existing fields
        "wait_queue_entry_time": ...,
        "forward_entry_time": ...,
        "prefill_run_batch_start_time": ...,
        "prefill_run_batch_end_time": ...,
        "prefill_finished_time": ...,
        "diff_realtime_monotonic": ...,
    }
    return state
```

- **Gate 1**: bench runs without `--enable-metrics`, so `enable_metrics=False` -> `return {}` -> zero fields cross.
- **Gate 2**: even with `enable_metrics=True`, the 6 R40 fields (`radix_prefix_ms`, `chunk_plan_ms`, `copy_ms`, `gap_prefill_ms`, `head_recompute_early_ms`, `head_recompute_late_ms`) + `chunk_plan_done_time` are absent from the allowlist.

The receiving side (`ReqTimeStatsBase.__setstate__`, line 284-292) does `self.__dict__.update(state)`, so missing fields stay at their dataclass default of `0.0`. The tokenizer_manager per-batch snapshot (tokenizer_manager.py:1674-1729) reads `recv_obj.time_stats[i].radix_prefix_ms` etc., always 0.0.

The prior memory (`r40-ttft-breakdown-architecture-block-2026-07-09.md`) diagnosed this as a *timing* issue ("radix walk happens AFTER first batch output is sent"). That diagnosis was incomplete: the timing issue is real but secondary; the `__getstate__` allowlist is the primary blocker (the data never serializes regardless of timing).

### Fix

Make the 7 R40 fields + `diff_realtime_monotonic` always cross the boundary (7 floats + 1 timestamp + 1 diff = negligible overhead), regardless of `enable_metrics`. The heavier metrics fields stay gated:

```python
def __getstate__(self) -> object:
    r40 = {
        "radix_prefix_ms": self.radix_prefix_ms,
        "chunk_plan_ms": self.chunk_plan_ms,
        "copy_ms": self.copy_ms,
        "gap_prefill_ms": self.gap_prefill_ms,
        "head_recompute_early_ms": self.head_recompute_early_ms,
        "head_recompute_late_ms": self.head_recompute_late_ms,
        "chunk_plan_done_time": self.chunk_plan_done_time,
        "diff_realtime_monotonic": global_diff_realtime_monotonic,
    }
    if not self.enable_metrics:
        return r40
    state = { ... 5 metrics fields + diff ... }
    state.update(r40)
    return state
```

`diff_realtime_monotonic` MUST be included unconditionally because `__setstate__` uses it to convert cross-process monotonic timestamps (`chunk_plan_done_time` ends with "time" and is subject to `convert_time_cross_thread`). Without it, `__setstate__` would KeyError.

---

## 2. Bug #2: `NameError: name 'plan' is not defined`

### Diagnosis

After fix #1, 4/6 fields became non-zero but `head_recompute_early_ms` / `head_recompute_late_ms` stayed 0. A temporary debug print (gated on `SGLANG_P4_DEBUG`) in the scheduler revealed:

```
[P4DBG] EXCEPTION in head_recompute setters: NameError: name 'plan' is not defined
```

radix_cache.py:3960-3967:
```python
_early_n = int(os.environ.get("SGLANG_CHUNK_HEAD_RECOMPUTE_EARLY_N", "2") or 2)
_plan_copies = getattr(plan, "copy_count", 1) or 1   # <- plan undefined
_early_share = min(1.0, _early_n / _plan_copies)
_rope_no_gap = _rope_ms * 0.70
_ts.set_head_recompute_early_ms(_rope_no_gap * _early_share)
_ts.set_head_recompute_late_ms(_rope_no_gap * (1.0 - _early_share))
```

`_try_placeholder_chunk_lossy_match(self, req, key, exact_values, exact_node)` has no `plan` parameter. The comment said "flows from _build_chunk_plan via `plan`" but `plan` was never passed in. `set_copy_ms` (line 3953) and `set_gap_prefill_ms` (line 3959) execute before the NameError, so they survived; `set_head_recompute_early/late_ms` (line 3966-3967) never ran. The `except Exception: pass` (line 3968) silently swallowed it.

### Fix

`layout` (defined at radix_cache.py:3646) holds the chunk-copy plan in scope; `len(layout)` is the total chunk-copy count. Replace the undefined `plan.copy_count`:

```python
_plan_copies = len(layout) if layout else 1
```

Post-fix debug confirms correct values:
```
[P4DBG] rope_ms=6.126 early_share=0.500 plan_copies=4 hre=2.144 hrl=2.144
[P4DBG] rope_ms=6.936 early_share=1.000 plan_copies=2 hre=4.855 hrl=0.000
[P4DBG] rope_ms=16.262 early_share=1.000 plan_copies=2 hre=11.384 hrl=0.000
```

`early_share = min(1.0, 2/plan_copies)`: plan_copies=2 -> early_share=1.0 (all early, late=0); plan_copies=4 -> early_share=0.5 (split). Correct.

---

## 3. Verification

### Unit test (`results/verify_p4_getstate_fix.py`, 5 tests PASS)

1. `enable_metrics=False`: `__getstate__` returns all 7 R40 fields + `diff_realtime_monotonic` ✓
2. `enable_metrics=True`: also includes the 5 metrics fields (13 keys total) ✓
3. pickle round-trip preserves R40 field values (incl. `chunk_plan_done_time` ±epsilon after cross-thread time conversion) ✓
4. `chunk_plan_done_time=0.0` (lossless mode, no chunk plan) stays ~0.0 after round-trip ✓
5. regression: state is non-empty when `enable_metrics=False` (pre-fix returned `{}`) ✓

### End-to-end bench (3-case R32_f015, 15 rows)

All 6 previously-blocked columns now non-zero (see TL;DR table). `head_recompute_late_ms` is 3/15 because `early_share=1.0` (plan_copies ≤ 2) zeroes the late share - correct behavior.

---

## 4. Impact

- **Unlocks TTFT breakdown measurement**: the 6 radix/chunk timing fields now flow to rows.csv, enabling per-stage TTFT analysis for R32 / R38b / future configs. Previously only `tokenize_ms` was measurable; the other 6 stages were invisible.
- **Corrects R40 architecture-block diagnosis**: the block was `__getstate__` allowlist + NameError, NOT (primarily) a timing issue. The prior memory `r40-ttft-breakdown-architecture-block-2026-07-09.md` is superseded.
- **Enables honest speedup attribution**: with copy_ms / gap_prefill_ms / head_recompute_early/late_ms visible, we can now decompose the 1.43× R32 speedup into its constituent stages (copy vs RoPE vs head-recompute) - important for the §3 production-config framing (R32 is a speed-accuracy tradeoff, not accuracy-preserving).

---

## 5. 引用

- `results/verify_p4_getstate_fix.py` - unit test (5 PASS)
- `results/p4_verify/run_r32_f015_3case.sh` - 3-case bench launcher
- `results/p4_verify/r32_f015_3case/rows.csv` - post-fix data (6/6 fields non-zero)
- `python/sglang/srt/observability/req_time_stats.py:642-680` - `__getstate__` fix
- `python/sglang/srt/mem_cache/radix_cache.py:3942-3970` - NameError fix + comment
- Supersedes memory `r40-ttft-breakdown-architecture-block-2026-07-09.md`