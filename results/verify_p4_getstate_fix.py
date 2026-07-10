#!/usr/bin/env python3
"""P4 __getstate__ fix verification (2026-07-10).

Verifies that SchedulerReqTimeStats.__getstate__ now serializes the 6 R40
TTFT-breakdown fields + chunk_plan_done_time across the zmq pickle boundary
even when enable_metrics=False (the bench default).

Checks:
  1. enable_metrics=False: __getstate__ returns all 7 R40 fields + diff_realtime_monotonic
  2. enable_metrics=True: also includes the 5 metrics fields
  3. pickle round-trip preserves R40 field values
  4. __setstate__ time-conversion handles chunk_plan_done_time (no KeyError)
  5. 0.0 chunk_plan_done_time (lossless mode) survives round-trip cleanly
"""
import pickle
import sys
from pathlib import Path

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.observability.req_time_stats import (
    SchedulerReqTimeStats,
    global_diff_realtime_monotonic,
    calibrate_time_diff,
)

calibrate_time_diff()  # ensure global_diff_realtime_monotonic is set

R40_FIELDS = [
    "radix_prefix_ms", "chunk_plan_ms", "copy_ms", "gap_prefill_ms",
    "head_recompute_early_ms", "head_recompute_late_ms", "chunk_plan_done_time",
]

def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond

print("=== P4 __getstate__ fix verification ===\n")

all_pass = True

# --- Test 1: enable_metrics=False carries R40 fields ---
print("Test 1: enable_metrics=False (bench default)")
ts = SchedulerReqTimeStats()
ts.enable_metrics = False
ts.radix_prefix_ms = 1.5
ts.chunk_plan_ms = 2.0
ts.copy_ms = 0.5
ts.gap_prefill_ms = 0.3
ts.head_recompute_early_ms = 0.8
ts.head_recompute_late_ms = 0.2
ts.chunk_plan_done_time = 12345.678

state = ts.__getstate__()
for f in R40_FIELDS:
    all_pass &= check(f"  field '{f}' in state", f in state)
all_pass &= check("diff_realtime_monotonic in state", "diff_realtime_monotonic" in state)
all_pass &= check("metrics fields NOT in state (enable_metrics=False)",
                  "wait_queue_entry_time" not in state)
print(f"  state keys: {sorted(state.keys())}")
print()

# --- Test 2: enable_metrics=True carries both R40 + metrics fields ---
print("Test 2: enable_metrics=True")
ts2 = SchedulerReqTimeStats()
ts2.enable_metrics = True
ts2.wait_queue_entry_time = 100.0
ts2.radix_prefix_ms = 1.5
state2 = ts2.__getstate__()
print(f"  [debug] ts2.enable_metrics={ts2.enable_metrics}, n_state2_keys={len(state2)}")
for f in R40_FIELDS:
    all_pass &= check(f"  field '{f}' in state", f in state2)
all_pass &= check("wait_queue_entry_time in state", "wait_queue_entry_time" in state2)
all_pass &= check("forward_entry_time in state", "forward_entry_time" in state2)
print()

# --- Test 3: pickle round-trip preserves R40 values ---
print("Test 3: pickle round-trip preserves R40 values")
ts3 = SchedulerReqTimeStats()
ts3.enable_metrics = False
ts3.radix_prefix_ms = 1.5
ts3.chunk_plan_ms = 2.0
ts3.copy_ms = 0.5
ts3.gap_prefill_ms = 0.3
ts3.head_recompute_early_ms = 0.8
ts3.head_recompute_late_ms = 0.2
ts3.chunk_plan_done_time = 12345.678

# Simulate the zmq crossing: pickle.dumps on scheduler side, loads on tokenizer side.
# __getstate__ controls what crosses; __setstate__ reconstructs.
data = pickle.dumps(ts3)
ts3_recv = pickle.loads(data)
for f in R40_FIELDS:
    if f == "chunk_plan_done_time":
        # __setstate__ applies convert_time_cross_thread to "*_time" fields.
        # On the same process, old_diff == new_diff, so the value is unchanged.
        all_pass &= check(f"  {f} preserved (±epsilon)",
                          abs(getattr(ts3_recv, f) - getattr(ts3, f)) < 1e-6)
    else:
        all_pass &= check(f"  {f} preserved",
                          getattr(ts3_recv, f) == getattr(ts3, f))
print(f"  recv.radix_prefix_ms = {ts3_recv.radix_prefix_ms}")
print(f"  recv.chunk_plan_ms   = {ts3_recv.chunk_plan_ms}")
print(f"  recv.chunk_plan_done_time = {ts3_recv.chunk_plan_done_time}")
print()

# --- Test 4: 0.0 chunk_plan_done_time (lossless mode, no chunk plan) ---
print("Test 4: chunk_plan_done_time=0.0 (lossless mode, no chunk plan)")
ts4 = SchedulerReqTimeStats()
ts4.enable_metrics = False
ts4.chunk_plan_done_time = 0.0  # lossless path never sets this
data4 = pickle.dumps(ts4)
ts4_recv = pickle.loads(data4)
# After cross-thread conversion on same process: 0.0 + diff - diff = 0.0
all_pass &= check("  0.0 chunk_plan_done_time stays ~0.0 after round-trip",
                  abs(ts4_recv.chunk_plan_done_time) < 1e-6)
print(f"  recv.chunk_plan_done_time = {ts4_recv.chunk_plan_done_time}")
print()

# --- Test 5: pre-fix regression check (would have been empty) ---
print("Test 5: regression check - state is NOT empty when enable_metrics=False")
ts5 = SchedulerReqTimeStats()
ts5.enable_metrics = False
state5 = ts5.__getstate__()
all_pass &= check("  state is non-empty (pre-fix returned {})", len(state5) > 0)
all_pass &= check("  state has >= 8 keys (7 R40 + diff)", len(state5) >= 8)
print()

print("=" * 50)
if all_pass:
    print("ALL TESTS PASSED - P4 __getstate__ fix verified")
    sys.exit(0)
else:
    print("SOME TESTS FAILED")
    sys.exit(1)