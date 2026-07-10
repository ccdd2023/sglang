#!/bin/bash
# Scale-15 equal-budget ablation: direction A (node-kind interface-recompute)
# vs R32-uniform (sweep) vs R38b-position vs lossless. All configs use the same
# fixed byte_to_tok path (P1.2) and the same pool (pandas_15case_v1).
#
# Configs (n=15, 5 agents, verdict):
#   lossless       - baseline accuracy + slowest TTFT (no reuse)
#   r32  (0.30)    - uniform FRAC, continuity with scale15 (re-run with fix)
#   r32_f015       - uniform FRAC sweep low
#   r32_f026       - uniform FRAC = frac* (equal-budget to node-kind interface)
#   r32_f045       - uniform FRAC sweep high
#   r38b           - position-stratified EARLY=0.60/LATE=0.15 (re-run with fix)
#   nodekind       - direction A: K = signature + docstring (interface)
#   nodekind_sig   - direction A variant: K = signature only
#
# Re-runs lossless/r32/r38b because the byte_to_tok fix (P1.2) changes their
# TTFT (offsets unchanged -> accuracy unchanged, only faster). Comment out any
# line to skip. ~1-2h GPU depending on OOM relaunches.
cd /home/gfy/CodeMAS_Project/sglang-kvflow
L=results/scale15_5x5/launchers
O=results/scale15_5x5
run () {
  local name="$1"; shift
  mkdir -p "$O/$name"
  # Clean prior run artifacts (specific files, not rm -rf, so the launcher
  # writes fresh rows.csv/outputs.jsonl without mixing runs).
  rm -f "$O/$name/rows.csv" "$O/$name/outputs.jsonl" "$O/$name/sglang_server.log" "$O/$name/FAIR_SUMMARY.md"
  echo "=== $name ==="
  bash "$L/$1" "$2" "$3" > "$O/$name.stdout" 2>&1 && echo "ok $name" || echo "FAILED $name (see $name.stdout)"
}

run lossless     run_lossless.sh
run r32          run_r32.sh
run r32_f015     run_r32_frac.sh 0.15 f015
run r32_f026     run_r32_frac.sh 0.26 f026
run r32_f045     run_r32_frac.sh 0.45 f045
run r38b         run_r38b.sh
run nodekind     run_nodekind.sh
run nodekind_sig run_nodekind_sig.sh
echo "=== ALL DONE ==="
echo "analyze: python3 results/scale15_5x5/analyze_ablation_nodekind.py"
