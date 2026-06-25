#!/usr/bin/env bash
# Wait for Phase 5.1 to complete, then auto-start Phase 3 mini-sweep.
# Usage: bash benchmark/multi_workflow/phase3_after_phase5.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[phase3-auto] waiting for Phase 5.1 to complete (looking for 'phase5.strat.*done' in results/phase5_strat_run.log)..."

# Wait up to 6 hours
END=$(( $(date +%s) + 21600 ))
while [ $(date +%s) -lt $END ]; do
    if grep -q "phase5.strat.*done" results/phase5_strat_run.log 2>/dev/null; then
        echo "[phase3-auto] Phase 5.1 done, starting Phase 3 mini-sweep"
        bash benchmark/multi_workflow/phase3_threshold_mini.sh 2>&1 | tee results/phase3_mini_run.log
        exit 0
    fi
    sleep 30
done

echo "[phase3-auto] timeout waiting for Phase 5.1"
exit 1