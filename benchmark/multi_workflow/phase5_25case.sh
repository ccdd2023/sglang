#!/usr/bin/env bash
# Phase 5.1: 25-case pass@1 scale-up via per-case driver.
#
# Same strategy as phase2_per_case.sh but uses 25 cases from
# swe_verified_100_instances.json. Goal: confirm §6.5 gate holds at
# larger sample size.
#
# 25 cases × 2 runs × ~5 min/case = ~250 min = ~4.2h wall time.

set -euo pipefail

cd "$(dirname "$0")/../.."

DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET="${DATASET:-results/repo_level_datasets/swe_verified_100_instances.json}"
MAX_CASES=25

# Discover first 25 case IDs from dataset
CASE_IDS=($(python -c "import json; d=json.load(open('$DATASET')); [print(c['instance_id']) for c in d[:$MAX_CASES]]"))
echo "[phase5.1] dataset=$DATASET first $MAX_CASES cases: ${CASE_IDS[*]}"

BASELINE_ROOT="results/swe_percase25_baseline_${DATE_TAG}"
V44_ROOT="results/swe_percase25_v44_${DATE_TAG}"
mkdir -p "$BASELINE_ROOT" "$V44_ROOT"

run_case() {
    local mode_flag="$1"
    local root="$2"
    local case_id="$3"
    local case_dir="$root/$case_id"
    mkdir -p "$case_dir"
    local idfile="$case_dir/_ids.txt"
    echo "$case_id" > "$idfile"
    echo "[phase5.1] === case=$case_id mode_flag=$mode_flag ==="
    python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm \
        --dataset "$DATASET" \
        --instance-id-file "$idfile" \
        --max-cases 1 \
        --out-dir "$case_dir" \
        --server-timeout 240 \
        --eval-timeout 300 \
        --skip-candidate-tests \
        $mode_flag 2>&1 | tee "$case_dir/run.log" | tail -3
    pkill -9 -f "sglang.launch_server\|sglang::scheduler\|sglang::detokenizer" 2>/dev/null || true
    sleep 3
}

echo "[phase5.1] === BASELINE per-case ($MAX_CASES cases) ==="
for case_id in "${CASE_IDS[@]}"; do
    run_case "" "$BASELINE_ROOT" "$case_id"
done

echo "[phase5.1] === V44 per-case ($MAX_CASES cases) ==="
for case_id in "${CASE_IDS[@]}"; do
    run_case "--enable-placeholder-knn" "$V44_ROOT" "$case_id"
done

echo "[phase5.1] === AGGREGATE ==="
python -m benchmark.multi_workflow.aggregate_per_case_pass_at_1 \
    --baseline-root "$BASELINE_ROOT" \
    --v44-root "$V44_ROOT" \
    --baseline-label "$(basename $BASELINE_ROOT)" \
    --out-md "results/per_case25_compare_${DATE_TAG}.md" \
    --out-json "results/per_case25_compare_${DATE_TAG}.json"

echo "[phase5.1] done. compare: results/per_case25_compare_${DATE_TAG}.md"