#!/usr/bin/env bash
# Phase 5.1: stratified 27-case pass@1 scale-up via per-case driver.
#
# 27 cases stratified across 10 repos (3 per repo, 1-2 for small repos).
# Same strategy as phase2_per_case.sh: each case runs in its own sglang
# server subprocess to avoid the _delete_leaf harness race.
#
# 27 cases × 2 runs × ~5 min/case = ~270 min = ~4.5h wall time.

set -euo pipefail

cd "$(dirname "$0")/../.."

DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET="${DATASET:-results/repo_level_datasets/swe_verified_100_instances.json}"

# Stratified 27 cases (3 per repo, capped by dataset availability)
CASE_IDS=(
    astropy__astropy-12907
    astropy__astropy-13033
    astropy__astropy-13236
    django__django-10097
    django__django-10554
    django__django-10880
    matplotlib__matplotlib-13989
    matplotlib__matplotlib-14623
    matplotlib__matplotlib-20488
    mwaskom__seaborn-3069
    mwaskom__seaborn-3187
    pallets__flask-5014
    psf__requests-1142
    psf__requests-1724
    psf__requests-1766
    pydata__xarray-2905
    pydata__xarray-3095
    pydata__xarray-3151
    pylint-dev__pylint-4551
    pylint-dev__pylint-4604
    pylint-dev__pylint-4661
    pytest-dev__pytest-10051
    pytest-dev__pytest-10081
    pytest-dev__pytest-10356
    scikit-learn__scikit-learn-10297
    scikit-learn__scikit-learn-10844
    scikit-learn__scikit-learn-10908
)

BASELINE_ROOT="results/swe_strat27_baseline_${DATE_TAG}"
V44_ROOT="results/swe_strat27_v44_${DATE_TAG}"
mkdir -p "$BASELINE_ROOT" "$V44_ROOT"

run_case() {
    local mode_flag="$1"
    local root="$2"
    local case_id="$3"
    local case_dir="$root/$case_id"
    mkdir -p "$case_dir"
    local idfile="$case_dir/_ids.txt"
    echo "$case_id" > "$idfile"
    echo "[phase5.strat] === case=$case_id mode_flag=$mode_flag ==="
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

echo "[phase5.strat] === BASELINE per-case (${#CASE_IDS[@]} cases) ==="
for case_id in "${CASE_IDS[@]}"; do
    run_case "" "$BASELINE_ROOT" "$case_id"
done

echo "[phase5.strat] === V44 per-case (${#CASE_IDS[@]} cases) ==="
for case_id in "${CASE_IDS[@]}"; do
    run_case "--enable-placeholder-knn" "$V44_ROOT" "$case_id"
done

echo "[phase5.strat] === AGGREGATE ==="
python -m benchmark.multi_workflow.aggregate_per_case_pass_at_1 \
    --baseline-root "$BASELINE_ROOT" \
    --v44-root "$V44_ROOT" \
    --baseline-label "$(basename $BASELINE_ROOT)" \
    --out-md "results/strat27_compare_${DATE_TAG}.md" \
    --out-json "results/strat27_compare_${DATE_TAG}.json"

echo "[phase5.strat] done. compare: results/strat27_compare_${DATE_TAG}.md"