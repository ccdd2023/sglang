#!/usr/bin/env bash
# Phase 3: topk_similarity threshold sweep for v44 placeholder_knn_reuse.
#
# Drives `bench_swe_generated_patch_kvcomm.py` over a list of (threshold, topk) pairs
# using the already-plumbed --placeholder-knn-min-cosine and --placeholder-knn-topk
# flags. Each invocation writes its own --out-dir so we can compare pass@1 per cell.
#
# Prereq:
#   - SGLang server can be cold-started per run (server is launched inside bench_swe).
#     For total cost control, also set --reuse-server in v2 if you keep the server up
#     between runs.
#   - Results land in results/swe_kn_sweep_<label>/
#
# Usage:
#   bash benchmark/multi_workflow/sweep_placeholder_knn_threshold.sh \
#       --max-cases 10 --start-index 0 --repo-filter '' \
#       --thresholds 1.00 0.99 0.97 0.95 0.90 0.85 \
#       --topks 1 3 5
#
# Notes:
#   - This script does NOT auto-launch sglang. Each bench_swe invocation launches its
#     own server (3-5 min warmup). For 6 thresholds × 3 topks × 1 case-set = 18 runs
#     × ~5 min = ~90 min GPU wall time.
#   - Add --reuse-server if you intend to keep one server up and only vary env vars.
#     bench_swe reads env vars at request time via HiRadixCache, so server-restart is
#     not strictly required, but cold cache state may dilute the per-cell signal.

set -euo pipefail

# ---- arg parsing -----------------------------------------------------------
MAX_CASES=10
START_INDEX=0
REPO_FILTER=""
THRESHOLDS=(1.00 0.99 0.97 0.95 0.90 0.85)
TOPKS=(1 3 5)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cases)        MAX_CASES="$2"; shift 2 ;;
        --start-index)      START_INDEX="$2"; shift 2 ;;
        --repo-filter)      REPO_FILTER="$2"; shift 2 ;;
        --thresholds)       shift; THRESHOLDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do THRESHOLDS+=("$1"); shift; done ;;
        --topks)            shift; TOPKS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TOPKS+=("$1"); shift; done ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- per-cell bench_swe invocation -----------------------------------------
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_ROOT="results/swe_kn_sweep_${DATE_TAG}"
mkdir -p "$RESULTS_ROOT"

echo "[sweep] results dir: $RESULTS_ROOT"
echo "[sweep] max_cases=$MAX_CASES start_index=$START_INDEX repo_filter='$REPO_FILTER'"
echo "[sweep] thresholds=${THRESHOLDS[*]} topks=${TOPKS[*]}"
echo "[sweep] extra args: ${EXTRA_ARGS[*]:-none}"
echo

declare -a SUMMARY=()
for th in "${THRESHOLDS[@]}"; do
    for k in "${TOPKS[@]}"; do
        label="sim${th}_k${k}"
        out_dir="$RESULTS_ROOT/$label"
        echo "[sweep] === threshold=$th topk=$k → $out_dir ==="
        cmd=(
            python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm
            --max-cases "$MAX_CASES"
            --start-index "$START_INDEX"
            --enable-placeholder-knn
            --placeholder-knn-min-cosine "$th"
            --placeholder-knn-topk "$k"
            --out-dir "$out_dir"
        )
        if [[ -n "$REPO_FILTER" ]]; then
            cmd+=(--repo-filter "$REPO_FILTER")
        fi
        cmd+=("${EXTRA_ARGS[@]}")
        # Surface the command so the user can replay later
        printf '[sweep] cmd: %q ' "${cmd[@]}"; printf '\n'
        if "${cmd[@]}"; then
            SUMMARY+=("OK   threshold=$th topk=$k → $out_dir")
        else
            SUMMARY+=("FAIL threshold=$th topk=$k → exit=$?")
        fi
        echo
    done
done

# ---- summary ---------------------------------------------------------------
SUMMARY_FILE="$RESULTS_ROOT/SWEEP_SUMMARY.txt"
{
    echo "placeholder_knn_reuse threshold sweep"
    echo "date: $DATE_TAG"
    echo "max_cases=$MAX_CASES start_index=$START_INDEX repo_filter='$REPO_FILTER'"
    echo "thresholds=${THRESHOLDS[*]}"
    echo "topks=${TOPKS[*]}"
    echo
    echo "Per-cell result:"
    for line in "${SUMMARY[@]}"; do
        echo "  $line"
    done
} > "$SUMMARY_FILE"

echo "[sweep] summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"