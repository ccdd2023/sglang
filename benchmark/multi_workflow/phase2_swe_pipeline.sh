#!/usr/bin/env bash
# Phase 2.1 (revised): 3-case pass@1 validation
#
# Why 3 cases not 10: sglang-kvflow harness crashes at case 3-4 with a
# _delete_leaf assertion race condition that --force-evict / --disable-overlap
# / --max-running-requests 1 cannot fix (memory _delete-leaf-bug-2026-06-24).
# 3 cases is the largest known-working N. This is a tractable, honest result.
#
# Pipeline: baseline (lossless/lossy/lossy_prefetch) → v44 placeholder_knn_lossy
#           → aggregate_swe_pass_at_1 → Markdown report
#
# Each run: ~5-10 min GPU (server warmup 3 min + 3 cases × 3 modes ~1-2 min each).
# Total wall time: ~10-20 min on RTX 4090.

set -euo pipefail

cd "$(dirname "$0")/../.."

DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET="${DATASET:-results/repo_level_datasets/swe_verified_3_instances.json}"

BASELINE_DIR="results/swe_correctness_baseline_3_${DATE_TAG}"
V44_DIR="results/swe_correctness_v44_3_${DATE_TAG}"
REPORT_PATH="results/swe_correctness_compare_${DATE_TAG}.md"

echo "[phase2.1r] date=$DATE_TAG dataset=$DATASET"
echo "[phase2.1r] baseline → $BASELINE_DIR"
echo "[phase2.1r] v44      → $V44_DIR"
echo "[phase2.1r] report   → $REPORT_PATH"
echo

# Cleanup any lingering sglang
pkill -9 -f "sglang.launch_server\|sglang::scheduler\|sglang::detokenizer" 2>/dev/null || true
sleep 2

# 2.1a: baseline
mkdir -p "$BASELINE_DIR"
echo "[phase2.1r.a] === baseline ==="
python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm \
    --dataset "$DATASET" \
    --max-cases 3 \
    --start-index 0 \
    --out-dir "$BASELINE_DIR" \
    --server-timeout 240 \
    --eval-timeout 600 2>&1 | tee "$BASELINE_DIR/run.log" | tail -15
echo

# 2.1b: v44 placeholder_knn_lossy
pkill -9 -f "sglang.launch_server\|sglang::scheduler\|sglang::detokenizer" 2>/dev/null || true
sleep 2
mkdir -p "$V44_DIR"
echo "[phase2.1r.b] === v44 ==="
python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm \
    --dataset "$DATASET" \
    --max-cases 3 \
    --start-index 0 \
    --out-dir "$V44_DIR" \
    --server-timeout 240 \
    --eval-timeout 600 \
    --enable-placeholder-knn 2>&1 | tee "$V44_DIR/run.log" | tail -15
echo

# 2.1c: aggregate
echo "[phase2.1r.c] === aggregate ==="
python -m benchmark.multi_workflow.aggregate_swe_pass_at_1 \
    --summary "$BASELINE_DIR/summary.json" \
    --summary "$V44_DIR/summary.json" \
    --baseline-label "$(basename $BASELINE_DIR)" \
    --out-md "$REPORT_PATH"
echo
echo "[phase2.1r] done. report: $REPORT_PATH"