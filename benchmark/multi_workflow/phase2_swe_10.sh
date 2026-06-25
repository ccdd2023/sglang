#!/usr/bin/env bash
# Phase 2.1 sequential driver:
#  2.1a) baseline (lossless + lossy + lossy_prefetch) on 10 SWE cases
#  2.1b) v44 placeholder_knn_lossy on the same 10 cases
#
# Prereq:
#   - GPU is free (no other sglang process running)
#   - swebench_local_envs/repos/ has all 10 cases ready (astropy, django, matplotlib, ...)
#   - Three flags are MANDATORY for > 3 cases:
#       --disable-overlap-schedule : serialize prefill batches
#       --max-running-requests 1   : single concurrent request
#       --force-evict              : force-evict fallback for lock-pressure OOMs
#     Without these, scheduler hits `_delete_leaf` AssertionError around case 3-4.
#     (See memory: _delete-leaf-bug-2026-06-24.md + 100-case-force-evict-fix.md)
#
# Each invocation is ~15-25 min (server warmup + 10 cases × 3 modes serial).
# Total wall time: ~30-50 min on RTX 4090.

set -euo pipefail

DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET="${DATASET:-results/repo_level_datasets/swe_verified_10_instances.json}"

BASELINE_DIR="results/swe_correctness_baseline_10_${DATE_TAG}"
V44_DIR="results/swe_correctness_v44_10_${DATE_TAG}"

echo "[phase2.1] date_tag=$DATE_TAG dataset=$DATASET"
echo "[phase2.1] baseline → $BASELINE_DIR"
echo "[phase2.1] v44      → $V44_DIR"
echo

# 2.1a: baseline (lossless/lossy/lossy_prefetch)
mkdir -p "$BASELINE_DIR"
echo "[phase2.1a] === baseline ==="
python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm \
    --dataset "$DATASET" \
    --max-cases 10 \
    --start-index 0 \
    --out-dir "$BASELINE_DIR" \
    --server-timeout 240 \
    --eval-timeout 900 \
    --disable-overlap-schedule \
    --max-running-requests 1 \
    --force-evict 2>&1 | tee "$BASELINE_DIR/run.log" | tail -20
echo

# 2.1b: v44 placeholder_knn_lossy (adds placeholder_knn_lossy mode on top of baseline)
mkdir -p "$V44_DIR"
echo "[phase2.1b] === v44 ==="
python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm \
    --dataset "$DATASET" \
    --max-cases 10 \
    --start-index 0 \
    --out-dir "$V44_DIR" \
    --server-timeout 240 \
    --eval-timeout 900 \
    --enable-placeholder-knn \
    --disable-overlap-schedule \
    --max-running-requests 1 \
    --force-evict 2>&1 | tee "$V44_DIR/run.log" | tail -20
echo

# 2.1c: aggregate
echo "[phase2.1c] === aggregate ==="
python -m benchmark.multi_workflow.aggregate_swe_pass_at_1 \
    --summary "$BASELINE_DIR/summary.json" \
    --summary "$V44_DIR/summary.json" \
    --baseline-label "$(basename $BASELINE_DIR)" \
    --out-md "results/swe_pass_at_1_compare_${DATE_TAG}.md"
echo
echo "[phase2.1] all done. compare: results/swe_pass_at_1_compare_${DATE_TAG}.md"