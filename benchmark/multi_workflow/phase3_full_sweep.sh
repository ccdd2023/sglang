#!/usr/bin/env bash
# Phase 3 FULL sweep — 60 case = 10 cases × 6 thresholds × K=5
# 3-cases-per-server chunks (delimiter: `_delete_leaf` race fires at case 4).
# 4 servers × 6 thresholds = 24 server starts.
# Time estimate: ~5 min/server × 24 = ~2 hours.

set -euo pipefail

cd "$(dirname "$0")/../.."

# 10 cases stratified across 10 repos (same as Phase 2 per-case driver)
CASES=(
  "astropy__astropy-12907"
  "django__django-10097"
  "matplotlib__matplotlib-13989"
  "mwaskom__seaborn-3069"
  "pallets__flask-5014"
  "psf__requests-1142"
  "pydata__xarray-2905"
  "pylint-dev__pylint-4551"
  "pytest-dev__pytest-10051"
  "scikit-learn__scikit-learn-10297"
)

# Split into chunks of 3 (last chunk may have 1)
CHUNK_SIZES=(3 3 3 1)

SWEEP_ROOT="results/swe_percase_threshold_full_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$SWEEP_ROOT"
echo "[phase3-full] sweep root: $SWEEP_ROOT"

THRESHOLDS=(0.85 0.90 0.95 0.97 0.99 1.00)
TOPK=5

# Build chunk file lists — use index in filename to avoid collision
INDEX=0
CHUNK_FILES=()
CHUNK_NAMES=()
CI=0
for SIZE in "${CHUNK_SIZES[@]}"; do
  CHUNK_FILE="$SWEEP_ROOT/_ids_chunk_${CI}_s${SIZE}.txt"
  END=$((INDEX + SIZE))
  printf '%s\n' "${CASES[@]:$INDEX:$SIZE}" > "$CHUNK_FILE"
  CHUNK_FILES+=("$CHUNK_FILE")
  CHUNK_NAMES+=("c${SIZE}_$(printf '%02d' $((INDEX+1)))")
  INDEX=$END
  CI=$((CI+1))
done

for T in "${THRESHOLDS[@]}"; do
  for I in "${!CHUNK_FILES[@]}"; do
    SIZE="${CHUNK_SIZES[$I]}"
    NAME="${CHUNK_NAMES[$I]}_k${TOPK}"
    CELL_DIR="$SWEEP_ROOT/t${T}_${NAME}_t${T}"
    mkdir -p "$CELL_DIR"
    echo "[phase3-full] === t=${T} chunk=${NAME} (${SIZE} cases in 1 server) ==="
    python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm \
      --dataset results/repo_level_datasets/swe_verified_10_instances.json \
      --instance-id-file "${CHUNK_FILES[$I]}" \
      --max-cases "$SIZE" \
      --out-dir "$CELL_DIR" \
      --server-timeout 360 \
      --eval-timeout 300 \
      --skip-candidate-tests \
      --enable-placeholder-knn \
      --placeholder-knn-min-cosine "$T" \
      --placeholder-knn-topk "$TOPK" \
      --force-evict \
      --disable-overlap-schedule \
      --max-running-requests 1 \
      > "$CELL_DIR/run.log" 2>&1 || {
        rc=$?
        echo "[phase3-full] cell t${T}_${NAME} FAILED rc=$rc (continuing)"
        echo "FAILED_RC=$rc" > "$CELL_DIR/_status.txt"
        continue
      }
    echo "OK" > "$CELL_DIR/_status.txt"
  done
done

echo "[phase3-full] all 24 (threshold, chunk) cells done"
echo "[phase3-full] sweep root: $SWEEP_ROOT"