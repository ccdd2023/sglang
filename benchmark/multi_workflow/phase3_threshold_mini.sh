#!/usr/bin/env bash
# Phase 3 mini: threshold sweep (placeholder_knn_min_cosine × placeholder_knn_topk)
#
# Caveat: per-case driver runs each case in a fresh server, so anchor pool
# is never populated. This means v44 placeholder_knn_lossy always falls back
# to standard lossy, regardless of threshold. Result: all thresholds × topk
# combinations produce byte-identical output.
#
# This script verifies that hypothesis. If we ever fix _delete_leaf bug and
# can run multi-request same-server, then this sweep becomes meaningful.
#
# Configurations tested (mini sweep):
#   thresholds: 0.85, 0.95, 1.00  (default 0.70)
#   topks: 1, 3, 5  (default 4)
#   cases: 3 (astropy, django, matplotlib)
# Total: 9 runs × 5 min = 45 min.

set -euo pipefail

cd "$(dirname "$0")/../.."

DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET="${DATASET:-results/repo_level_datasets/swe_verified_3_instances.json}"
CASES=("astropy__astropy-12907" "django__django-10097" "matplotlib__matplotlib-13989")
THRESHOLDS=(0.85 0.95 1.00)
TOPKS=(1 3 5)

OUT_ROOT="results/swe_percase_threshold_${DATE_TAG}"
mkdir -p "$OUT_ROOT"

run_cell() {
    local threshold="$1"
    local topk="$2"
    local case_id="$3"
    local label="t${threshold}_k${topk}_${case_id}"
    local out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"
    echo "$case_id" > "$out_dir/_ids.txt"
    echo "[phase3-mini] === $label ==="
    python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm \
        --dataset "$DATASET" \
        --instance-id-file "$out_dir/_ids.txt" \
        --max-cases 1 \
        --out-dir "$out_dir" \
        --server-timeout 240 \
        --eval-timeout 300 \
        --skip-candidate-tests \
        --enable-placeholder-knn \
        --placeholder-knn-min-cosine "$threshold" \
        --placeholder-knn-topk "$topk" 2>&1 | tee "$out_dir/run.log" | tail -3
    pkill -9 -f "sglang.launch_server\|sglang::scheduler\|sglang::detokenizer" 2>/dev/null || true
    sleep 3
}

for threshold in "${THRESHOLDS[@]}"; do
    for topk in "${TOPKS[@]}"; do
        for case_id in "${CASES[@]}"; do
            run_cell "$threshold" "$topk" "$case_id"
        done
    done
done

echo "[phase3-mini] === AGGREGATE ==="
python -c "
import json
from pathlib import Path
from collections import defaultdict

root = Path('$OUT_ROOT')
results = defaultdict(dict)  # (case_id, threshold, topk) -> stats

for cell_dir in sorted(root.iterdir()):
    if not cell_dir.is_dir():
        continue
    parts = cell_dir.name.split('_')
    if len(parts) < 4:
        continue
    threshold = parts[0].lstrip('t')
    topk = parts[1].lstrip('k')
    case_id = '_'.join(parts[2:])
    sj = cell_dir / 'summary.json'
    if not sj.exists():
        continue
    d = json.loads(sj.read_text())
    for case in d['results']:
        for m in case.get('modes', []):
            if m['mode'] == 'placeholder_knn_lossy':
                patch = Path(m['patch_path'])
                results[(case_id, threshold, topk)] = {
                    'extracted': m['diff_extracted'],
                    'synth_ok': m['patch_synthesis']['ok'],
                    'apply_rc': m.get('apply_check', {}).get('returncode'),
                    'patch_bytes': patch.stat().st_size if patch.exists() else 0,
                    'topk_sim': m.get('lossy_meta', {}).get('placeholder_knn_topk_similarity_mean', 0.0),
                    'copy_method': m.get('lossy_meta', {}).get('placeholder_knn_copy_method', ''),
                }

# Cross-cell byte-equality: for each case, do all (threshold, topk) cells
# produce byte-equal placeholder_knn_lossy patches?
print('## Threshold sweep invariance check')
print()
for case_id in sorted({k[0] for k in results}):
    cells = {k: v for k, v in results.items() if k[0] == case_id}
    print(f'### {case_id}')
    print()
    print('| threshold | topk | bytes | apply_rc | sim | copy_method | byte-equal-to-base-losyy? |')
    print('|---|---|---|---|---|---|---|')
    # Find baseline lossy patch (from baseline run if exists)
    base_dir = Path('results/swe_percase_baseline_20260624T085604Z') / case_id
    base_losy = base_dir / 'lossy.patch'
    for (cid, th, k), v in sorted(cells.items()):
        cell_dir = root / f't{th}_k{k}_{cid}'
        patch = cell_dir / 'placeholder_knn_lossy.patch'
        equal = '?'
        if base_losy.exists() and patch.exists():
            equal = 'EQUAL' if patch.read_bytes() == base_losy.read_bytes() else 'DIFFER'
        sim = v['topk_sim']
        sim_s = f'{sim:.3f}' if sim else '—'
        print(f'| {th} | {k} | {v[\"patch_bytes\"]} | {v[\"apply_rc\"]} | {sim_s} | {v[\"copy_method\"]} | {equal} |')
    print()
"

echo "[phase3-mini] done. compare: $OUT_ROOT/_summary.txt"