#!/usr/bin/env bash
# Phase 2.1 UNBLOCKED: per-case server-restart to avoid _delete_leaf race.
#
# Strategy: instead of running N cases in one server (which triggers the
# radix_cache._delete_leaf assertion race around case 1-3), run each case
# in a separate sglang server invocation. Each invocation:
#   1. Launches sglang (3 min warmup)
#   2. Runs 1 case × 4 modes (lossless / lossy / lossy_prefetch / placeholder_knn_lossy)
#   3. Tears down server
#   4. Aggregates that case's result
#
# This is slower (20 launches for 10 cases × 2 runs) but each launch is self-
# contained so no cross-case KV state can race.
#
# Total: 20 launches × ~5 min = ~100 min. Acceptable for unblocking.

set -euo pipefail

cd "$(dirname "$0")/../.."

DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET="${DATASET:-results/repo_level_datasets/swe_verified_10_instances.json}"

# Discover case IDs from dataset
CASE_IDS=($(python -c "import json; d=json.load(open('$DATASET')); [print(c['instance_id']) for c in d]"))
echo "[per-case] dataset=$DATASET found ${#CASE_IDS[@]} cases: ${CASE_IDS[*]}"

# Where to write
BASELINE_ROOT="results/swe_percase_baseline_${DATE_TAG}"
V44_ROOT="results/swe_percase_v44_${DATE_TAG}"
mkdir -p "$BASELINE_ROOT" "$V44_ROOT"

# Helper: run one case in one mode
run_case() {
    local mode_flag="$1"
    local root="$2"
    local case_id="$3"
    local case_dir="$root/$case_id"
    mkdir -p "$case_dir"
    # Write instance-id file
    local idfile="$case_dir/_ids.txt"
    echo "$case_id" > "$idfile"
    echo "[per-case] === case=$case_id mode_flag=$mode_flag ==="
    python -m benchmark.multi_workflow.bench_swe_generated_patch_kvcomm \
        --dataset "$DATASET" \
        --instance-id-file "$idfile" \
        --max-cases 1 \
        --out-dir "$case_dir" \
        --server-timeout 240 \
        --eval-timeout 300 \
        --skip-candidate-tests \
        $mode_flag 2>&1 | tee "$case_dir/run.log" | tail -5
    # Kill any lingering server
    pkill -9 -f "sglang.launch_server\|sglang::scheduler\|sglang::detokenizer" 2>/dev/null || true
    sleep 3
}

# Baseline run: per case
echo "[per-case] === BASELINE (lossless/lossy/lossy_prefetch) per-case ==="
for case_id in "${CASE_IDS[@]}"; do
    run_case "" "$BASELINE_ROOT" "$case_id"
done

# v44 run: per case
echo "[per-case] === V44 (placeholder_knn_lossy + others) per-case ==="
for case_id in "${CASE_IDS[@]}"; do
    run_case "--enable-placeholder-knn" "$V44_ROOT" "$case_id"
done

# Aggregate
echo "[per-case] === AGGREGATE ==="
python -c "
import json
from pathlib import Path

def collect(root):
    by_case = {}
    for case_dir in sorted(Path(root).iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith('_'):
            continue
        sj = case_dir / 'summary.json'
        if not sj.exists():
            continue
        d = json.loads(sj.read_text())
        for case in d.get('results', []):
            cid = case['instance_id']
            by_case[cid] = {}
            for m in case.get('modes', []):
                patch = Path(m['patch_path'])
                by_case[cid][m['mode']] = {
                    'extracted': m['diff_extracted'],
                    'synth_ok': m['patch_synthesis']['ok'],
                    'apply_rc': m.get('apply_check', {}).get('returncode'),
                    'patch_bytes': patch.stat().st_size if patch.exists() else 0,
                    'first_match_reason': m.get('lossy_meta', {}).get('lossy_first_match_reason'),
                    'topk_sim': m.get('lossy_meta', {}).get('placeholder_knn_topk_similarity_mean', 0.0),
                    'copy_method': m.get('lossy_meta', {}).get('placeholder_knn_copy_method', ''),
                }
    return by_case

base = collect('$BASELINE_ROOT')
v44 = collect('$V44_ROOT')

print(f'{\"case_id\":<35} {\"mode\":<22} {\"base_extracted\":<14} {\"v44_extracted\":<13} {\"base_bytes\":<10} {\"v44_bytes\":<9} {\"equal\"}')
print('-' * 110)
for cid in sorted(set(base) | set(v44)):
    for mode in ['lossless', 'lossy', 'lossy_prefetch', 'placeholder_knn_lossy']:
        b = base.get(cid, {}).get(mode, {})
        v = v44.get(cid, {}).get(mode, {})
        if not b and not v: continue
        be = b.get('extracted')
        ve = v.get('extracted')
        bb = b.get('patch_bytes', 0)
        vb = v.get('patch_bytes', 0)
        eq = '?'
        if mode in base.get(cid, {}) and mode in v44.get(cid, {}):
            bb_path = Path('$BASELINE_ROOT') / cid / f'{mode}.patch'
            vb_path = Path('$V44_ROOT') / cid / f'{mode}.patch'
            try:
                eq = 'EQUAL' if bb_path.read_bytes() == vb_path.read_bytes() else 'DIFFER'
            except FileNotFoundError:
                eq = 'N/A'
        print(f'{cid:<35} {mode:<22} {str(be):<14} {str(ve):<13} {bb:<10} {vb:<9} {eq}')
" | tee "results/swe_percase_compare_${DATE_TAG}.txt"
echo "[per-case] done. compare: results/swe_percase_compare_${DATE_TAG}.txt"