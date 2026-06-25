#!/usr/bin/env python3
"""Aggregate Phase 3 FULL sweep byte-equality.

Cell-level summary.json structure:
  cell_dir/summary.json
    results: [{instance_id, modes: [{mode, patch_path, lossy_meta}], ...}]
"""
import json
import sys
import hashlib
from pathlib import Path
from collections import defaultdict


def parse_cell_name(name):
    """t0.85_c3_01_k5_t0.85 → (0.85, 'c3_01')"""
    parts = name.split("_")
    threshold = float(parts[0].lstrip("t"))
    chunk = parts[1] + "_" + parts[2]
    return threshold, chunk


def aggregate(sweep_root):
    sweep_root = Path(sweep_root)
    if not sweep_root.exists():
        print(f"ERROR: {sweep_root} does not exist")
        return 1

    # {case_id: {threshold: (pk_sha, l_sha, anchor_pool_hit, copy_method)}}
    by_case = defaultdict(dict)

    cells = sorted([d for d in sweep_root.iterdir() if d.is_dir() and d.name.startswith("t")])
    total_patches = 0

    for cell in cells:
        threshold, chunk = parse_cell_name(cell.name)
        sf = cell / "summary.json"
        if not sf.exists():
            continue
        with open(sf) as f:
            data = json.load(f)
        for r in data.get("results", []):
            case_id = r["instance_id"]
            modes = {m["mode"]: m for m in r.get("modes", [])}
            pk = modes.get("placeholder_knn_lossy")
            lsy = modes.get("lossy")
            if not pk or not lsy:
                continue
            pk_path = Path(pk["patch_path"])
            lsy_path = Path(lsy["patch_path"])
            pk_sha = hashlib.sha1(pk_path.read_bytes()).hexdigest()[:8] if pk_path.exists() else None
            lsy_sha = hashlib.sha1(lsy_path.read_bytes()).hexdigest()[:8] if lsy_path.exists() else None
            meta = pk.get("lossy_meta", {})
            by_case[case_id][threshold] = {
                "pk_sha": pk_sha,
                "lsy_sha": lsy_sha,
                "pk_bytes": pk_path.stat().st_size if pk_path.exists() else 0,
                "lsy_bytes": lsy_path.stat().st_size if lsy_path.exists() else 0,
                "anchor_pool_hit": meta.get("placeholder_anchor_pool_hit_count", 0),
                "copy_method": meta.get("placeholder_knn_copy_method", "—"),
                "topk_sim_mean": meta.get("placeholder_knn_topk_similarity_mean", 0.0),
            }
            total_patches += 1

    # Report 1: per-threshold byte-equality (placeholder_knn_lossy vs in-cell lossy)
    n_within_eq = 0
    n_within_ne = 0
    for case_id, cells_data in by_case.items():
        for t, d in cells_data.items():
            if d["pk_sha"] == d["lsy_sha"]:
                n_within_eq += 1
            else:
                n_within_ne += 1

    # Report 2: per-case cross-threshold stability
    n_cross_eq = 0
    n_cross_ne = 0
    for case_id, cells_data in by_case.items():
        shas = {d["pk_sha"] for d in cells_data.values()}
        if len(shas) == 1:
            n_cross_eq += 1
        else:
            n_cross_ne += 1

    # Report 3: anchor pool populated?
    pool_populated = any(
        d["anchor_pool_hit"] > 0
        for cells_data in by_case.values()
        for d in cells_data.values()
    )

    # Report 4: cross-experiment byte-equality vs Phase 2 baseline
    phase2_lossy_root = Path("results/swe_percase_baseline_20260624T085604Z")
    phase2_lossy = {}
    if phase2_lossy_root.exists():
        for d in sorted(phase2_lossy_root.iterdir()):
            if not d.is_dir() or not (d / "summary.json").exists():
                continue
            case_id = d.name
            with open(d / "summary.json") as f:
                data = json.load(f)
            if not data.get("results"):
                continue
            modes = {m["mode"]: m for m in data["results"][0]["modes"]}
            lsy = modes.get("lossy")
            if lsy and Path(lsy["patch_path"]).exists():
                phase2_lossy[case_id] = hashlib.sha1(Path(lsy["patch_path"]).read_bytes()).hexdigest()[:8]

    phase2_v44_root = Path("results/swe_percase_v44_20260624T085604Z")
    phase2_v44 = {}
    if phase2_v44_root.exists():
        for d in sorted(phase2_v44_root.iterdir()):
            if not d.is_dir() or not (d / "summary.json").exists():
                continue
            case_id = d.name
            with open(d / "summary.json") as f:
                data = json.load(f)
            if not data.get("results"):
                continue
            modes = {m["mode"]: m for m in data["results"][0]["modes"]}
            pk = modes.get("placeholder_knn_lossy")
            if pk and Path(pk["patch_path"]).exists():
                phase2_v44[case_id] = hashlib.sha1(Path(pk["patch_path"]).read_bytes()).hexdigest()[:8]

    # Print report
    print(f"# Phase 3 FULL Sweep — Byte-Equality Report\n")
    print(f"Sweep root: {sweep_root}")
    print(f"Cells (threshold × chunk): {len(cells)}")
    print(f"Total (case × threshold) patches: {total_patches}")
    print(f"Unique cases: {len(by_case)}")
    print(f"")
    print(f"## Check 1: Within-cell byte-equality (placeholder_knn_lossy vs in-cell lossy)")
    print(f"Equal: {n_within_eq}/{total_patches}")
    print(f"Not equal: {n_within_ne}/{total_patches}")
    print(f"")
    print(f"## Check 2: Cross-threshold byte-equality (same case, all 6 thresholds)")
    print(f"Same SHA across thresholds: {n_cross_eq}/{len(by_case)} cases")
    print(f"Multiple SHAs: {n_cross_ne}/{len(by_case)} cases")
    print(f"")
    print(f"## Check 3: Anchor pool populated in any cell? {'**YES**' if pool_populated else 'NO'}")
    print(f"")

    # Detailed per-case table
    print(f"## Per-case × per-threshold table\n")
    print(f"| case_id | " + " | ".join(f"t={t}" for t in sorted({t for cells_data in by_case.values() for t in cells_data.keys()})) + " |")
    print(f"|---" * (1 + len({t for cells_data in by_case.values() for t in cells_data.keys()})) + "|")
    for case_id in sorted(by_case):
        cells_data = by_case[case_id]
        row = [case_id]
        for t in sorted(cells_data.keys()):
            d = cells_data[t]
            row.append(f"{d['pk_sha']}({d['anchor_pool_hit']})")
        print(f"| " + " | ".join(row) + " |")

    # Per-cell detail
    print(f"\n## Per-cell detailed byte-equality")
    for cell in cells:
        threshold, chunk = parse_cell_name(cell.name)
        sf = cell / "summary.json"
        if not sf.exists():
            continue
        with open(sf) as f:
            data = json.load(f)
        results = data.get("results", [])
        print(f"\n### threshold={threshold} chunk={chunk}")
        print(f"| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |")
        for r in results:
            modes = {m["mode"]: m for m in r.get("modes", [])}
            pk = modes.get("placeholder_knn_lossy")
            lsy = modes.get("lossy")
            if not pk or not lsy:
                continue
            pk_path = Path(pk["patch_path"])
            lsy_path = Path(lsy["patch_path"])
            pk_sha = hashlib.sha1(pk_path.read_bytes()).hexdigest()[:8] if pk_path.exists() else "—"
            lsy_sha = hashlib.sha1(lsy_path.read_bytes()).hexdigest()[:8] if lsy_path.exists() else "—"
            eq = "✅" if pk_sha == lsy_sha else "❌"
            meta = pk.get("lossy_meta", {})
            print(f"| {r['instance_id']} | {pk_path.stat().st_size if pk_path.exists() else 0} | {pk_sha} | {lsy_path.stat().st_size if lsy_path.exists() else 0} | {lsy_sha} | {eq} | {meta.get('placeholder_anchor_pool_hit_count', 0)} | {meta.get('placeholder_knn_copy_method', '—')} |")

    # Cross-experiment table
    if phase2_lossy or phase2_v44:
        print(f"\n## Cross-experiment byte-equality")
        print(f"| case_id | phase3 placeholder_knn_lossy (any t) | phase2 lossy | phase2 v44 | all equal? |")
        print(f"|---|---|---|---|---|")
        for case_id in sorted(set(by_case) & set(phase2_lossy) & set(phase2_v44)):
            shas_p3 = sorted({d["pk_sha"] for d in by_case[case_id].values()})
            p3_display = shas_p3[0] if len(shas_p3) == 1 else f"{shas_p3[0]}+{len(shas_p3)-1}"
            p2l = phase2_lossy[case_id]
            p2v = phase2_v44[case_id]
            eq = "✅" if all(s == p2l == p2v for s in shas_p3) else "❌"
            print(f"| {case_id} | {p3_display} | {p2l} | {p2v} | {eq} |")

    # Final summary
    print(f"\n## Final summary")
    if n_within_ne == 0 and n_cross_ne == 0:
        print(f"✅ **All {total_patches} placeholder_knn_lossy patches byte-identical to lossy baseline**")
        print(f"✅ **All {len(by_case)} cases have byte-identical placeholder_knn_lossy across 6 thresholds**")
        print(f"")
        if pool_populated:
            print(f"**Anchor pool populated** — v44 active path tested and produced byte-equal output")
        else:
            print(f"**Anchor pool NOT populated** — fallback invariance (regression = 0pp by definition)")
    else:
        print(f"❌ Regression detected: {n_within_ne} within-cell diffs, {n_cross_ne} cross-threshold diffs")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: aggregate_phase3_full.py <sweep_root>")
        sys.exit(1)
    sys.exit(aggregate(sys.argv[1]))