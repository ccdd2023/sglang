#!/usr/bin/env python3
"""
Phase 2.1c (revised): aggregate per-case pass@1 from per-case driver output.

The per-case driver writes one summary.json per case (in
results/swe_percase_baseline_*/<case_id>/summary.json and same for v44).

A "pass" is candidate_test.returncode == 0 (only happens when --eval is
enabled; with --skip-candidate-tests, all rows are None). When skipping
candidate tests, we can still compare:
- diff_extracted (True if model produced a diff)
- synth_ok (True if patch synthesis succeeded)
- apply_rc (rc=0 means `git apply --check` succeeded)
- patch_bytes (size sanity)

Usage:
    python -m benchmark.multi_workflow.aggregate_per_case_pass_at_1 \
        --baseline-root results/swe_percase_baseline_* \
        --v44-root results/swe_percase_v44_* \
        --out-md results/per_case_pass_at_1_compare.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_root(root: Path) -> dict:
    """Return {case_id: {mode: stats_dict}} from a per-case root."""
    out = {}
    if not root.exists():
        return out
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith("_"):
            continue
        sj = case_dir / "summary.json"
        if not sj.exists():
            continue
        d = json.loads(sj.read_text())
        for case in d.get("results", []):
            cid = case["instance_id"]
            out[cid] = {}
            for m in case.get("modes", []):
                patch = Path(m["patch_path"]) if m.get("patch_path") else None
                out[cid][m["mode"]] = {
                    "diff_extracted": m["diff_extracted"],
                    "synth_ok": m["patch_synthesis"]["ok"],
                    "apply_rc": m.get("apply_check", {}).get("returncode"),
                    "patch_bytes": patch.stat().st_size if patch and patch.exists() else 0,
                    "first_match_reason": m.get("lossy_meta", {}).get("lossy_first_match_reason", ""),
                    "topk_sim": m.get("lossy_meta", {}).get("placeholder_knn_topk_similarity_mean", 0.0),
                    "copy_method": m.get("lossy_meta", {}).get("placeholder_knn_copy_method", ""),
                    "candidate_test_rc": (m.get("candidate_test", {}) or {}).get("returncode"),
                }
    return out


def render_table(base: dict, v44: dict) -> str:
    all_cases = sorted(set(base) | set(v44))
    modes = ["lossless", "lossy", "lossy_prefetch", "placeholder_knn_lossy"]
    headers = ["case_id", "repo"]
    for m in modes:
        headers += [f"{m}_base", f"{m}_v44", "eq"]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")

    for cid in all_cases:
        b_case = base.get(cid, {})
        v_case = v44.get(cid, {})
        # find repo from first available
        repo = ""
        for src in (b_case, v_case):
            for m in modes:
                if m in src and "repo" not in locals():
                    pass
        # collect per mode
        cells = [cid[:34] if len(cid) > 34 else cid, repo[:20]]
        for m in modes:
            b = b_case.get(m)
            v = v_case.get(m)
            if b is None and v is None:
                cells += ["-", "-", "-"]
                continue
            b_str = _fmt(b) if b else "-"
            v_str = _fmt(v) if v else "-"
            eq = "-"
            if b and v and b.get("patch_bytes") and v.get("patch_bytes"):
                eq = "EQUAL" if b["patch_bytes"] == v["patch_bytes"] else "DIFFER"
            cells += [b_str, v_str, eq]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _fmt(s: dict) -> str:
    """One-line summary of a case-mode entry."""
    bits = []
    if s.get("diff_extracted"):
        bits.append("ext")
    if s.get("synth_ok"):
        bits.append("syn")
    if s.get("apply_rc") == 0:
        bits.append("app✓")
    elif s.get("apply_rc") is None:
        bits.append("app?")
    elif s.get("apply_rc") == 128:
        bits.append("app✗")
    if s.get("candidate_test_rc") == 0:
        bits.append("pass✓")
    elif s.get("candidate_test_rc") is not None:
        bits.append(f"pass✗({s['candidate_test_rc']})")
    if s.get("topk_sim", 0) > 0:
        bits.append(f"sim={s['topk_sim']:.3f}")
    if s.get("copy_method") and s["copy_method"] not in ("", "none"):
        bits.append(f"copy={s['copy_method']}")
    return f"{s.get('patch_bytes', 0)}B[" + ",".join(bits) + "]"


def render_summary(base: dict, v44: dict) -> str:
    """Aggregate counts: how many cases had pass for each mode."""
    modes = ["lossless", "lossy", "lossy_prefetch", "placeholder_knn_lossy"]
    out = ["", "## Aggregate counts", ""]
    out.append("| mode | source | extracted | synth_ok | apply_pass | candidate_pass |")
    out.append("|---|---|---|---|---|---|")
    for m in modes:
        for src_label, src in [("baseline", base), ("v44", v44)]:
            counts = defaultdict(int)
            for cid, modes_d in src.items():
                if m in modes_d:
                    s = modes_d[m]
                    if s.get("diff_extracted"): counts["extracted"] += 1
                    if s.get("synth_ok"): counts["synth_ok"] += 1
                    if s.get("apply_rc") == 0: counts["apply_pass"] += 1
                    if s.get("candidate_test_rc") == 0: counts["candidate_pass"] += 1
            total = sum(1 for cid in src if m in src[cid])
            out.append(
                f"| {m} | {src_label} | {counts['extracted']}/{total} | "
                f"{counts['synth_ok']}/{total} | {counts['apply_pass']}/{total} | "
                f"{counts['candidate_pass']}/{total} |"
            )
    return "\n".join(out)


def render_byte_equality(base: dict, v44: dict, baseline_root: Path, v44_root: Path) -> str:
    """Check byte-equal between baseline and v44 patch files."""
    out = ["", "## Byte-equality: baseline patch vs v44 patch", ""]
    out.append("| case_id | mode | baseline | v44 | equal |")
    out.append("|---|---|---|---|---|")
    for cid in sorted(set(base) | set(v44)):
        for m in ["lossless", "lossy", "lossy_prefetch", "placeholder_knn_lossy"]:
            b_case = base.get(cid, {}).get(m)
            v_case = v44.get(cid, {}).get(m)
            if not b_case or not v_case:
                continue
            b_path = baseline_root / cid / f"{m}.patch"
            v_path = v44_root / cid / f"{m}.patch"
            if not (b_path.exists() and v_path.exists()):
                continue
            b_bytes = b_path.read_bytes()
            v_bytes = v_path.read_bytes()
            eq = "EQUAL" if b_bytes == v_bytes else "DIFFER"
            out.append(f"| {cid} | {m} | {len(b_bytes)}B | {len(v_bytes)}B | {eq} |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--v44-root", type=Path, required=True)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    base = load_root(args.baseline_root)
    v44 = load_root(args.v44_root)

    if not base and not v44:
        print("No data found in either root", file=sys.stderr)
        sys.exit(2)

    md = []
    md.append("# Phase 2.1 per-case pass@1 summary\n")
    md.append(f"baseline: `{args.baseline_root}`")
    md.append(f"v44:      `{args.v44_root}`")
    md.append("")
    md.append("Format: <bytes>B[ext,syn,app✓/✗,sim=X.XXX,copy=method]")
    md.append("")
    md.append("## Per-case × per-mode table\n")
    md.append(render_table(base, v44))
    md.append(render_summary(base, v44))
    md.append(render_byte_equality(base, v44, args.baseline_root, args.v44_root))

    text = "\n".join(md)
    print(text)
    if args.out_md:
        args.out_md.write_text(text)
        print(f"\nWrote: {args.out_md}", file=sys.stderr)
    if args.out_json:
        import json as _json
        args.out_json.write_text(_json.dumps({"baseline": base, "v44": v44}, indent=2))
        print(f"Wrote JSON: {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()