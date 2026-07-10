#!/usr/bin/env python3
"""Equal-budget ablation analysis for direction A (node-kind interface-recompute).

Compares, at matched recompute budget B, the accuracy (type_match) and TTFT of:
  lossless | R32-uniform (sweep 0.15/0.26/0.30/0.45) | R38b-position |
  node-kind-interface | node-kind-signature

Decisive question: does code structure (AST interface boundary) buy accuracy
over uniform/position at equal B? (R34 lacked this ablation -> dismissed as a
global FRAC bump.)

Per config measures:
  type_match   - from outputs.jsonl output_text via score_r38 (fixed /n denom)
  TTFT         - rows.csv ttft_ms
  B            - rows.csv placeholder_chunk_pool_total_tokens_dense (recompute cost)
  reuse        - rows.csv codeaware_reused_tokens (>0 sanity, hard constraint)
  node_kind_k  - rows.csv placeholder_chunk_pool_node_kind_k_count (fire-rate guard)

Auto-skips configs whose outputs.jsonl is missing (so a partial run still reports).
"""
import json, sys, csv, random
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
sys.path.insert(0, str(ROOT / "results/lossy_alg_round38/scripts"))
from score_r38 import parse_verdict, classify_fail_reason, classify_patch, _patch_lookup  # noqa

BASE = ROOT / "results/scale15_5x5"

# config label -> (outputs.jsonl, rows.csv). Order = sweep order for the table.
CONFIGS = [
    ("lossless",     BASE / "lossless/outputs.jsonl",     BASE / "lossless/rows.csv"),
    ("R32_f015",     BASE / "r32_f015/outputs.jsonl",     BASE / "r32_f015/rows.csv"),
    ("R32_f026",     BASE / "r32_f026/outputs.jsonl",     BASE / "r32_f026/rows.csv"),
    ("R32_f030",     BASE / "r32/outputs.jsonl",          BASE / "r32/rows.csv"),
    ("R32_f045",     BASE / "r32_f045/outputs.jsonl",     BASE / "r32_f045/rows.csv"),
    ("R38b",         BASE / "r38b/outputs.jsonl",         BASE / "r38b/rows.csv"),
    ("nodekind",     BASE / "nodekind/outputs.jsonl",     BASE / "nodekind/rows.csv"),
    ("nodekind_sig", BASE / "nodekind_sig/outputs.jsonl", BASE / "nodekind_sig/rows.csv"),
]


def score(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(rows)
    nfail = npass = nunk = 0
    per_case = defaultdict(lambda: [0, 0])  # case -> [fail, agree]
    for r in rows:
        v = parse_verdict(r["output_text"])
        cid = r["case_id"]
        if v == "FAIL":
            nfail += 1
            if classify_fail_reason(r["output_text"]) == classify_patch(_patch_lookup(cid)):
                per_case[cid][1] += 1
            per_case[cid][0] += 1
        elif v == "PASS":
            npass += 1
        else:
            nunk += 1
    agree = sum(pc[1] for pc in per_case.values())
    return {
        "n": n, "PASS": npass, "FAIL": nfail, "UNK": nunk,
        "type_match": agree,
        "type_match_pct": agree / n if n else 0.0,   # fixed /n denominator
        "per_case": {c: {"fail": v[0], "agree": v[1]} for c, v in per_case.items()},
    }


def rows_stats(csv_path: Path):
    if not csv_path.exists():
        return {}
    cols = ["ttft_ms", "codeaware_reused_tokens", "c2_chunk_reused_tokens",
            "placeholder_chunk_pool_total_tokens_dense",
            "placeholder_chunk_pool_hit_count",
            "placeholder_chunk_pool_node_kind_k_count"]
    acc = {c: [] for c in cols}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for c in cols:
                try:
                    acc[c].append(float(r[c]))
                except (KeyError, ValueError):
                    pass
    out = {}
    for c, vals in acc.items():
        if vals:
            out[c + "_avg"] = sum(vals) / len(vals)
            out[c + "_sum"] = sum(vals)
    return out


def bootstrap_ci(per_case, n_iter=10000, seed=42):
    rnd = random.Random(seed)
    cases = list(per_case.keys())
    if not cases:
        return (0.0, 0.0)
    agrees = [per_case[c]["agree"] for c in cases]
    nc = len(cases)
    n_total = nc * 5
    samples = sorted(sum(agrees[rnd.randrange(nc)] for _ in range(nc)) / n_total
                     for _ in range(n_iter))
    return (samples[int(0.025 * n_iter)], samples[int(0.975 * n_iter)])


def main():
    import numpy as np  # noqa
    summary = {}
    for label, oj, cj in CONFIGS:
        if not oj.exists():
            print(f"{label:<14} (not found: {oj.name})")
            continue
        s = score(oj)
        st = rows_stats(cj)
        s["ci_95"] = bootstrap_ci(s["per_case"])
        s["ttft_avg"] = st.get("ttft_ms_avg", 0.0)
        # Budget proxy: c2_chunk_reused_tokens (per-request body tokens copied).
        # B (recompute) = total_chunk_tokens - c2_reused, and total is constant
        # across configs (same hit set) -> equal c2_reused <=> equal B.
        # placeholder_chunk_pool_total_tokens_dense is CUMULATIVE + counts
        # decisions not tokens, so unusable for per-request budget.
        s["c2_reused_avg"] = st.get("c2_chunk_reused_tokens_avg", 0.0)
        s["reuse_avg"] = st.get("codeaware_reused_tokens_avg", 0.0)
        s["hit_avg"] = st.get("placeholder_chunk_pool_hit_count_avg", 0.0)
        s["nodekind_k_avg"] = st.get("placeholder_chunk_pool_node_kind_k_count_avg", 0.0)
        s["nodekind_k_sum"] = st.get("placeholder_chunk_pool_node_kind_k_count_sum", 0.0)
        s["hit_sum"] = st.get("placeholder_chunk_pool_hit_count_sum", 0.0)
        summary[label] = s

    print(f"\n{'config':<14} {'n':>4} {'type_m':>7} {'/n%':>6} {'CI95':>14} "
          f"{'TTFT':>6} {'c2_reuse':>9} {'reuse':>7} {'nk_k':>6} {'fire%':>6}")
    print("-" * 95)
    for label, s in summary.items():
        lo, hi = s["ci_95"]
        fire = (100.0 * s["nodekind_k_sum"] / s["hit_sum"]) if s["hit_sum"] else 0.0
        print(f"{label:<14} {s['n']:>4} {s['type_match']:>7} {s['type_match_pct']*100:>5.1f}% "
              f"[{lo*100:>4.1f},{hi*100:>4.1f}] {s['ttft_avg']:>6.0f} {s['c2_reused_avg']:>9.0f} "
              f"{s['reuse_avg']:>7.0f} {s['nodekind_k_avg']:>6.0f} {fire:>5.1f}%")

    # ---- node-kind fire-rate guard (R34 no-op check) ----
    print("\n=== node-kind fire-rate guard (R34 no-op check) ===")
    for label in ("nodekind", "nodekind_sig"):
        s = summary.get(label)
        if not s:
            continue
        fire = (100.0 * s["nodekind_k_sum"] / s["hit_sum"]) if s["hit_sum"] else 0.0
        verdict = "OK (fires on most hits)" if fire > 50 else "WARNING: low fire rate (possible no-op)"
        print(f"  {label:<14}: node_kind_k_sum={s['nodekind_k_sum']:.0f} / hit_sum={s['hit_sum']:.0f} "
              f"= {fire:.1f}%  {verdict}")

    # ---- Pareto: type_match vs reuse (c2_reused = body copied; higher = less
    # recompute B = faster, since B = total - c2_reused and total is constant).
    # Maximize c2_reused AND type_match. Sort by c2_reused desc (fastest first);
    # a config is on the frontier if its type_match beats all faster configs.
    print("\n=== Pareto: type_match% vs reuse (c2_reused; higher=faster/lower-B) ===")
    pts = [(lbl, s["c2_reused_avg"], s["type_match_pct"] * 100, s["ttft_avg"])
           for lbl, s in summary.items() if lbl != "lossless" and s["c2_reused_avg"] > 0]
    pts.sort(key=lambda p: -p[1])  # most reuse first
    print(f"{'config':<14} {'c2_reuse':>9} {'type_m%':>8} {'TTFT':>6} {'Pareto?':>8}")
    frontier = []
    best_acc = -1
    for lbl, reuse, acc, ttft in pts:
        on_frontier = acc > best_acc
        if on_frontier:
            frontier.append(lbl)
            best_acc = acc
        mark = "<==" if on_frontier else ("*node*" if lbl.startswith("nodekind") else "")
        print(f"{lbl:<14} {reuse:>9.0f} {acc:>7.1f}% {ttft:>6.0f} {mark:>8}")
    print(f"  Pareto frontier: {frontier}")

    # ---- Vertical slice: node-kind vs R32 at matched reuse (== matched B) ----
    print("\n=== vertical slice: node-kind vs R32 at matched reuse (== equal B) ===")
    nk = summary.get("nodekind")
    if nk and nk["c2_reused_avg"] > 0:
        r32s = [(lbl, abs(s["c2_reused_avg"] - nk["c2_reused_avg"]), s)
                for lbl, s in summary.items() if lbl.startswith("R32_")]
        r32s.sort(key=lambda x: x[1])
        if r32s:
            clbl, _, cs = r32s[0]
            print(f"  nodekind      : reuse={nk['c2_reused_avg']:.0f}  type_match={nk['type_match_pct']*100:.1f}%  "
                  f"TTFT={nk['ttft_avg']:.0f}")
            print(f"  {clbl:<13}: reuse={cs['c2_reused_avg']:.0f}  type_match={cs['type_match_pct']*100:.1f}%  "
                  f"TTFT={cs['ttft_avg']:.0f}  (closest R32, d_reuse={cs['c2_reused_avg']-nk['c2_reused_avg']:.0f})")
            d = nk["type_match_pct"] - cs["type_match_pct"]
            verdict = ("node-kind WINS at equal B -> code structure buys accuracy"
                       if d > 0.005 else
                       "node-kind LOSES/ties at equal B -> falsification (per-chunk "
                       "adaptive boundary has no gain)")
            print(f"  -> type_match delta = {d*100:+.1f}pp.  {verdict}")
    else:
        print("  (nodekind config not run yet)")

    # ---- common complete cases (generalized to all run configs) ----
    print("\n=== common complete cases (5 FAIL agents in ALL run configs) ===")
    complete = {lbl: {c for c, pc in s["per_case"].items() if pc["fail"] == 5}
                for lbl, s in summary.items()}
    common = set.intersection(*complete.values()) if complete else set()
    print(f"  common complete cases: {len(common)} / configs: {list(summary)}")
    if common:
        cs = sorted(common)
        nt = len(cs) * 5
        print(f"  {'config':<14} {'type_m':>7} {'/common%':>9} {'TTFT':>6} {'reuse':>7}")
        for lbl, s in summary.items():
            tm = sum(s["per_case"][c]["agree"] for c in cs)
            print(f"  {lbl:<14} {tm:>7} {tm/nt*100:>8.1f}% {s['ttft_avg']:>6.0f} {s['c2_reused_avg']:>7.0f}")

    out = ROOT / "results/scale15_5x5/ablation_nodekind_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
