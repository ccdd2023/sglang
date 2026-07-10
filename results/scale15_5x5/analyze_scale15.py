#!/usr/bin/env python3
"""Scale-15 analysis: type_match/25 + FAIL_acc + per-case + bootstrap CIs.

Scores lossless / R32 / R38b on the 15 diverse cases and reports honest
fixed-denominator metrics + 95% bootstrap CIs on type_match.
"""
import json, sys, csv, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/lossy_alg_round38/scripts")))
from score_r38 import parse_verdict, classify_fail_reason, classify_patch, _patch_lookup, GT_PATH

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
GT = json.loads(GT_PATH.read_text())
RUNS = {
    "lossless": ROOT/"results/scale15_5x5/lossless/outputs.jsonl",
    "R32":      ROOT/"results/scale15_5x5/r32/outputs.jsonl",
    "R38b":     ROOT/"results/scale15_5x5/r38b/outputs.jsonl",
}
ROWS = {
    "lossless": ROOT/"results/scale15_5x5/lossless/rows.csv",
    "R32":      ROOT/"results/scale15_5x5/r32/rows.csv",
    "R38b":     ROOT/"results/scale15_5x5/r38b/rows.csv",
}

def score(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(rows)
    nfail = npass = nunk = 0
    per_case = defaultdict(lambda: [0,0])  # case -> [fail, agree]
    for r in rows:
        v = parse_verdict(r["output_text"])
        cid = r["case_id"]
        if v == "FAIL":
            nfail += 1
            mt = classify_fail_reason(r["output_text"])
            gtt = classify_patch(_patch_lookup(cid))
            per_case[cid][0] += 1
            if mt == gtt:
                per_case[cid][1] += 1
        elif v == "PASS": npass += 1
        else: nunk += 1
    type_match = sum(a for _, a in [(c, per_case[c][1]) for c in per_case])  # total agree
    # actually sum agrees
    total_agree = sum(pc[1] for pc in per_case.values())
    return {
        "n": n, "PASS": npass, "FAIL": nfail, "UNK": nunk,
        "type_match": total_agree,
        "type_match_over_25": total_agree / n if n else 0,
        "type_match_over_FAIL": total_agree / nfail if nfail else 0,
        "FAIL_acc": nfail / n if n else 0,  # GT all FAIL
        "per_case": {c: {"fail": v[0], "agree": v[1]} for c, v in per_case.items()},
    }

def ttft_reuse(csv_path):
    if not csv_path.exists(): return None
    ttfts, c2s, cas = [], [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k, lst in [("ttft_ms", ttfts), ("c2_chunk_reused_tokens", c2s), ("codeaware_reused_tokens", cas)]:
                try: lst.append(float(r[k]))
                except: pass
    return {
        "ttft_avg": sum(ttfts)/len(ttfts) if ttfts else 0,
        "c2_reused_avg": sum(c2s)/len(c2s) if c2s else 0,
        "n": len(ttfts),
    }

def bootstrap_ci(per_case, n_iter=10000, seed=42):
    """Bootstrap type_match/25 over cases. Resample cases with replacement."""
    rnd = random.Random(seed)
    cases = list(per_case.keys())
    if not cases: return (0,0)
    agrees = [per_case[c]["agree"] for c in cases]
    n_cases = len(cases)
    # each case has 5 agents; type_match/25 = sum(agree)/25 over resampled cases (5 agents each)
    n_total = n_cases * 5
    samples = []
    for _ in range(n_iter):
        s = sum(agrees[rnd.randrange(n_cases)] for _ in range(n_cases))
        samples.append(s / n_total)
    samples.sort()
    lo = samples[int(0.025*n_iter)]
    hi = samples[int(0.975*n_iter)]
    return (lo, hi)

import numpy as np
print(f"{'config':<12} {'n':>4} {'PASS':>5} {'FAIL':>5} {'UNK':>4} {'type_match':>11} {'/25':>7} {'/FAIL':>7} {'FAIL_acc':>9} {'TTFT':>7} {'c2_reuse':>9}")
print("-"*100)
summary = {}
for label, path in RUNS.items():
    if not path.exists():
        print(f"{label:<12} (not found: {path})"); continue
    s = score(path)
    t = ttft_reuse(ROWS[label]) or {}
    ci = bootstrap_ci(s["per_case"])
    s["ci_95"] = ci
    s["ttft_avg"] = t.get("ttft_avg", 0)
    s["c2_reused_avg"] = t.get("c2_reused_avg", 0)
    summary[label] = s
    print(f"{label:<12} {s['n']:>4} {s['PASS']:>5} {s['FAIL']:>5} {s['UNK']:>4} {s['type_match']:>11} "
          f"{s['type_match_over_25']*100:>6.1f}% {s['type_match_over_FAIL']*100:>6.1f}% {s['FAIL_acc']*100:>8.1f}% "
          f"{t.get('ttft_avg',0):>6.0f} {t.get('c2_reused_avg',0):>8.1f}")

print("\n=== 95% bootstrap CI on type_match/25 (resample 15 cases) ===")
for label, s in summary.items():
    lo, hi = s["ci_95"]
    print(f"  {label:<12}: {s['type_match_over_25']*100:.1f}%  CI=[{lo*100:.1f}%, {hi*100:.1f}%]")

print("\n=== per-case type_match (which cases drive the signal) ===")
cases = sorted(set(c for s in summary.values() for c in s["per_case"]))
hdr = f"{'case':<52}" + "".join(f"{l:>8}" for l in summary)
print(hdr)
for c in cases:
    row = f"{c[:52]:<52}"
    for label in summary:
        pc = summary[label]["per_case"].get(c, {"fail":0,"agree":0})
        row += f"  {pc['agree']}/{pc['fail']*1:<6}"
    print(row)

print("\n=== COMMON COMPLETE CASES (5 agents in ALL 3 configs) ===")
# a case is "complete" for a config if it has 5 FAIL verdicts (all 5 agents ran + said FAIL)
complete = {}
for label, s in summary.items():
    complete[label] = {c for c, pc in s["per_case"].items() if pc["fail"] == 5}
common = set.intersection(*complete.values()) if complete else set()
print(f"lossless complete: {len(complete.get('lossless',[]))} | R32: {len(complete.get('R32',[]))} | R38b: {len(complete.get('R38b',[]))}")
print(f"common complete (5 agents all 3): {len(common)}")
if common:
    print(f"\n{'config':<12} {'type_match':>10} {'/common':>9} {'FAIL_acc':>9}")
    common_sorted = sorted(common)
    for label, s in summary.items():
        tm = sum(s["per_case"][c]["agree"] for c in common_sorted)
        n_fail = sum(s["per_case"][c]["fail"] for c in common_sorted)
        n_total = len(common_sorted) * 5
        print(f"  {label:<12} {tm:>10} {tm/n_total*100:>8.1f}% {n_fail/n_total*100:>8.1f}%")
    # bootstrap CI on common set
    print("\n=== 95% bootstrap CI on type_match (common complete cases) ===")
    for label, s in summary.items():
        pc_common = {c: s["per_case"][c] for c in common_sorted}
        lo, hi = bootstrap_ci(pc_common)
        tm = sum(pc_common[c]["agree"] for c in common_sorted)
        n_total = len(common_sorted) * 5
        print(f"  {label:<12}: {tm}/{n_total} = {tm/n_total*100:.1f}%  CI=[{lo*100:.1f}%, {hi*100:.1f}%]")
    # per-family breakdown on common cases
    print("\n=== per-family type_match (common cases) ===")
    from collections import defaultdict
    fam = defaultdict(lambda: defaultdict(int))
    for c in common_sorted:
        fam_name = c.rsplit("__",1)[0].split(".")[-1]
        for label, s in summary.items():
            fam[fam_name][label] += s["per_case"][c]["agree"]
    print(f"{'family':<28}" + "".join(f"{l:>10}" for l in summary))
    for fn in sorted(fam):
        row = f"{fn[:28]:<28}"
        for label in summary:
            row += f"{fam[fn][label]:>10}"
        print(row)

out = ROOT/"results/scale15_5x5/scale15_summary.json"
out.write_text(json.dumps(summary, indent=2, default=str))
print(f"\nwrote {out}")
