#!/usr/bin/env python3
"""P1'' v2 paired analysis: broader definition of "complete case".

The original (v1) required ALL 5 agents to say FAIL. With 12 cases having
5/5 verdicts (2 all-FAIL, 1 all-PASS, 9 mixed), v1 only sees 2 vs lossless's
3 = small intersection.

v2 uses BROADER inclusion: any case where all 5 agents RAN (= 5 verdicts in
outputs.jsonl, regardless of FAIL/PASS/UNKNOWN). This is the honest
"every agent finished" definition. With OOM at task 7, 2 cases (task 7 and
task 15) are incomplete in R32_f045; the other 13 are "ran" (12 fully + 1
partial = 12 truly complete).

Per-case metric: AGREE = max count of identical verdicts among 5 agents
(unanimity degree). With 5/5 same verdict = 5; with 4/1 = 4; with 3/2 = 3.

Then paired test: per case, agree(R32_f045) - agree(lossless). Positive
means R32_f045 has higher consensus.
"""
import json
import sys
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
sys.path.insert(0, str(ROOT / "results/lossy_alg_round38/scripts"))
from score_r38 import parse_verdict  # noqa


def score_per_case_broad(path):
    """Return {case_id: {'agree_max': max_verdict_count, 'verdicts': [...],
    'n_agents': N}} for cases where N=5 (all 5 agents ran)."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    per_case = defaultdict(list)
    for r in rows:
        per_case[r["case_id"]].append(parse_verdict(r["output_text"]))
    out = {}
    for c, vs in per_case.items():
        if len(vs) == 5:
            cnt = Counter(vs)
            out[c] = {
                "verdicts": vs,
                "n_agents": 5,
                "agree_max": cnt.most_common(1)[0][1],
                "all_same": len(cnt) == 1,
                "verdict_set": tuple(sorted(cnt.keys())),
            }
    return out


def main():
    lossless = score_per_case_broad(ROOT / "results/scale15_5x5/lossless/outputs.jsonl")
    r32_f045 = score_per_case_broad(ROOT / "results/scale15_5x5/r32_f045/outputs.jsonl")
    r32_f030 = score_per_case_broad(ROOT / "results/scale15_5x5/r32/outputs.jsonl")
    r32_f026 = score_per_case_broad(ROOT / "results/scale15_5x5/r32_f026/outputs.jsonl")
    r32_f015 = score_per_case_broad(ROOT / "results/scale15_5x5/r32_f015/outputs.jsonl")

    print(f"lossless: {len(lossless)} cases with 5/5 agents ran")
    print(f"R32_f045: {len(r32_f045)} cases with 5/5 agents ran")
    print(f"R32_f030: {len(r32_f030)} cases with 5/5 agents ran")
    print(f"R32_f026: {len(r32_f026)} cases with 5/5 agents ran")
    print(f"R32_f015: {len(r32_f015)} cases with 5/5 agents ran")

    common = sorted(set(lossless) & set(r32_f045))
    print(f"\nlossless & R32_f045 common complete (5 agents in BOTH): {len(common)}")
    if common:
        print(f"\n{'case':<60} {'lossless':>10} {'R32_f045':>10} {'delta':>7}")
        for c in common:
            l = lossless[c]["agree_max"]
            r = r32_f045[c]["agree_max"]
            print(f"{c[:60]:<60} {l:>10}/5 {r:>10}/5 {r-l:+d}")
        # Mean delta
        deltas = [r32_f045[c]["agree_max"] - lossless[c]["agree_max"] for c in common]
        mean_d = sum(deltas) / len(deltas)
        print(f"\nmean agree delta (R32_f045 - lossless) over {len(common)} cases = {mean_d:+.2f}")
        # Bootstrap CI on mean
        rnd = random.Random(42)
        samples = sorted(sum(rnd.choice(deltas) for _ in deltas) / len(deltas)
                         for _ in range(10000))
        lo, hi = samples[250], samples[9750]
        print(f"bootstrap 95% CI on mean delta = [{lo:+.2f}, {hi:+.2f}]")
        # Wilcoxon
        try:
            from scipy.stats import wilcoxon
            if len(set(deltas)) > 1:
                stat, p = wilcoxon(deltas, alternative="two-sided")
                print(f"Wilcoxon signed-rank p = {p:.3f}")
            else:
                print(f"Wilcoxon: all deltas identical ({set(deltas)}) -> n/a")
        except ImportError:
            print("(scipy not installed; skipping Wilcoxon)")

        # Verdict-set agreement (qualitative): does R32_f045 ever disagree on the
        # majority verdict vs lossless?
        print(f"\nVerdict-set disagreement (cases where lossless and R32_f045 disagree "
              f"on the majority verdict):")
        n_disagree_major = 0
        for c in common:
            l_cnt = Counter(lossless[c]["verdicts"])
            r_cnt = Counter(r32_f045[c]["verdicts"])
            l_major = l_cnt.most_common(1)[0][0]
            r_major = r_cnt.most_common(1)[0][0]
            if l_major != r_major:
                n_disagree_major += 1
                print(f"  {c[:60]}: lossless={l_major}({l_cnt[l_major]}/5) vs "
                      f"R32_f045={r_major}({r_cnt[r_major]}/5)")
        print(f"  {n_disagree_major}/{len(common)} cases disagree on majority verdict")

    # Cross-config table: which R32 sweep point is closest to lossless?
    print("\n=== R32 sweep vs lossless (common complete cases 5/5) ===")
    r32_sweep = {"R32_f015": r32_f015, "R32_f026": r32_f026,
                 "R32_f030": r32_f030, "R32_f045": r32_f045}
    print(f"{'config':<10} {'n_common':>9} {'mean_delta':>11} {'mean|delta|':>11} {'disagree_major':>15}")
    for name, cfg in r32_sweep.items():
        cm = sorted(set(lossless) & set(cfg))
        if not cm:
            print(f"{name:<10} {0:>9} {'-':>11} {'-':>11} {'-':>15}")
            continue
        deltas = [cfg[c]["agree_max"] - lossless[c]["agree_max"] for c in cm]
        abs_deltas = [abs(d) for d in deltas]
        n_disagree = sum(1 for c in cm
                         if Counter(lossless[c]["verdicts"]).most_common(1)[0][0]
                         != Counter(cfg[c]["verdicts"]).most_common(1)[0][0])
        print(f"{name:<10} {len(cm):>9} {sum(deltas)/len(deltas):>+10.2f} "
              f"{sum(abs_deltas)/len(abs_deltas):>10.2f} {n_disagree:>13}/{len(cm)}")


if __name__ == "__main__":
    main()