#!/usr/bin/env python3
"""P1'' paired analysis: R32_f045 vs lossless on cases both ran.

The standard bootstrap CI (per-config) treats each case as independent sample,
but ignores the matched-case pairing. A paired test on common complete cases
removes case-level variance and gives a TIGHTER CI on the difference.

Also computes the per-case agreement pattern: does R32_f045 win on some cases
and lossless on others (case-difficulty effect), or are they consistently
within ±1 of each other (no systematic effect)?

Method:
  - Common complete cases = cases with 5 FAIL agents in BOTH configs
  - Per case: type_match (agree count) per config
  - Paired delta = type_match(lossless, c) - type_match(R32_f045, c)
  - Wilcoxon signed-rank + sign test (small n -> non-parametric)
  - Bootstrap CI on the mean delta (cases resampled with replacement)
"""
import json
import sys
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
sys.path.insert(0, str(ROOT / "results/lossy_alg_round38/scripts"))
from score_r38 import parse_verdict  # noqa


def score_per_case(path):
    """Return {case_id: agree_count} for cases where ALL 5 agents said FAIL."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    per_case = defaultdict(lambda: {"fail": 0, "agree": 0})
    for r in rows:
        v = parse_verdict(r["output_text"])
        cid = r["case_id"]
        if v == "FAIL":
            per_case[cid]["fail"] += 1
            per_case[cid]["agree"] += 1  # FAIL counts as a verdict match here
    return {c: pc for c, pc in per_case.items() if pc["fail"] == 5}


def main():
    lossless = score_per_case(ROOT / "results/scale15_5x5/lossless/outputs.jsonl")
    r32_f045 = score_per_case(ROOT / "results/scale15_5x5/r32_f045/outputs.jsonl")
    r32_f030 = score_per_case(ROOT / "results/scale15_5x5/r32/outputs.jsonl")
    r32_f015 = score_per_case(ROOT / "results/scale15_5x5/r32_f015/outputs.jsonl")
    r32_f026 = score_per_case(ROOT / "results/scale15_5x5/r32_f026/outputs.jsonl")

    print(f"lossless complete (5/5 FAIL): {len(lossless)} cases")
    print(f"R32_f045 complete (5/5 FAIL): {len(r32_f045)} cases")
    print(f"R32_f030 complete (5/5 FAIL): {len(r32_f030)} cases")
    print(f"R32_f026 complete (5/5 FAIL): {len(r32_f026)} cases")
    print(f"R32_f015 complete (5/5 FAIL): {len(r32_f015)} cases")

    common_lossless_f045 = sorted(set(lossless) & set(r32_f045))
    common_lossless_f030 = sorted(set(lossless) & set(r32_f030))
    common_all_r32 = sorted(set(lossless) & set(r32_f015) & set(r32_f026)
                            & set(r32_f030) & set(r32_f045))
    print(f"\nlossless & R32_f045 common complete: {len(common_lossless_f045)}")
    print(f"lossless & R32_f030 common complete: {len(common_lossless_f030)}")
    print(f"lossless & ALL R32 common complete:  {len(common_all_r32)}")

    def paired_table(common_cases, configs, label):
        """configs: list of (name, per_case_dict). Show per-case agree and deltas."""
        if not common_cases:
            print(f"\n=== {label}: no common complete cases ===")
            return
        n = len(common_cases)
        print(f"\n=== {label} ({n} common cases) ===")
        # Header
        hdr = f"{'case':<60}" + "".join(f"{n:>10}" for n, _ in configs) + "  Δ vs lossless"
        print(hdr)
        deltas = {n: [] for n, _ in configs if n != "lossless"}
        rows_for_stats = []
        for c in common_cases:
            short = c[:60]
            line = f"{short:<60}"
            tm = {}
            for name, pc in configs:
                a = pc[c]["agree"]
                tm[name] = a
                line += f"{a:>10}/5"
            for n, _ in configs:
                if n != "lossless":
                    d = tm[n] - tm["lossless"]
                    deltas[n].append(d)
            line += f"  {tm.get(list(d for n,d in configs if n != 'lossless')[0],0) - tm['lossless']:+d}" \
                    if False else ""  # skip inline; below
            print(line)
        # Summary stats
        for n, ds in deltas.items():
            mean_d = sum(ds) / len(ds)
            abs_d = sum(abs(d) for d in ds) / len(ds)
            print(f"  {n}: mean delta vs lossless = {mean_d:+.2f} agree/case; "
                  f"mean |delta| = {abs_d:.2f}")
            rows_for_stats.append((n, ds))
        # Wilcoxon signed-rank (paired, no normal assumption)
        try:
            from scipy.stats import wilcoxon, binomtest
            have_scipy = True
        except ImportError:
            have_scipy = False
        if have_scipy:
            for n, ds in rows_for_stats:
                if len(set(ds)) <= 1:
                    print(f"  {n}: Wilcoxon: all deltas identical ({set(ds)}) -> p=n/a")
                    continue
                try:
                    stat, p = wilcoxon(ds, alternative="two-sided")
                    n_pos = sum(1 for d in ds if d > 0)
                    n_neg = sum(1 for d in ds if d < 0)
                    n_zero = sum(1 for d in ds if d == 0)
                    # Sign test (binomial)
                    n_nz = n_pos + n_neg
                    if n_nz > 0:
                        sign_p = binomtest(min(n_pos, n_neg), n_nz, 0.5,
                                           alternative="two-sided").pvalue
                    else:
                        sign_p = 1.0
                    print(f"  {n}: Wilcoxon p={p:.3f} (n_pos={n_pos}, n_neg={n_neg}, "
                          f"n_zero={n_zero}), Sign test p={sign_p:.3f}")
                except Exception as e:
                    print(f"  {n}: Wilcoxon failed: {e}")
        # Bootstrap CI on mean delta
        print(f"\n  bootstrap CI (n=10000) on mean delta vs lossless:")
        rnd = random.Random(42)
        for n, ds in rows_for_stats:
            samples = sorted(sum(rnd.choice(ds) for _ in ds) / len(ds)
                             for _ in range(10000))
            lo, hi = samples[250], samples[9750]
            print(f"    {n}: mean delta CI = [{lo:+.2f}, {hi:+.2f}] agree/case")

    paired_table(common_lossless_f045,
                 [("lossless", lossless), ("R32_f045", r32_f045)],
                 "P1'' target: R32_f045 vs lossless")
    paired_table(common_lossless_f030,
                 [("lossless", lossless), ("R32_f030", r32_f030)],
                 "reference: R32_f030 vs lossless")
    paired_table(common_all_r32,
                 [("lossless", lossless),
                  ("R32_f015", r32_f015),
                  ("R32_f026", r32_f026),
                  ("R32_f030", r32_f030),
                  ("R32_f045", r32_f045)],
                 "full R32 sweep vs lossless")


if __name__ == "__main__":
    main()