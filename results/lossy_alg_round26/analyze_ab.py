#!/usr/bin/env python3
"""R26 A/B analyzer: 3B model × 3 agents, lossy vs lossless."""
import csv, json, re, statistics
from pathlib import Path

P = re.compile(r"\bVERDICT:\s*(PASS|FAIL)\b", re.I)
BASE = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/lossy_alg_round26")
LOSSY_DIR = BASE / "r26_3b_3agent"
LOSSLESS_DIR = BASE / "r26_3b_3agent_lossless"


def load_outputs(d):
    out = {}
    p = d / "outputs.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["case_id"], r["agent_idx"])] = r.get("output_text", "")
    return out


def verdict(text):
    m = P.search(text or "")
    return m.group(1).upper() if m else None


def main():
    lossy_out = load_outputs(LOSSY_DIR)
    lossless_out = load_outputs(LOSSLESS_DIR)
    lossy_rows = list(csv.DictReader(open(LOSSY_DIR / "rows.csv")))
    lossless_rows = list(csv.DictReader(open(LOSSLESS_DIR / "rows.csv")))

    # Filter to reusers (agent_idx > 1) for fair comparison
    lossy_reusers = [r for r in lossy_rows if int(r.get("agent_idx", 0)) > 1]
    lossless_reusers = [r for r in lossless_rows if int(r.get("agent_idx", 0)) > 1]

    # TTFT speedup (reusers only)
    def ttft_stats(rows):
        vals = [float(r["ttft_ms"]) for r in rows]
        return {
            "mean": statistics.mean(vals),
            "p50": statistics.median(vals),
            "p90": statistics.quantiles(vals, n=10)[8] if len(vals) >= 10 else max(vals),
            "n": len(vals),
        }

    lossy_ttft = ttft_stats(lossy_reusers)
    lossless_ttft = ttft_stats(lossless_reusers)

    # UNK rate
    def unk_rate(out_dict, rows):
        unk = 0
        verdicts = {}
        for r in rows:
            cid = r["case_id"]
            aid = int(r["agent_idx"])
            text = out_dict.get((cid, aid), "")
            v = verdict(text)
            verdicts[(cid, aid)] = v
            if v is None:
                unk += 1
        return unk, len(rows), verdicts

    lossy_unk, lossy_n, lossy_vmap = unk_rate(lossy_out, lossy_rows)
    lossless_unk, lossless_n, lossless_vmap = unk_rate(lossless_out, lossless_rows)

    # Accuracy agreement (only on (case, agent) pairs where both produced verdicts)
    matches = 0
    total = 0
    mismatches = []
    for k in lossy_vmap:
        if k in lossless_vmap:
            l = lossy_vmap[k]
            ll = lossless_vmap[k]
            if l is not None and ll is not None:
                total += 1
                if l == ll:
                    matches += 1
                else:
                    mismatches.append((k, l, ll))

    # Per-agent TTFT (use dict lookup)
    lossless_by_key = {(r["case_id"], int(r["agent_idx"])): float(r["ttft_ms"]) for r in lossless_reusers}

    # Code-aware reuse breakdown
    def avg(rows, col):
        return statistics.mean([float(r.get(col, 0) or 0) for r in rows])

    print("=" * 70)
    print("R26 A/B RESULTS: 3B × 3 agents × verdict task")
    print("=" * 70)
    print()
    print("TTFT (reusers only, agent_idx > 1):")
    print(f"  Lossy:   mean={lossy_ttft['mean']:.1f}ms  p50={lossy_ttft['p50']:.1f}ms  p90={lossy_ttft['p90']:.1f}ms  n={lossy_ttft['n']}")
    print(f"  Lossless: mean={lossless_ttft['mean']:.1f}ms  p50={lossless_ttft['p50']:.1f}ms  p90={lossless_ttft['p90']:.1f}ms  n={lossless_ttft['n']}")
    print(f"  Speedup: mean={lossless_ttft['mean']/lossy_ttft['mean']:.3f}x  p50={lossless_ttft['p50']/lossy_ttft['p50']:.3f}x  p90={lossless_ttft['p90']/lossy_ttft['p90']:.3f}x")
    print()
    print("UNK garbage rate:")
    print(f"  Lossy:   {lossy_unk}/{lossy_n} = {lossy_unk/lossy_n*100:.1f}%")
    print(f"  Lossless: {lossless_unk}/{lossless_n} = {lossless_unk/lossless_n*100:.1f}%")
    print()
    print("Accuracy agreement vs lossless:")
    print(f"  Matches: {matches}/{total} = {matches/max(1,total)*100:.1f}%")
    print()
    print("Cached tokens (reusers only):")
    print(f"  Lossy radix_prefix:    {avg(lossy_reusers, 'radix_prefix_tokens'):.1f}")
    print(f"  Lossy codeaware:       {avg(lossy_reusers, 'codeaware_reused_tokens'):.1f}")
    print(f"  Lossy c2_chunk:        {avg(lossy_reusers, 'c2_chunk_reused_tokens'):.1f}")
    print(f"  Lossless radix_prefix: {avg(lossless_reusers, 'radix_prefix_tokens'):.1f}")
    print()
    print("Per-agent TTFT (ms) — lossy vs lossless:")
    for a in sorted(set(int(r["agent_idx"]) for r in lossy_reusers)):
        l = [float(r["ttft_ms"]) for r in lossy_reusers if int(r["agent_idx"]) == a]
        ll_match = [float(r["ttft_ms"]) for r in lossless_reusers if int(r["agent_idx"]) == a]
        if l and ll_match:
            print(f"  Agent {a}: lossy mean={statistics.mean(l):.1f}  lossless mean={statistics.mean(ll_match):.1f}  speedup={statistics.mean(ll_match)/statistics.mean(l):.3f}x")

    print()
    print("=" * 70)
    print("R19 BEST comparison (7B-Coder × 5 agents): 1.29× + 8% UNK + 80% agreement")
    print("=" * 70)

    out = {
        "lossy_ttft_mean": lossy_ttft["mean"],
        "lossless_ttft_mean": lossless_ttft["mean"],
        "speedup_mean": lossless_ttft["mean"] / lossy_ttft["mean"],
        "speedup_p50": lossless_ttft["p50"] / lossy_ttft["p50"],
        "speedup_p90": lossless_ttft["p90"] / lossy_ttft["p90"],
        "lossy_unk_pct": lossy_unk / lossy_n * 100,
        "lossless_unk_pct": lossless_unk / lossless_n * 100,
        "accuracy_agreement_pct": matches / max(1, total) * 100,
        "n_pairs": total,
        "lossy_codeaware_avg": avg(lossy_reusers, "codeaware_reused_tokens"),
        "lossy_c2_chunk_avg": avg(lossy_reusers, "c2_chunk_reused_tokens"),
        "r19_baseline": "1.29× + 8% UNK + 80% agreement (7B-Coder × 5 agents)",
    }
    (BASE / "ab_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {BASE / 'ab_results.json'}")


if __name__ == "__main__":
    main()