#!/usr/bin/env python3
"""R27 A/B analyzer: Coder-3B × 3 agents, lossy vs lossless. Compare vs R26 + R19."""
import csv, json, re, statistics, subprocess
from pathlib import Path

P = re.compile(r"\bVERDICT:\s*(PASS|FAIL)\b", re.I)
BASE = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/lossy_alg_round27")
LOSSY_DIR = BASE / "r27_coder3b_3agent"
LOSSLESS_DIR = BASE / "r27_coder3b_3agent_lossless"
GT_PATH = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/lossy_alg_round21/ground_truth.json")


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
    gt = json.loads(GT_PATH.read_text())
    lossy_out = load_outputs(LOSSY_DIR)
    lossless_out = load_outputs(LOSSLESS_DIR)
    lossy_rows = list(csv.DictReader(open(LOSSY_DIR / "rows.csv")))
    lossless_rows = list(csv.DictReader(open(LOSSLESS_DIR / "rows.csv")))

    lossy_reusers = [r for r in lossy_rows if int(r.get("agent_idx", 0)) > 1]
    lossless_reusers = [r for r in lossless_rows if int(r.get("agent_idx", 0)) > 1]

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

    def verdicts_with_gt(out_dict, rows):
        unk, pass_n, fail_n, fail_correct = 0, 0, 0, 0
        for r in rows:
            cid = r["case_id"]
            aid = int(r["agent_idx"])
            text = out_dict.get((cid, aid), "")
            v = verdict(text)
            if v is None:
                unk += 1
            elif v == "PASS":
                pass_n += 1
            elif v == "FAIL":
                fail_n += 1
                if gt.get(cid) == "FAIL":
                    fail_correct += 1
        return {
            "n": len(rows),
            "unk": unk,
            "unk_pct": unk / len(rows) * 100,
            "pass": pass_n,
            "fail": fail_n,
            "gt_fail_n": sum(1 for r in rows if gt.get(r["case_id"]) == "FAIL"),
            "fail_correct": fail_correct,
            "fail_acc": fail_correct / max(1, sum(1 for r in rows if gt.get(r["case_id"]) == "FAIL")) * 100,
        }

    lossy_v = verdicts_with_gt(lossy_out, lossy_rows)
    lossless_v = verdicts_with_gt(lossless_out, lossless_rows)

    # Per-agent speedup
    lossless_by_key = {(r["case_id"], int(r["agent_idx"])): float(r["ttft_ms"]) for r in lossless_reusers}

    def avg(rows, col):
        return statistics.mean([float(r.get(col, 0) or 0) for r in rows])

    print("=" * 70)
    print("R27 RESULTS: Qwen2.5-Coder-3B-Instruct × 3 agents")
    print("=" * 70)
    print()
    print("TTFT (reusers, agent_idx > 1):")
    print(f"  Lossy:    mean={lossy_ttft['mean']:.1f}ms  p50={lossy_ttft['p50']:.1f}ms  p90={lossy_ttft['p90']:.1f}ms  n={lossy_ttft['n']}")
    print(f"  Lossless: mean={lossless_ttft['mean']:.1f}ms  p50={lossless_ttft['p50']:.1f}ms  p90={lossless_ttft['p90']:.1f}ms  n={lossless_ttft['n']}")
    print(f"  Speedup: mean={lossless_ttft['mean']/lossy_ttft['mean']:.3f}x  p50={lossless_ttft['p50']/lossy_ttft['p50']:.3f}x")
    print()
    print("Verdict quality (vs ground truth — all 5 cases are FAIL):")
    print(f"  Lossy:    PASS={lossy_v['pass']}  FAIL={lossy_v['fail']}  UNK={lossy_v['unk']}  ({lossy_v['unk_pct']:.1f}%)  FAIL_acc={lossy_v['fail_acc']:.1f}%")
    print(f"  Lossless: PASS={lossless_v['pass']}  FAIL={lossless_v['fail']}  UNK={lossless_v['unk']}  ({lossless_v['unk_pct']:.1f}%)  FAIL_acc={lossless_v['fail_acc']:.1f}%")
    print()
    print("Code-aware reuse (lossy reusers):")
    print(f"  radix_prefix: {avg(lossy_reusers, 'radix_prefix_tokens'):.1f}")
    print(f"  codeaware:    {avg(lossy_reusers, 'codeaware_reused_tokens'):.1f}")
    print(f"  c2_chunk:     {avg(lossy_reusers, 'c2_chunk_reused_tokens'):.1f}")
    print()
    print("Per-agent TTFT (lossy vs lossless):")
    for a in sorted(set(int(r["agent_idx"]) for r in lossy_reusers)):
        l = [float(r["ttft_ms"]) for r in lossy_reusers if int(r["agent_idx"]) == a]
        ll = [float(r["ttft_ms"]) for r in lossless_reusers if int(r["agent_idx"]) == a]
        if l and ll:
            print(f"  Agent {a}: lossy={statistics.mean(l):.1f}ms  lossless={statistics.mean(ll):.1f}ms  speedup={statistics.mean(ll)/statistics.mean(l):.3f}x")

    out = {
        "speedup_mean": lossless_ttft["mean"] / lossy_ttft["mean"],
        "speedup_p50": lossless_ttft["p50"] / lossy_ttft["p50"],
        "speedup_p90": lossless_ttft["p90"] / lossy_ttft["p90"],
        "lossy_fail_acc": lossy_v["fail_acc"],
        "lossless_fail_acc": lossless_v["fail_acc"],
        "lossy_unk_pct": lossy_v["unk_pct"],
        "lossless_unk_pct": lossless_v["unk_pct"],
        "lossy_pass": lossy_v["pass"],
        "lossy_fail": lossy_v["fail"],
        "lossless_pass": lossless_v["pass"],
        "lossless_fail": lossless_v["fail"],
        "codeaware_avg": avg(lossy_reusers, "codeaware_reused_tokens"),
    }
    (BASE / "ab_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {BASE / 'ab_results.json'}")


if __name__ == "__main__":
    main()