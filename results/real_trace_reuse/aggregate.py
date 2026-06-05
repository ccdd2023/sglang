"""Aggregate the SWE-bench replay log into a real-trace reuse report.

Reads results/real_trace_reuse/data/replay_log.jsonl and produces:
  - results/real_trace_reuse/report.md
  - results/real_trace_reuse/data/swe_bench_aggregate.json
  - results/real_trace_reuse/plots/{hit_rate_per_agent_pair, cache_savings_per_task}.png
  - results/real_trace_reuse/plots/modifier_calibration.png  (predicted vs. d_norm from store)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
DATA_DIR = PROJECT_ROOT / "results" / "real_trace_reuse" / "data"
PLOT_DIR = PROJECT_ROOT / "results" / "real_trace_reuse" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _per_agent_pair_stats(records: list[dict]) -> dict:
    """For each pair of agents (e.g., planner→coder), count matches where
    the second request was allowed lossy reuse against the first."""
    by_instance: dict = defaultdict(list)
    for r in records:
        by_instance[r["instance_id"]].append(r)
    pairs = [("planner", "coder"), ("coder", "reviewer"),
             ("planner", "reviewer"), ("planner", "planner"),
             ("coder", "coder"), ("reviewer", "reviewer")]
    out = {}
    for src, dst in pairs:
        n = 0
        n_allowed = 0
        n_matched_content = 0
        for inst, rs in by_instance.items():
            # find first request by src agent and any by dst agent
            src_recs = [r for r in rs if r["agent"] == src]
            dst_recs = [r for r in rs if r["agent"] == dst]
            if not src_recs or not dst_recs:
                continue
            n += 1
            for dr in dst_recs:
                if dr.get("lossy_final_reuse_allowed") is True:
                    n_allowed += 1
                if dr.get("lossy_first_matched_content_signature"):
                    n_matched_content += 1
        out[f"{src}→{dst}"] = {
            "n_pairs": n,
            "n_lossy_allowed": n_allowed,
            "n_matched_content": n_matched_content,
            "hit_rate": round(n_matched_content / max(n, 1), 3),
        }
    return out


def _cache_savings(records: list[dict]) -> dict:
    """Sum lossy_anchor_match_len across all records, by instance."""
    by_instance: dict = defaultdict(lambda: {"n_records": 0, "n_lossy": 0, "tokens_saved": 0})
    for r in records:
        inst = r["instance_id"]
        by_instance[inst]["n_records"] += 1
        if r.get("lossy_anchor_match_used") is True:
            by_instance[inst]["n_lossy"] += 1
            by_instance[inst]["tokens_saved"] += int(r.get("lossy_anchor_match_len", 0) or 0)
    total_records = sum(s["n_records"] for s in by_instance.values())
    total_lossy = sum(s["n_lossy"] for s in by_instance.values())
    total_saved = sum(s["tokens_saved"] for s in by_instance.values())
    return {
        "n_instances": len(by_instance),
        "n_records": total_records,
        "n_lossy_matches": total_lossy,
        "tokens_saved": total_saved,
        "hit_rate_overall": round(total_lossy / max(total_records, 1), 3),
    }


def _modifier_calibration(records: list[dict]) -> dict:
    """Cross-tabulate lossy_predicted_distance with the actual rope_delta
    as a proxy. In the absence of a real d_norm ground truth, this is a
    sanity check that the modifier didn't under/over-predict on real data."""
    pred = [r.get("lossy_predicted_distance") for r in records
            if r.get("lossy_predicted_distance") is not None]
    rope = [r.get("lossy_anchor_rope_delta") for r in records
            if r.get("lossy_anchor_rope_delta") is not None]
    conf = [r.get("lossy_context_aware_confidence") for r in records
            if r.get("lossy_context_aware_confidence") is not None]
    mult = [r.get("lossy_context_aware_multiplier") for r in records
            if r.get("lossy_context_aware_multiplier") is not None]
    return {
        "n_with_pred": len(pred),
        "n_with_rope": len(rope),
        "n_with_conf": len(conf),
        "n_with_mult": len(mult),
        "pred_mean": round(sum(pred) / max(1, len(pred)), 3) if pred else None,
        "conf_mean": round(sum(conf) / max(1, len(conf)), 3) if conf else None,
        "mult_mean": round(sum(mult) / max(1, len(mult)), 3) if mult else None,
    }


def _plot_hit_rate(stats: dict, out_path: Path) -> None:
    pairs = list(stats.keys())
    rates = [stats[p]["hit_rate"] for p in pairs]
    n_pairs = [stats[p]["n_pairs"] for p in pairs]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(range(len(pairs)), rates, color="#3a86ff", edgecolor="black")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.invert_yaxis()
    ax.set_xlabel("hit rate (content_signature matched)")
    ax.set_title("SWE-bench replay — KVCOMM hit rate by agent pair")
    for i, (rate, n) in enumerate(zip(rates, n_pairs)):
        ax.text(rate + 0.01, i, f"{rate:.1%} (n={n})", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_modifier_calibration(records: list[dict], out_path: Path) -> None:
    pred = [r["lossy_predicted_distance"] for r in records
            if r.get("lossy_predicted_distance") is not None]
    conf = [r["lossy_context_aware_confidence"] for r in records
            if r.get("lossy_context_aware_confidence") is not None]
    if not pred or not conf:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred, conf, alpha=0.5, s=10)
    ax.set_xlabel("lossy_predicted_distance (from table)")
    ax.set_ylabel("lossy_context_aware_confidence (applied)")
    ax.set_title("Modifier calibration on SWE-bench replay")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_savings(savings: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["total lossy matches", "tokens saved"],
           [savings["n_lossy_matches"], savings["tokens_saved"] / 1000.0],
           color=["#3a86ff", "#fb5607"])
    ax.set_ylabel("count / thousands of tokens")
    ax.set_title("SWE-bench replay — overall KVCOMM contribution")
    for i, v in enumerate([savings["n_lossy_matches"], savings["tokens_saved"] / 1000.0]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _md_table(rows, cols):
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    out = [head, sep]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=str(DATA_DIR / "replay_log.jsonl"))
    p.add_argument("--out-json", default=str(DATA_DIR / "swe_bench_aggregate.json"))
    p.add_argument("--out-md", default=str(DATA_DIR.parent / "report.md"))
    args = p.parse_args()

    records = _load_log(Path(args.log))
    if not records:
        print(f"[aggregate] no log at {args.log}; run replay_server.py first")
        return
    print(f"[aggregate] loaded {len(records)} records")

    pair_stats = _per_agent_pair_stats(records)
    savings = _cache_savings(records)
    cal = _modifier_calibration(records)

    summary = {
        "n_records": len(records),
        "agent_pair_stats": pair_stats,
        "cache_savings": savings,
        "modifier_calibration": cal,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_hit_rate(pair_stats, PLOT_DIR / "hit_rate_per_agent_pair.png")
    _plot_savings(savings, PLOT_DIR / "cache_savings_per_task.png")
    _plot_modifier_calibration(records, PLOT_DIR / "modifier_calibration.png")

    md = ["# Real-Trace Reuse Rate (SWE-bench Verified)\n"]
    md.append(f"- **Total records**: {len(records)}")
    md.append(f"- **Distinct instances**: {savings['n_instances']}\n")
    md.append("## 1. Hit rate by agent pair\n")
    rows = []
    for p, s in pair_stats.items():
        rows.append({
            "pair": p, "n": s["n_pairs"],
            "lossy_allowed": s["n_lossy_allowed"],
            "matched_content": s["n_matched_content"],
            "hit_rate": f"{s['hit_rate']:.1%}",
        })
    md.append(_md_table(rows, [("pair", "agent pair"), ("n", "n pairs"),
                                 ("lossy_allowed", "lossy allowed"),
                                 ("matched_content", "matched content sig"),
                                 ("hit_rate", "hit rate")]))
    md.append("\n## 2. Cache savings (overall)\n")
    md.append(_md_table([{
        "n_instances": savings["n_instances"],
        "n_records": savings["n_records"],
        "n_lossy": savings["n_lossy_matches"],
        "tokens_saved": savings["tokens_saved"],
        "hit_rate": f"{savings['hit_rate_overall']:.1%}",
    }], [("n_instances", "instances"), ("n_records", "records"),
          ("n_lossy", "lossy matches"), ("tokens_saved", "tokens saved"),
          ("hit_rate", "hit rate overall")]))
    md.append("\n## 3. Modifier calibration\n")
    md.append(_md_table([{
        "n_pred": cal["n_with_pred"],
        "pred_mean": cal["pred_mean"],
        "n_conf": cal["n_with_conf"],
        "conf_mean": cal["conf_mean"],
        "n_mult": cal["n_with_mult"],
        "mult_mean": cal["mult_mean"],
    }], [("n_pred", "n with predicted_distance"),
          ("pred_mean", "mean predicted_d"),
          ("n_conf", "n with confidence"),
          ("conf_mean", "mean confidence"),
          ("n_mult", "n with multiplier"),
          ("mult_mean", "mean multiplier")]))
    md.append("\n## 4. Plots\n")
    for f in sorted(PLOT_DIR.glob("*.png")):
        md.append(f"- ![](plots/{f.name})")
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[aggregate] wrote {args.out_json}")
    print(f"[aggregate] wrote {args.out_md}")


if __name__ == "__main__":
    main()
