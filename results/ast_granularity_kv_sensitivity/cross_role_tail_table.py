#!/usr/bin/env python3
"""Per-span tail analysis for the cross-role AST-granularity verdict.

The cross-role view in Table tab:ast-granularity-cross-role reports
class tail = 20% (12 of 60 cells with d_norm > 0.5), statement_window
tail = 13.3% (8 of 60), control_block / method / file_prefix tail
in 6.7-8.3%. This script asks: which (path, granularity) pairs
actually drive those tail counts?

Output: results/ast_granularity_kv_sensitivity/cross_role_tail.md
        results/ast_granularity_kv_sensitivity/data/cross_role_tail.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/ast_granularity_kv_sensitivity/data/ast_granularity_distance_7b.json"
OUT_JSON = ROOT / "results/ast_granularity_kv_sensitivity/data/cross_role_tail.json"
OUT_MD = ROOT / "results/ast_granularity_kv_sensitivity/cross_role_tail.md"

THRESHOLD = 0.5


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", flush=True)
        return 1
    d = json.loads(SRC.read_text())
    rec = d["records"]
    by_gran_path: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_gran: dict[str, list[dict]] = defaultdict(list)
    for r in rec:
        if r["agent_role"] == "planner":
            continue
        by_gran_path[(r["granularity"], r["path"])].append(r)
        by_gran[r["granularity"]].append(r)
    out: dict = {}
    for (g, p), rows in by_gran_path.items():
        n = len(rows)
        tail = [r for r in rows if r["d_norm"] > THRESHOLD]
        if not tail:
            continue
        max_row = max(rows, key=lambda r: r["d_norm"])
        out[(g, p)] = {
            "n_cells": n,
            "tail_cells": len(tail),
            "tail_rate": round(len(tail) / n, 4) if n else 0.0,
            "max_d_norm": round(max_row["d_norm"], 4),
            "max_role": max_row["agent_role"],
            "max_span_tokens": max_row["span_tokens"],
            "max_start_line": max_row["start_line"],
            "max_end_line": max_row["end_line"],
            "repo": max_row.get("repo") or "",
        }
    sorted_keys = sorted(out.keys(), key=lambda k: (out[k]["max_d_norm"]), reverse=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    serialised = {f"{g}|{p}": v for (g, p), v in out.items()}
    OUT_JSON.write_text(json.dumps(serialised, indent=2), encoding="utf-8")
    lines = ["# Cross-role tail spans (d_norm > 0.5), by (granularity, path)\n"]
    lines.append(f"Source: ast_granularity_distance_7b.json, cross-role cells only (planner excluded).\n")
    lines.append(f"Threshold: d_norm > {THRESHOLD}.\n")
    lines.append("Sorted by max cross-role d_norm, descending.\n")
    lines.append("\n| Granularity | Path | n cells | tail cells | tail rate | max d_norm | max role | span tokens | lines | repo |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---|---|")
    for (g, p) in sorted_keys[:25]:
        s = out[(g, p)]
        lines.append(
            f"| {g} | {p} | {s['n_cells']} | {s['tail_cells']} | {s['tail_rate']*100:.1f}% | "
            f"{s['max_d_norm']:.3f} | {s['max_role']} | {s['max_span_tokens']} | "
            f"{s['max_start_line']}-{s['max_end_line']} | {s['repo']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
