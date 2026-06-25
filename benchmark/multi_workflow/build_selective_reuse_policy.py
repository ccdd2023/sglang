#!/usr/bin/env python3
"""Build the selective AST reuse policy from AST granularity artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from benchmark.multi_workflow.selective_ast_reuse import (
    DEFAULT_AST_DISTANCE_PATH,
    build_selective_policy,
    read_ast_granularity_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ast-distance", type=Path, default=DEFAULT_AST_DISTANCE_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/selective_ast_reuse/data/selective_reuse_policy.json"),
    )
    parser.add_argument("--p90-threshold", type=float, default=0.45)
    parser.add_argument("--max-tail-rate", type=float, default=0.10)
    parser.add_argument("--extended", action="store_true",
                        help="Use the extended safe-granularity set (function/method/control_block/file_prefix).")
    args = parser.parse_args()

    summary = read_ast_granularity_summary(args.ast_distance)
    policy = build_selective_policy(
        summary,
        p90_threshold=args.p90_threshold,
        max_tail_rate=args.max_tail_rate,
        extended=args.extended,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Selective AST reuse policy",
        "",
        f"- Source: `{args.ast_distance}`",
        f"- p90 threshold: `{args.p90_threshold}`",
        f"- max tail rate: `{args.max_tail_rate}`",
        "",
        "| granularity | decision | p90 | max | tail>0.5 | retention tokens | reason |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for gran, item in policy["granularities"].items():
        lines.append(
            f"| `{gran}` | `{item['decision']}` | {item['p90']:.3f} | {item['max']:.3f} | "
            f"{item['tail_rate_gt_0_5']:.3f} | {item['retention_tokens']} | `{item['reason']}` |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"policy": str(args.out), "markdown": str(args.out.with_suffix(".md"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
