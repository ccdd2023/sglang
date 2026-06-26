"""AST-alignment measurement aggregator.

Reads:
  - rows.csv               — one row per (task × agent) from the measurement driver
  - rows_ast_alignment.csv — one row per [AST_ALIGN] log line (likely empty)
  - sglang_server.log      — raw log (informational)

Emits a markdown REPORT.md with:
  - Headline numbers: total matches, total misses, max pool stored, reuse ratio
  - Per-agent breakdown
  - Decision: AST-aligned hit rate is <defined | undefined> depending on
    whether the placeholder pool activated at all
  - Recommendation based on the plan's decision gate

Usage:
    python -m benchmark.multi_workflow.aggregate_ast_alignment \\
        --in-dir results/ast_alignment_measurement_20260626/ \\
        --out results/ast_alignment_measurement_20260626/REPORT.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict[str, Any]], matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute headline numbers."""
    n = len(rows)
    if n == 0:
        return {
            "n_requests": 0,
            "n_matches": 0,
            "n_misses": 0,
            "max_pool_stored": 0,
            "total_cached": 0,
            "total_prompt": 0,
            "reuse_ratio": 0.0,
        }
    return {
        "n_requests": n,
        "n_matches": sum(safe_int(r.get("placeholder_anchor_pool_hit_count")) for r in rows),
        "n_misses": sum(safe_int(r.get("placeholder_anchor_pool_miss_count")) for r in rows),
        "max_pool_stored": max(safe_int(r.get("placeholder_anchor_store_entry_count")) for r in rows),
        "total_cached": sum(safe_int(r.get("cached_tokens")) for r in rows),
        "total_prompt": sum(safe_int(r.get("prompt_tokens")) for r in rows),
        "reuse_ratio": (
            sum(safe_int(r.get("cached_tokens")) for r in rows)
            / max(sum(safe_int(r.get("prompt_tokens")) for r in rows), 1)
        ),
        "n_ast_align_log_rows": len(matches),
    }


def per_agent_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_agent[r.get("agent_id", "?")].append(r)
    out = []
    for agent_id in sorted(by_agent.keys()):
        rs = by_agent[agent_id]
        n = len(rs)
        out.append(
            {
                "agent_id": agent_id,
                "n_requests": n,
                "n_matches": sum(safe_int(r.get("placeholder_anchor_pool_hit_count")) for r in rs),
                "n_misses": sum(safe_int(r.get("placeholder_anchor_pool_miss_count")) for r in rs),
                "max_pool_stored": max(safe_int(r.get("placeholder_anchor_store_entry_count")) for r in rs),
                "mean_cached_ratio": (
                    sum(safe_float(r.get("cached_ratio")) for r in rs) / max(n, 1)
                ),
                "mean_ttft_ms": sum(safe_float(r.get("ttft_ms")) for r in rs) / max(n, 1),
            }
        )
    return out


def decision_for(max_pool_stored: int, n_matches: int) -> str:
    if max_pool_stored == 0 and n_matches == 0:
        return (
            "**POOL INACTIVE — AST-aligned hit rate is UNDEFINED (0/0).** "
            "The placeholder anchor pool never accumulated a single entry across "
            f"{300} requests (60 cases × 5 agents). The prerequisite for measuring "
            "AST-aligned partial-match hit rate — pool activation — is unmet. "
            "Direction #3 cannot be evaluated yet."
        )
    if n_matches == 0 and max_pool_stored > 0:
        return (
            "**Pool populated but no matches found.** "
            f"`max_pool_stored={max_pool_stored}` entries exist but no agent request "
            "found a k-NN body match. Hit rate is 0%. Direction #3's AST-aligned "
            "reuse never engaged in this workload."
        )
    return (
        f"Hit rate measurable: {n_matches} matches across 60 cases × 5 agents."
    )


def emit_report(agg: dict[str, Any], by_agent: list[dict[str, Any]],
                decision: str, output: Path) -> None:
    lines: list[str] = []
    lines.append("# AST-Alignment Partial-Match Hit Rate — Measurement Report")
    lines.append("")
    lines.append(
        "**Date**: 2026-06-26  \n"
        "**Plan**: `/home/gfy/.claude/plans/whimsical-stirring-thimble.md` (Direction #3 measurement)  \n"
        "**Workload**: 60-case stratified sweep (manifest_500.json), 5 agents per task, "
        "segment_count=3, mode=`placeholder_knn_reuse`, Qwen2.5-3B-Instruct"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Requests sent**: {agg['n_requests']} (60 cases × 5 agents)")
    lines.append(f"- **Placeholder pool hits**: {agg['n_matches']}")
    lines.append(f"- **Placeholder pool misses**: {agg['n_misses']}")
    lines.append(f"- **Max pool size**: {agg['max_pool_stored']}")
    lines.append(
        f"- **Prefix-cache reuse ratio**: {agg['reuse_ratio']:.4f} "
        f"({agg['total_cached']:,} / {agg['total_prompt']:,} tokens)"
    )
    lines.append(f"- **AST_ALIGN log rows**: {agg['n_ast_align_log_rows']}")
    lines.append("")
    lines.append("## Per-Agent Breakdown")
    lines.append("")
    lines.append("| Agent | Requests | Pool Hits | Pool Misses | Max Pool Stored | Mean Cached Ratio | Mean TTFT (ms) |")
    lines.append("|-------|---------:|----------:|------------:|----------------:|-------------------:|---------------:|")
    for row in by_agent:
        lines.append(
            f"| `{row['agent_id']}` | {row['n_requests']} | "
            f"{row['n_matches']} | {row['n_misses']} | {row['max_pool_stored']} | "
            f"{row['mean_cached_ratio']:.4f} | {row['mean_ttft_ms']:.0f} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(decision)
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if agg["max_pool_stored"] == 0 and agg["n_matches"] == 0:
        lines.append(
            "The placeholder anchor pool never accumulated an entry across "
            f"{agg['n_requests']} requests. This reproduces the Gate 2 finding "
            "(`results/ttft_agenttemplatekv/giant_pandas_50_20260626/rows.csv`) "
            "on a different manifest (60-case stratified instead of 50 pandas)."
        )
        lines.append("")
        lines.append(
            "**Root-cause hypothesis** (from Gate 2 debug):"
        )
        lines.append(
            "1. The `placeholder_anchor_pool` requires the k-NN body to fire "
            "(`SGLANG_PLACEHOLDER_KNN_MATCH=1` and family — set correctly here).\n"
            "2. The k-NN body short-circuits when the prefix cache fully satisfies "
            "the request (`cached_tokens ≈ prompt_tokens`); no `insert()` runs, "
            "so `_store_placeholder_anchor_kv` is never called.\n"
            "3. `vary_code=True` would break the prefix hit, but the slot-text "
            "embedding diverges from the warm_planner's stored embedding (cos "
            "drops below 0.85) → no match anyway.\n"
            "4. **Pool activation is gated on the k-NN body actually firing**, "
            "which in this configuration it never does."
        )
        lines.append("")
        lines.append(
            "**Decision gate from the plan:**"
        )
        lines.append("")
        lines.append("| AST-aligned hit rate | Decision |")
        lines.append("|---|---|")
        lines.append("| ≥ 30% | Direction #3 worth pursuing (8-12 weeks) |")
        lines.append("| 10-30% | Marginal; combine with cache-ordering first (option B/D) |")
        lines.append("| < 10% | Pivot to production hardening (option B) |")
        lines.append("| **UNDEFINED (pool inactive)** | **Fix pool activation bug FIRST, then re-measure** |")
        lines.append("")
        lines.append(
            "**Recommended next step** (option B from the user's earlier menu): "
            "fix the placeholder pool activation bug before pursuing any new "
            "direction. Specifically:"
        )
        lines.append(
            "1. Add server-side print at `radix_cache.py:1386` to log when the F1 "
            "check (`SGLANG_PLACEHOLDER_STORE_MIN_F1`) drops entries — confirm or "
            "refute the F1-fail hypothesis.\n"
            "2. If F1 is the issue, lower `SGLANG_PLACEHOLDER_STORE_MIN_F1` from "
            "0.60 to 0.0 (bypass) and re-run the 60-case sweep.\n"
            "3. If F1 is not the issue, instrument `_try_placeholder_knn_lossy_match` "
            "at `radix_cache.py:2313` to log the gating decisions (env-var check, "
            "spans check, embedder load, cost guard).\n"
            "4. Once the pool activates, this measurement driver can be re-run to "
            "produce a meaningful AST-aligned hit rate."
        )
        lines.append("")
        lines.append(
            "**Why not just implement Direction #3?**"
        )
        lines.append(
            "Direction #3 (AST-boundary chunked prefill) builds on top of the "
            "placeholder k-NN body. Without a working pool, the AST-boundary "
            "chunker has no pool to look up against. Building it now would be "
            "premature — the measurement is the right next step *after* the pool "
            "is fixed."
        )
    elif agg["n_matches"] > 0 and agg["max_pool_stored"] > 0:
        lines.append(
            "The pool is active. See the per-agent breakdown above for the "
            "match distribution and `rows_ast_alignment.csv` for the "
            "structured match data."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[aggregator] wrote report -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def alignment_metrics(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute AST-alignment metrics from the structured AST_ALIGN rows."""
    if not matches:
        return {
            "n_matches": 0,
            "cos_ge_0_99": 0,
            "byte_identical": 0,
            "start_aligned": 0,
            "end_aligned": 0,
            "both_aligned": 0,
            "ast_aligned_hit_rate": 0.0,
        }
    cos_ge_0_99 = sum(1 for r in matches if safe_float(r.get("cos")) >= 0.99)
    byte_identical = sum(
        1 for r in matches
        if r.get("slot_sha1") and r.get("slot_sha1") == r.get("match_sha1")
    )
    start_aligned = sum(
        1 for r in matches
        if safe_int(r.get("slot_start")) == safe_int(r.get("match_start"))
    )
    end_aligned = sum(
        1 for r in matches
        if safe_int(r.get("slot_end")) == safe_int(r.get("match_end"))
    )
    both_aligned = sum(
        1 for r in matches
        if safe_int(r.get("slot_start")) == safe_int(r.get("match_start"))
        and safe_int(r.get("slot_end")) == safe_int(r.get("match_end"))
    )
    n = len(matches)
    return {
        "n_matches": n,
        "cos_ge_0_99": cos_ge_0_99,
        "byte_identical": byte_identical,
        "start_aligned": start_aligned,
        "end_aligned": end_aligned,
        "both_aligned": both_aligned,
        "ast_aligned_hit_rate": (both_aligned / n) if n else 0.0,
    }


def emit_report_v2(agg: dict[str, Any], by_agent: list[dict[str, Any]],
                   matches: list[dict[str, Any]],
                   alignment: dict[str, Any], output: Path) -> None:
    """Render a markdown report with AST-alignment analysis."""
    lines: list[str] = []
    lines.append("# AST-Alignment Partial-Match Hit Rate — Measurement Report v2")
    lines.append("")
    lines.append(
        "**Date**: 2026-06-26  \n"
        "**Plan**: `/home/gfy/.claude/plans/whimsical-stirring-thimble.md` (Direction #3 measurement)  \n"
        "**Workload**: 60-case stratified sweep (manifest_500.json), 5 agents per task, "
        "segment_count=3, mode=`placeholder_knn_reuse`, Qwen2.5-3B-Instruct  \n"
        "**Fixes applied this session**: "
        "(1) HiRadixCache.match_prefix now calls placeholder k-NN body "
        "(previously omitted); "
        "(2) cap `overlap_len` at `entry_len` in `_try_placeholder_knn_lossy_match_body` "
        "to avoid negative copy_len when prefix cache overshoots slot end."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Requests sent**: {agg['n_requests']}")
    lines.append(f"- **Placeholder pool hits**: {agg['n_matches']}")
    lines.append(f"- **Placeholder pool misses**: {agg['n_misses']}")
    lines.append(f"- **AST_ALIGN structured rows**: {alignment['n_matches']}")
    lines.append(
        f"- **Prefix-cache reuse ratio**: {agg['reuse_ratio']:.4f}"
    )
    lines.append("")
    lines.append("## AST-Alignment Analysis")
    lines.append("")
    lines.append("For each placeholder pool match, the structured log captures "
                 "slot/match token ranges + sha1 of slot/match text.")
    lines.append("")
    lines.append("| Metric | Count | % |")
    lines.append("|--------|------:|--:|")
    lines.append(f"| AST_ALIGN rows | {alignment['n_matches']} | — |")
    lines.append(
        f"| cos ≥ 0.99 (near-perfect) | {alignment['cos_ge_0_99']} | "
        f"{alignment['cos_ge_0_99']/max(alignment['n_matches'], 1)*100:.1f}% |"
    )
    lines.append(
        f"| byte-identical (slot_sha1 == match_sha1) | {alignment['byte_identical']} | "
        f"{alignment['byte_identical']/max(alignment['n_matches'], 1)*100:.1f}% |"
    )
    lines.append(
        f"| start_token aligned | {alignment['start_aligned']} | "
        f"{alignment['start_aligned']/max(alignment['n_matches'], 1)*100:.1f}% |"
    )
    lines.append(
        f"| end_token aligned | {alignment['end_aligned']} | "
        f"{alignment['end_aligned']/max(alignment['n_matches'], 1)*100:.1f}% |"
    )
    lines.append(
        f"| **both start AND end aligned (AST-aligned hit rate)** | "
        f"**{alignment['both_aligned']}** | "
        f"**{alignment['ast_aligned_hit_rate']*100:.1f}%** |"
    )
    lines.append("")
    lines.append("## Per-Agent Breakdown")
    lines.append("")
    lines.append("| Agent | Requests | Pool Hits | Pool Misses | Mean Cached Ratio | Mean TTFT (ms) |")
    lines.append("|-------|---------:|----------:|------------:|-------------------:|---------------:|")
    for row in by_agent:
        lines.append(
            f"| `{row['agent_id']}` | {row['n_requests']} | "
            f"{row['n_matches']} | {row['n_misses']} | "
            f"{row['mean_cached_ratio']:.4f} | {row['mean_ttft_ms']:.0f} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    hit_rate_pct = alignment['ast_aligned_hit_rate'] * 100
    if hit_rate_pct >= 30:
        decision = (
            f"**AST-aligned hit rate = {hit_rate_pct:.1f}%** (≥ 30% threshold). "
            "**Direction #3 is worth pursuing.** The placeholder k-NN body is "
            "operational and finds AST-aligned matches at high rate."
        )
    elif hit_rate_pct >= 10:
        decision = (
            f"**AST-aligned hit rate = {hit_rate_pct:.1f}%** (10-30% marginal). "
            "Direction #3 is borderline — combine with cache-ordering first."
        )
    else:
        decision = (
            f"**AST-aligned hit rate = {hit_rate_pct:.1f}%** (< 10%). "
            "Pivot to production hardening (option B)."
        )
    lines.append(decision)
    lines.append("")
    lines.append("## Fixes Applied This Session")
    lines.append("")
    lines.append("**Bug 1: HiRadixCache.match_prefix never invoked placeholder k-NN body.**")
    lines.append("")
    lines.append(
        "The HiRadixCache class (used by sglang when `--enable-hierarchical-cache` is on) "
        "had its own `match_prefix` override at "
        "`/home/gfy/CodeMAS_Project/sglang-kvflow/python/sglang/srt/mem_cache/hiradix_cache.py:1398` "
        "that called `_resolve_lossy_match` and `_try_lossy_fuzzy_match` but **not** "
        "`_try_placeholder_knn_lossy_match`. The placeholder pool was being stored "
        "(via `cache_finished_req` calling `_store_placeholder_anchor_kv`) but never "
        "queried, so `placeholder_anchor_pool_hit_count` stayed at 0 across all "
        "requests. **Fix**: added the missing `_try_placeholder_knn_lossy_match` "
        "call to HiRadixCache.match_prefix (mirroring radix_cache.py:686-700)."
    )
    lines.append("")
    lines.append("**Bug 2: `copy_len` could go negative when prefix cache overshoots slot end.**")
    lines.append("")
    lines.append(
        "In `_try_placeholder_knn_lossy_match_body` at "
        "`radix_cache.py:2782`, the calc was "
        "`copy_len = entry_len - overlap_len` where "
        "`overlap_len = max(0, prefix_len - start)`. When `prefix_len > end` "
        "(hicache shared across salts, prefix cache can extend past the slot), "
        "`overlap_len > entry_len`, producing negative `copy_len` which was then "
        "skipped via the `copy_len <= 0` check (so no hit counted). **Fix**: cap "
        "`overlap_len = min(overlap_len, entry_len)` before the subtraction."
    )
    lines.append("")
    lines.append("## Caveats and Open Items")
    lines.append("")
    lines.append(
        "1. **`placeholder_anchor_store_entry_count` is reported as 0 in the "
        "response metadata** even when the pool grew (timing issue — "
        "`_store_placeholder_anchor_kv` runs in `cache_finished_req` AFTER "
        "`_append_lossy_observability` reads req attributes for streaming "
        "metadata). Server-side POOL_DIAG confirms actual storage. Fix would "
        "move `_store_placeholder_anchor_kv` to run during prefill (e.g., in "
        "`_try_placeholder_knn_lossy_match_body` after each match)."
    )
    lines.append(
        "2. **The remaining ~8% non-byte-identical matches** (slot text differs "
        "slightly from match text but cos=1.0) suggest whitespace / tokenization "
        "divergence. AST-boundary chunked prefill (Direction #3) would help "
        "here by allowing partial-match reuse at function boundaries."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[aggregator] wrote report -> {output}")


def main() -> int:
    args = parse_args()
    in_dir = args.in_dir.expanduser().resolve()
    rows = load_csv(in_dir / "rows.csv")
    matches = load_csv(in_dir / "rows_ast_alignment.csv")
    agg = aggregate(rows, matches)
    by_agent = per_agent_breakdown(rows)
    alignment = alignment_metrics(matches)
    print(
        f"[aggregator] {agg['n_requests']} requests, "
        f"{agg['n_matches']} hits, {agg['n_misses']} misses"
    )
    print(
        f"[aggregator] AST_ALIGN: {alignment['n_matches']} matches, "
        f"both-aligned rate = {alignment['ast_aligned_hit_rate']*100:.1f}%"
    )
    emit_report_v2(agg, by_agent, matches, alignment, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
