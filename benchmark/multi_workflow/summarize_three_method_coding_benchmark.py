#!/usr/bin/env python3
"""Build a claim-scoped audit for the narrowed three-method coding benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _static_row(result: dict[str, Any]) -> dict[str, Any]:
    quality = result["quality"]
    latency = result["latency"]
    physical = result["physical_reuse"]
    samples = int(result.get("samples") or len(result.get("rows", [])))
    dense_exact = int(quality["dense_exact_line"])
    reuse_exact = int(quality["reuse_exact_line"])
    cache_ready = float(
        latency.get(
            "cache_ready_speedup_vs_native_dense",
            latency.get("cache_ready_speedup"),
        )
    )
    if "build_amortized" in latency:
        n4_speedup = float(latency["build_amortized"]["4"]["speedup_vs_native_dense"])
    else:
        n4_speedup = float(latency["n4_including_build_speedup"])
    mean_reused = float(
        physical.get(
            "mean_reused_k_tokens",
            physical.get("mean_selected_tokens", 0.0),
        )
    )
    return {
        "method": result["method"],
        "engine": result.get("engine", "SGLang V40 bridge"),
        "samples": samples,
        "dense_exact_line": dense_exact,
        "reuse_exact_line": reuse_exact,
        "dense_exact_line_percent": _percent(dense_exact, samples),
        "reuse_exact_line_percent": _percent(reuse_exact, samples),
        "exact_line_delta_pp": _percent(reuse_exact - dense_exact, samples),
        "dense_code_sim_percent": float(quality["dense_code_sim_percent"]),
        "reuse_code_sim_percent": float(quality["reuse_code_sim_percent"]),
        "cache_ready_speedup_vs_native_dense": cache_ready,
        "n4_including_build_speedup_vs_native_dense": n4_speedup,
        "mean_reused_k_tokens": mean_reused,
    }


def _swebench_summary(
    dense: dict[str, Any],
    v40: dict[str, Any],
) -> dict[str, Any]:
    dense_official = dense["official"]
    v40_official = v40["official"]
    dense_runtime = dense["runtime"]
    v40_runtime = v40["runtime"]
    dense_resolved = set(dense_official["resolved_ids"])
    v40_resolved = set(v40_official["resolved_ids"])
    total = int(dense_official["total_instances"])
    if int(v40_official["total_instances"]) != total:
        raise ValueError("Dense and V40 SWE-bench totals do not match")
    dense_median = float(dense_runtime["median_ttft_ms"])
    v40_median = float(v40_runtime["median_ttft_ms"])
    dense_p95 = float(dense_runtime["p95_ttft_ms"])
    v40_p95 = float(v40_runtime["p95_ttft_ms"])
    return {
        "dataset": "SWE-bench Verified frozen mechanism cohort",
        "tasks": total,
        "dense": {
            "resolved": int(dense_official["resolved_instances"]),
            "accuracy_percent": _percent(
                int(dense_official["resolved_instances"]), total
            ),
            "empty_patches": int(dense_official["empty_patch_instances"]),
            "requests": int(dense_runtime["requests"]),
            "median_ttft_ms": dense_median,
            "p95_ttft_ms": dense_p95,
        },
        "v40": {
            "resolved": int(v40_official["resolved_instances"]),
            "accuracy_percent": _percent(
                int(v40_official["resolved_instances"]), total
            ),
            "accuracy_delta_pp_vs_dense": _percent(
                int(v40_official["resolved_instances"])
                - int(dense_official["resolved_instances"]),
                total,
            ),
            "empty_patches": int(v40_official["empty_patch_instances"]),
            "requests": int(v40_runtime["requests"]),
            "median_ttft_ms": v40_median,
            "p95_ttft_ms": v40_p95,
            "median_ttft_speedup_vs_dense": dense_median / v40_median,
            "p95_ttft_speedup_vs_dense": dense_p95 / v40_p95,
            "target_copy_events": int(v40_runtime["target_copy_events"]),
            "target_fallback_events": int(v40_runtime["target_fallback_events"]),
            "copied_tokens": int(v40_runtime["copied_tokens"]),
        },
        "paired_outcomes": {
            "dense_pass_v40_fail": sorted(dense_resolved - v40_resolved),
            "dense_fail_v40_pass": sorted(v40_resolved - dense_resolved),
            "both_pass": sorted(dense_resolved & v40_resolved),
            "both_fail": sorted(
                set(dense_official["submitted_ids"])
                - dense_resolved
                - v40_resolved
            ),
        },
    }


def build_audit(
    v40_static: dict[str, Any],
    cacheblend_static: dict[str, Any],
    kvcomm_static: dict[str, Any],
    dense_swe: dict[str, Any],
    v40_swe: dict[str, Any],
) -> dict[str, Any]:
    static_rows = [
        _static_row(v40_static),
        _static_row(cacheblend_static),
        _static_row(kvcomm_static),
    ]
    swe = _swebench_summary(dense_swe, v40_swe)
    v40_row = static_rows[0]
    competitors = static_rows[1:]
    return {
        "schema_version": 1,
        "benchmark_scope": {
            "methods": [
                "coding_grounded_observation_island_v40",
                "CacheBlend",
                "KVCOMM",
            ],
            "excluded": [
                "prefetch",
                "tail",
                "QCFuse",
                "FUSE-RAG",
                "ProphetKV",
            ],
        },
        "repobench_p_static_control": {
            "rows": static_rows,
            "claim_scope": (
                "Each method is compared only with Dense inside its native engine. "
                "Absolute TTFT and raw reused-token counts are not cross-engine ranks."
            ),
        },
        "swebench_verified_agent": {
            **swe,
            "claim_scope": (
                "Official task accuracy and TTFT compare V40 with Dense in the same "
                "SGLang/Qwen3 agent stack. Native KVCOMM and CacheBlend SWE-bench "
                "results were not run and are intentionally not imputed."
            ),
        },
        "decision": {
            "v40_beats_both_static_exact_line": all(
                v40_row["reuse_exact_line_percent"]
                > row["reuse_exact_line_percent"]
                for row in competitors
            ),
            "v40_beats_both_native_cache_ready_speedup": all(
                v40_row["cache_ready_speedup_vs_native_dense"]
                > row["cache_ready_speedup_vs_native_dense"]
                for row in competitors
            ),
            "v40_preserves_dense_swe_accuracy": (
                swe["v40"]["resolved"] >= swe["dense"]["resolved"]
            ),
            "current_claim": (
                "V40 demonstrates physical coding-aware KV reuse and a same-engine "
                "SWE-bench TTFT improvement, but it does not yet meet the accuracy "
                "or native speedup target against CacheBlend and KVCOMM."
            ),
        },
    }


def build_stability_diagnostic(
    root: Path,
    headline: dict[str, Any],
) -> dict[str, Any]:
    tasks = headline["paired_outcomes"]["dense_pass_v40_fail"]
    rows = []
    for task in tasks:
        dense_path = root / "dense" / f"canary_{task}" / "PIPELINE_STATUS.json"
        v40_path = (
            root
            / "coding_grounded_observation_island_v40"
            / f"canary_{task}"
            / "PIPELINE_STATUS.json"
        )
        dense = _read_json(dense_path)
        v40 = _read_json(v40_path)
        dense_pass = bool(dense["official"]["resolved_instances"])
        v40_pass = bool(v40["official"]["resolved_instances"])
        rows.append(
            {
                "task": task,
                "headline_dense_pass": True,
                "headline_v40_pass": False,
                "repeat_dense_pass": dense_pass,
                "repeat_v40_pass": v40_pass,
                "repeat_dense_empty_patch": bool(
                    dense["official"]["empty_patch_instances"]
                ),
                "repeat_v40_empty_patch": bool(
                    v40["official"]["empty_patch_instances"]
                ),
                "repeat_v40_copy_events": int(
                    v40["runtime"]["target_copy_events"]
                ),
                "stable_regression_reproduced": dense_pass and not v40_pass,
                "provenance": {
                    "dense": {
                        "path": str(dense_path.resolve()),
                        "sha256": _sha256(dense_path),
                    },
                    "v40": {
                        "path": str(v40_path.resolve()),
                        "sha256": _sha256(v40_path),
                    },
                },
            }
        )
    stable = sum(row["stable_regression_reproduced"] for row in rows)
    return {
        "status": "post_hoc_diagnostic_not_headline",
        "selected_after_observing_headline": True,
        "tasks": rows,
        "stable_regressions_reproduced": stable,
        "tasks_probed": len(rows),
        "conclusion": (
            "Neither headline Dense-pass/V40-fail outcome reproduced: both "
            "Dense repeats failed. The 6/12 versus 4/12 result remains a valid "
            "single-run point estimate, but its -16.7 pp difference is not a "
            "stable causal estimate of lossy-reuse damage."
        ),
    }


def render_markdown(audit: dict[str, Any], provenance: dict[str, Any]) -> str:
    static = audit["repobench_p_static_control"]["rows"]
    swe = audit["swebench_verified_agent"]
    lines = [
        "# Three-method coding KV-reuse audit",
        "",
        "## Outcome",
        "",
        audit["decision"]["current_claim"],
        "",
        "## RepoBench-P static control (50 frozen cases)",
        "",
        "| Method | Native engine | Exact Dense→reuse | CodeSim Dense→reuse | Cache-ready speedup | N=4 incl. build |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in static:
        lines.append(
            "| {method} | {engine} | {dense:.1f}%→{reuse:.1f}% ({delta:+.1f} pp) "
            "| {dense_sim:.2f}%→{reuse_sim:.2f}% | {speed:.3f}× | {n4:.3f}× |".format(
                method=row["method"],
                engine=row["engine"],
                dense=row["dense_exact_line_percent"],
                reuse=row["reuse_exact_line_percent"],
                delta=row["exact_line_delta_pp"],
                dense_sim=row["dense_code_sim_percent"],
                reuse_sim=row["reuse_code_sim_percent"],
                speed=row["cache_ready_speedup_vs_native_dense"],
                n4=row["n4_including_build_speedup_vs_native_dense"],
            )
        )
    lines += [
        "",
        audit["repobench_p_static_control"]["claim_scope"],
        "",
        "## SWE-bench Verified agent cohort (12 frozen tasks)",
        "",
        "| Arm | Resolved | Accuracy | Empty patches | Median TTFT | p95 TTFT |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Dense | {swe['dense']['resolved']}/12 | "
            f"{swe['dense']['accuracy_percent']:.1f}% | "
            f"{swe['dense']['empty_patches']} | "
            f"{swe['dense']['median_ttft_ms']:.1f} ms | "
            f"{swe['dense']['p95_ttft_ms']:.1f} ms |"
        ),
        (
            f"| V40 | {swe['v40']['resolved']}/12 | "
            f"{swe['v40']['accuracy_percent']:.1f}% "
            f"({swe['v40']['accuracy_delta_pp_vs_dense']:+.1f} pp) | "
            f"{swe['v40']['empty_patches']} | "
            f"{swe['v40']['median_ttft_ms']:.1f} ms "
            f"({swe['v40']['median_ttft_speedup_vs_dense']:.3f}×) | "
            f"{swe['v40']['p95_ttft_ms']:.1f} ms "
            f"({swe['v40']['p95_ttft_speedup_vs_dense']:.3f}×) |"
        ),
        "",
        (
            f"V40 made {swe['v40']['target_copy_events']} physical copy events, "
            f"copied {swe['v40']['copied_tokens']:,} tokens, and had "
            f"{swe['v40']['target_fallback_events']} fallbacks."
        ),
        "",
        "Dense-pass → V40-fail: "
        + ", ".join(swe["paired_outcomes"]["dense_pass_v40_fail"]),
        "",
        "Dense-fail → V40-pass: "
        + (
            ", ".join(swe["paired_outcomes"]["dense_fail_v40_pass"])
            or "none"
        ),
        "",
        swe["claim_scope"],
        "",
    ]
    if diagnostic := audit.get("stability_diagnostic"):
        lines += [
            "## Post-hoc stability diagnostic",
            "",
            (
                f"Stable Dense-pass→V40-fail outcomes reproduced: "
                f"{diagnostic['stable_regressions_reproduced']}/"
                f"{diagnostic['tasks_probed']}."
            ),
            "",
            "| Task | Headline Dense/V40 | Repeat Dense/V40 | V40 repeat copies |",
            "|---|---:|---:|---:|",
        ]
        for row in diagnostic["tasks"]:
            lines.append(
                f"| {row['task']} | pass/fail | "
                f"{'pass' if row['repeat_dense_pass'] else 'fail'}/"
                f"{'pass' if row['repeat_v40_pass'] else 'fail'} | "
                f"{row['repeat_v40_copy_events']} |"
            )
        lines += [
            "",
            diagnostic["conclusion"],
            "",
        ]
    lines += [
        "## Decision",
        "",
        "- Static exact-line target beaten: "
        + str(audit["decision"]["v40_beats_both_static_exact_line"]),
        "- Native cache-ready speedup target beaten: "
        + str(audit["decision"]["v40_beats_both_native_cache_ready_speedup"]),
        "- SWE-bench Dense accuracy preserved: "
        + str(audit["decision"]["v40_preserves_dense_swe_accuracy"]),
        "",
        "## Provenance",
        "",
    ]
    for label, item in provenance.items():
        lines.append(f"- {label}: `{item['path']}` (`sha256:{item['sha256']}`)")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v40-static", type=Path, required=True)
    parser.add_argument("--cacheblend-static", type=Path, required=True)
    parser.add_argument("--kvcomm-static", type=Path, required=True)
    parser.add_argument("--dense-swe", type=Path, required=True)
    parser.add_argument("--v40-swe", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "v40_static": args.v40_static,
        "cacheblend_static": args.cacheblend_static,
        "kvcomm_static": args.kvcomm_static,
        "dense_swe": args.dense_swe,
        "v40_swe": args.v40_swe,
    }
    provenance = {
        label: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for label, path in paths.items()
    }
    audit = build_audit(
        _read_json(args.v40_static),
        _read_json(args.cacheblend_static),
        _read_json(args.kvcomm_static),
        _read_json(args.dense_swe),
        _read_json(args.v40_swe),
    )
    if args.diagnostic_root is not None:
        audit["stability_diagnostic"] = build_stability_diagnostic(
            args.diagnostic_root,
            audit["swebench_verified_agent"],
        )
    audit["provenance"] = provenance
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        render_markdown(audit, provenance),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
