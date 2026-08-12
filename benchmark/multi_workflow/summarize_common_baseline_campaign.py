#!/usr/bin/env python3
"""Build a compact accuracy/TTFT audit from the common baseline campaign."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = (
    ("cacheblend", "dense"),
    ("cacheblend", "reuse"),
    ("kvcomm", "dense"),
    ("kvcomm", "reuse"),
)
SGLANG_ARMS = (
    ("dense", "dense"),
    ("coding_dependency_graph_cold_lcb", "coding-aware"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def official_report(run_dir: Path) -> Path | None:
    reports = sorted((run_dir / "reports/enroot").glob("*.json"))
    return reports[-1] if reports else None


def accuracy_row(campaign: Path, scope: str, backend: str, mode: str) -> dict[str, Any] | None:
    run_dir = campaign / "runs" / scope / f"{backend}_{mode}" / "all"
    runtime_path = run_dir / "RUNTIME_SUMMARY.json"
    report_path = official_report(run_dir)
    if not runtime_path.is_file() or report_path is None:
        return None
    runtime = read_json(runtime_path)
    report = read_json(report_path)
    submitted = int(report["submitted_instances"])
    resolved = int(report["resolved_instances"])
    return {
        "backend": backend,
        "mode": mode,
        "resolved": resolved,
        "submitted": submitted,
        "accuracy": resolved / submitted if submitted else None,
        "agent_requests": runtime["requests"],
        "descriptive_agent_median_ttft_ms": runtime["median_ttft_ms"],
        "physical_reuse_requests": runtime["physical_reuse_requests"],
        "reused_k_tokens": runtime["reused_k_tokens"],
        "reused_v_tokens": runtime["reused_v_tokens"],
        "fallback_requests": runtime["fallback_requests"],
        "official_report": str(report_path),
    }


def sglang_accuracy_row(
    campaign: Path, scope: str, arm: str, mode: str
) -> dict[str, Any] | None:
    tasks = 4 if scope == "canary" else 24
    run_dir = (
        campaign
        / "runs"
        / f"sglang_{scope}"
        / arm
        / f"full_{tasks}"
    )
    runtime_path = run_dir / "RUNTIME_SUMMARY.json"
    report_path = official_report(run_dir)
    if not runtime_path.is_file() or report_path is None:
        return None
    runtime = read_json(runtime_path)
    report = read_json(report_path)
    submitted = int(report["submitted_instances"])
    resolved = int(report["resolved_instances"])
    copied = int(runtime.get("copied_tokens") or 0)
    return {
        "backend": "sglang",
        "mode": mode,
        "resolved": resolved,
        "submitted": submitted,
        "accuracy": resolved / submitted if submitted else None,
        "agent_requests": int(runtime["requests"]),
        "descriptive_agent_median_ttft_ms": runtime["median_ttft_ms"],
        "physical_reuse_requests": int(runtime.get("target_copy_events") or 0),
        "reused_k_tokens": copied,
        "reused_v_tokens": copied,
        "fallback_requests": int(runtime.get("target_fallback_events") or 0),
        "official_report": str(report_path),
    }


def exact_row(campaign: Path, label: str, backend: str) -> dict[str, Any] | None:
    path = campaign / "exact_prompt_replay" / label / backend / "RESULT.json"
    if not path.is_file():
        return None
    result = read_json(path)
    targets = result["targets"]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in targets]

    return {
        "backend": backend,
        "status": result["status"],
        "targets": len(targets),
        "rounds_per_arm_per_target": targets[0]["rounds_per_arm"] if targets else 0,
        "median_cache_ready_speedup": statistics.median(values("cache_ready_speedup")),
        "median_n1_including_build_speedup": statistics.median(
            values("n1_including_build_speedup")
        ),
        "median_n4_including_build_speedup": statistics.median(
            values("n4_including_build_speedup")
        ),
        "median_n16_including_build_speedup": statistics.median(
            values("n16_including_build_speedup")
        ),
        "targets_cache_ready_faster": sum(
            value > 1 for value in values("cache_ready_speedup")
        ),
        "physical_reuse_rounds": (
            sum(int(row["physical_reuse_rounds"]) for row in targets)
            if all("physical_reuse_rounds" in row for row in targets)
            else int((result.get("summary") or {}).get("physical_copy_events") or 0)
        ),
        "result": str(path),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Common-prompt native baseline audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "Accuracy uses official SWE-bench resolution. The agent-observed TTFT column is descriptive only because free-running trajectories may diverge after the first different output.",
        "",
    ]
    for scope, rows in summary["accuracy"].items():
        lines.extend(
            [
                f"## {scope} accuracy",
                "",
                "| Backend | Arm | Resolved | Accuracy | Agent requests | Descriptive median TTFT (ms) | Physical reuse requests |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {backend} | {mode} | {resolved}/{submitted} | {accuracy} | {agent_requests} | {ttft} | {physical} |".format(
                    backend=row["backend"],
                    mode=row["mode"],
                    resolved=row["resolved"],
                    submitted=row["submitted"],
                    accuracy=fmt(row["accuracy"]),
                    agent_requests=row["agent_requests"],
                    ttft=fmt(row["descriptive_agent_median_ttft_ms"], 1),
                    physical=row["physical_reuse_requests"],
                )
            )
        lines.append("")
    for label, rows in summary["exact_ttft"].items():
        lines.extend(
            [
                f"## {label} exact-token TTFT",
                "",
                "| Backend | Frozen prompts | Cache-ready | N=1 incl. build | N=4 incl. build | N=16 incl. build | Faster prompts | Physical rounds |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {backend} | {targets} | {ready}× | {n1}× | {n4}× | {n16}× | {faster}/{targets} | {physical} |".format(
                    backend=row["backend"],
                    targets=row["targets"],
                    ready=fmt(row["median_cache_ready_speedup"]),
                    n1=fmt(row["median_n1_including_build_speedup"]),
                    n4=fmt(row["median_n4_including_build_speedup"]),
                    n16=fmt(row["median_n16_including_build_speedup"]),
                    faster=row["targets_cache_ready_faster"],
                    physical=row["physical_reuse_rounds"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- Compare each reuse arm with its own native Dense arm; engine-to-engine absolute TTFT is not a causal speedup.",
            "- Cache-ready excludes source construction. N=1/4/16 amortizes the measured construction cost over that many target uses.",
            "- Accuracy is final task resolution, not token agreement, NLL, attention similarity, or K/V distance.",
            "- All timed requests use frozen identical token IDs and an ABBA schedule on the registered GPU class.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = (args.output or campaign / "summary").resolve()
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "accuracy": {},
        "exact_ttft": {},
    }
    for scope in ("canary", "formal"):
        rows = [
            row
            for backend, mode in ARMS
            if (row := accuracy_row(campaign, scope, backend, mode)) is not None
        ]
        rows.extend(
            row
            for arm, mode in SGLANG_ARMS
            if (row := sglang_accuracy_row(campaign, scope, arm, mode)) is not None
        )
        if rows:
            summary["accuracy"][scope] = rows
    for label in ("one_task_canary", "fresh24"):
        rows = [
            row
            for backend in ("cacheblend", "kvcomm")
            if (row := exact_row(campaign, label, backend)) is not None
        ]
        if rows:
            summary["exact_ttft"][label] = rows
    for label in ("canary4", "fresh24"):
        row = exact_row(campaign, label, "sglang_coding")
        if row is not None:
            summary["exact_ttft"].setdefault(label, []).append(row)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "SUMMARY.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write(output / "RESULTS.md", markdown(summary))
    print(output / "RESULTS.md")


if __name__ == "__main__":
    main()
