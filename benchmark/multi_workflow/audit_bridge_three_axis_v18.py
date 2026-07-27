#!/usr/bin/env python3
"""Audit task correctness, Dense preservation, fidelity, and KV efficiency.

This script deliberately keeps two protocols separate:

* free-running official agent trajectories measure final task correctness;
* frozen, identical-prompt replay measures Dense-reference distribution
  fidelity and matched request latency.

Combining those measurements without labeling their protocol would make a
reuse arm's stochastic task rescue look like evidence of intrinsic fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = ("dense", "general", "coding_version_graph_v17")
REUSE_ARMS = ARMS[1:]
DEFAULT_SEED = 20260727
DEFAULT_BOOTSTRAPS = 100_000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(probability * len(ordered)))
    return ordered[index]


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    seed: int = DEFAULT_SEED,
    iterations: int = DEFAULT_BOOTSTRAPS,
) -> list[float]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one delta")
    rng = random.Random(seed)
    samples = [
        statistics.mean(rng.choice(deltas) for _ in deltas)
        for _ in range(iterations)
    ]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, centre - half), min(1.0, centre + half)]


def exact_mcnemar_p(dense_only: int, reuse_only: int) -> float:
    """Two-sided exact binomial McNemar p-value."""

    discordant = dense_only + reuse_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(dense_only, reuse_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def load_official_arm(campaign: Path, arm: str) -> dict[str, Any]:
    path = campaign / arm / "full_18" / "PIPELINE_STATUS.json"
    value = read_json(path)
    if value.get("state") != "complete":
        raise ValueError(f"{arm}: pipeline is not complete")
    if value.get("arm") != arm:
        raise ValueError(f"{arm}: arm identity mismatch")
    if value.get("prefetch") is not False:
        raise ValueError(f"{arm}: prefetch must be false")
    official = value["official"]
    if official["total_instances"] != 18:
        raise ValueError(f"{arm}: expected 18 instances")
    if official["submitted_instances"] != 18:
        raise ValueError(f"{arm}: expected 18 submitted instances")
    if official["error_instances"] or official["incomplete_ids"]:
        raise ValueError(f"{arm}: official evaluation is incomplete")
    if arm == "dense":
        if value["runtime"]["target_copy_events"] != 0:
            raise ValueError("Dense unexpectedly copied KV")
    elif value["runtime"]["target_copy_events"] <= 0:
        raise ValueError(f"{arm}: physical KV copying was not proven")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "status": value,
    }


def transition_audit(
    dense_ids: set[str],
    reuse_ids: set[str],
    universe: list[str],
) -> dict[str, Any]:
    both_pass = sorted(dense_ids & reuse_ids)
    dense_only = sorted(dense_ids - reuse_ids)
    reuse_only = sorted(reuse_ids - dense_ids)
    both_fail = sorted(set(universe) - dense_ids - reuse_ids)
    dense_passes = len(dense_ids)
    dense_fails = len(universe) - dense_passes
    deltas = [
        100.0
        * (
            float(instance_id in reuse_ids)
            - float(instance_id in dense_ids)
        )
        for instance_id in universe
    ]
    return {
        "both_pass": len(both_pass),
        "both_pass_ids": both_pass,
        "both_fail": len(both_fail),
        "both_fail_ids": both_fail,
        "damage_dense_pass_to_reuse_fail": len(dense_only),
        "damage_ids": dense_only,
        "damage_rate_given_dense_pass": len(dense_only) / dense_passes,
        "damage_rate_wilson95": wilson_ci(len(dense_only), dense_passes),
        "rescue_dense_fail_to_reuse_pass": len(reuse_only),
        "rescue_ids": reuse_only,
        "rescue_rate_given_dense_fail": len(reuse_only) / dense_fails,
        "rescue_rate_wilson95": wilson_ci(len(reuse_only), dense_fails),
        "reuse_minus_dense_pp": statistics.mean(deltas),
        "reuse_minus_dense_pp_paired_bootstrap95": paired_bootstrap_ci(deltas),
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(
            len(dense_only), len(reuse_only)
        ),
    }


def ratio_delta(candidate: float, baseline: float) -> dict[str, float]:
    return {
        "candidate_over_baseline": candidate / baseline,
        "candidate_minus_baseline_percent": 100.0 * (candidate / baseline - 1.0),
        "baseline_over_candidate_speedup": baseline / candidate,
        "candidate_reduction_vs_baseline_percent": (
            100.0 * (1.0 - candidate / baseline)
        ),
    }


def load_replay(path: Path) -> dict[str, Any]:
    summary_path = path / "REPLAY_SUMMARY.json"
    value = read_json(summary_path)
    if value.get("prompt_hashes_identical_across_arms") is not True:
        raise ValueError(f"{path}: replay prompts are not identical")
    for arm in ARMS:
        if value["arm_summaries"][arm]["requests"] != 60:
            raise ValueError(f"{path}: {arm} does not contain 60 requests")
    return {
        "path": str(summary_path),
        "sha256": sha256(summary_path),
        "summary": value,
    }


def build_audit(campaign: Path, replay_paths: list[Path]) -> dict[str, Any]:
    official = {arm: load_official_arm(campaign, arm) for arm in ARMS}
    submitted = official["dense"]["status"]["official"]["submitted_ids"]
    if len(set(submitted)) != 18:
        raise ValueError("Dense submitted IDs are not a unique 18-item set")
    for arm in REUSE_ARMS:
        if official[arm]["status"]["official"]["submitted_ids"] != submitted:
            raise ValueError(f"{arm}: submitted IDs or ordering differ from Dense")

    resolved = {
        arm: set(official[arm]["status"]["official"]["resolved_ids"])
        for arm in ARMS
    }
    accuracy = {}
    for arm in ARMS:
        count = len(resolved[arm])
        accuracy[arm] = {
            "resolved": count,
            "total": 18,
            "accuracy": count / 18.0,
            "accuracy_wilson95": wilson_ci(count, 18),
            "resolved_ids": sorted(resolved[arm]),
            "empty_patch_instances": official[arm]["status"]["official"][
                "empty_patch_instances"
            ],
        }

    transitions = {
        arm: transition_audit(resolved["dense"], resolved[arm], submitted)
        for arm in REUSE_ARMS
    }

    natural_efficiency = {}
    dense_runtime = official["dense"]["status"]["runtime"]
    for arm in ARMS:
        runtime = official[arm]["status"]["runtime"]
        entry = {
            key: runtime[key]
            for key in (
                "requests",
                "median_ttft_ms",
                "p95_ttft_ms",
                "median_request_elapsed_ms",
                "target_copy_events",
                "target_fallback_events",
                "copied_tokens",
                "rotated_k_tokens",
            )
        }
        if arm != "dense":
            entry["median_ttft_vs_dense"] = ratio_delta(
                runtime["median_ttft_ms"], dense_runtime["median_ttft_ms"]
            )
        natural_efficiency[arm] = entry

    replays = [load_replay(path) for path in replay_paths]
    replay_runs = []
    for index, replay in enumerate(replays, start=1):
        value = replay["summary"]
        matched = value["paired_on_version_graph_target_keys"]
        arms = value["arm_summaries"]
        fidelity = value["dense_reference_fidelity"]
        cohort = value["coding_active_shortened_span_cohort"]["arms"]
        replay_runs.append(
            {
                "run": index,
                "source": {
                    "path": replay["path"],
                    "sha256": replay["sha256"],
                },
                "matched_target_requests": matched["dense"]["requests"],
                "matched_cache_ready_ttft_ms": {
                    arm: matched[arm]["median_ttft_ms"] for arm in ARMS
                },
                "v17_ttft_vs_general": ratio_delta(
                    matched["coding_version_graph_v17"]["median_ttft_ms"],
                    matched["general"]["median_ttft_ms"],
                ),
                "general_ttft_vs_dense": ratio_delta(
                    matched["general"]["median_ttft_ms"],
                    matched["dense"]["median_ttft_ms"],
                ),
                "v17_ttft_vs_dense": ratio_delta(
                    matched["coding_version_graph_v17"]["median_ttft_ms"],
                    matched["dense"]["median_ttft_ms"],
                ),
                "n4_including_build_ms": {
                    arm: arms[arm]["median_n4_including_build_ms"]
                    for arm in REUSE_ARMS
                },
                "v17_n4_vs_general": ratio_delta(
                    arms["coding_version_graph_v17"][
                        "median_n4_including_build_ms"
                    ],
                    arms["general"]["median_n4_including_build_ms"],
                ),
                "dense_reference_fidelity": fidelity,
                "full_v17_js_reduction_vs_general_percent": 100.0
                * (
                    1.0
                    - fidelity["coding_version_graph_v17"][
                        "mean_top20_plus_residual_js"
                    ]
                    / fidelity["general"]["mean_top20_plus_residual_js"]
                ),
                "coding_active_shortened_span_cohort": cohort,
                "cohort_v17_js_reduction_vs_general_percent": 100.0
                * (
                    1.0
                    - cohort["coding_version_graph_v17"][
                        "mean_top20_plus_residual_js"
                    ]
                    / cohort["general"]["mean_top20_plus_residual_js"]
                ),
            }
        )

    source_artifacts = {
        arm: {
            "path": official[arm]["path"],
            "sha256": official[arm]["sha256"],
        }
        for arm in ARMS
    }
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "post_registered_three_axis_audit",
        "protocol_separation": {
            "official_free_running": (
                "Final task correctness and task-level Dense damage/rescue. "
                "Agent prompts may diverge after the first differing token."
            ),
            "frozen_replay": (
                "Identical prompt hashes across arms; mechanism, matched TTFT, "
                "and Dense-reference first-token distribution fidelity only."
            ),
            "prohibited_inference": (
                "A free-running task rescue is not evidence that reuse is "
                "intrinsically more faithful than Dense."
            ),
        },
        "source_artifacts": source_artifacts,
        "task_correctness": accuracy,
        "dense_preservation": transitions,
        "free_running_efficiency": natural_efficiency,
        "frozen_identical_prompt_replays": replay_runs,
        "decision": {
            "candidate": "coding_version_graph_v17",
            "promoted": False,
            "reasons": [
                "Official task accuracy is 5/18, below both Dense and General at 6/18.",
                "V17 has two Dense-pass damages and only one Dense-fail rescue.",
                "On identical target prompts V17 cache-ready TTFT is about 20% slower than General in both runs.",
                "V17 lowers JS divergence versus General in both runs, including the coding-active cohort, but the fidelity gain does not convert into task accuracy.",
                "The preregistered >=0.95 first-token agreement gate fails in both replay runs.",
            ],
            "next_design_constraint": (
                "Preserve General's reuse span on ordinary requests and spend "
                "coding-aware protection only where an online signal predicts "
                "Dense damage with enough precision to repay its latency cost."
            ),
        },
    }
    return result


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def render_markdown(audit: dict[str, Any]) -> str:
    correctness = audit["task_correctness"]
    preservation = audit["dense_preservation"]
    efficiency = audit["free_running_efficiency"]
    lines = [
        "# V18C three-axis audit",
        "",
        "This audit separates final task correctness from same-prompt Dense "
        "fidelity. The official runs are free-running agent trajectories; the "
        "frozen replays use identical prompt hashes.",
        "",
        "## Official final task correctness",
        "",
        "| Arm | Resolved | Accuracy | Empty patches | Median TTFT |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = correctness[arm]
        runtime = efficiency[arm]
        lines.append(
            f"| {arm} | {row['resolved']}/18 | {percent(row['accuracy'])} | "
            f"{row['empty_patch_instances']} | {runtime['median_ttft_ms']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "## Task-level preservation relative to Dense",
            "",
            "| Arm | Both pass | Damage | Rescue | Both fail | Δ accuracy (pp) | Paired bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in REUSE_ARMS:
        row = preservation[arm]
        ci = row["reuse_minus_dense_pp_paired_bootstrap95"]
        lines.append(
            f"| {arm} | {row['both_pass']} | "
            f"{row['damage_dense_pass_to_reuse_fail']} | "
            f"{row['rescue_dense_fail_to_reuse_pass']} | "
            f"{row['both_fail']} | {row['reuse_minus_dense_pp']:+.1f} | "
            f"[{ci[0]:+.1f}, {ci[1]:+.1f}] |"
        )
    lines.extend(
        [
            "",
            "Damage means Dense passed but the reuse arm failed. Rescue means "
            "Dense failed but the reuse arm passed. With only 18 tasks, all "
            "intervals are wide and no superiority claim is supported.",
            "",
            "## Identical-prompt replay",
            "",
            "| Run | Dense TTFT | General TTFT | V17 TTFT | V17 vs General | General JS | V17 JS | V17 JS reduction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit["frozen_identical_prompt_replays"]:
        ttft = row["matched_cache_ready_ttft_ms"]
        fidelity = row["dense_reference_fidelity"]
        lines.append(
            f"| {row['run']} | {ttft['dense']:.1f} ms | "
            f"{ttft['general']:.1f} ms | "
            f"{ttft['coding_version_graph_v17']:.1f} ms | "
            f"{row['v17_ttft_vs_general']['candidate_minus_baseline_percent']:+.1f}% | "
            f"{fidelity['general']['mean_top20_plus_residual_js']:.6f} | "
            f"{fidelity['coding_version_graph_v17']['mean_top20_plus_residual_js']:.6f} | "
            f"{row['full_v17_js_reduction_vs_general_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "JS is computed over the top-20 token probabilities plus one "
            "aggregate residual bucket. It is not full-vocabulary KL. V17 is "
            "consistently closer to Dense by this distribution metric, but it "
            "is consistently slower than General.",
            "",
            "## Decision",
            "",
            "**Do not promote V17.** It improves distribution fidelity relative "
            "to General, but official accuracy falls from 6/18 to 5/18 and "
            "matched cache-ready TTFT is about 20% worse. The next candidate "
            "must retain General on ordinary requests and activate coding-aware "
            "protection only when an online risk signal has sufficient damage "
            "precision to justify the lost reuse.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--replay", action="append", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args.campaign, args.replay)
    json_output = args.json_output or args.campaign / "THREE_AXIS_AUDIT.json"
    markdown_output = (
        args.markdown_output or args.campaign / "THREE_AXIS_AUDIT.md"
    )
    write_json(json_output, audit)
    markdown_output.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps(audit["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
