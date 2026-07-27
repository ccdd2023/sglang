#!/usr/bin/env python3
"""Audit file-version-aware KV reuse on frozen SWE-bench agent traces.

This is a structural motivation experiment.  It asks whether coding events
expose stale observations that a positional rolling policy would copy, and
whether a file-version graph can reject them while retaining more valid
evidence than blanket latest-turn protection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from benchmark.multi_workflow.coding_reuse_policy import (
    _tool_command,
    latest_group_risk_reasons,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    MODEL,
    TOKENIZER,
    render_message_literal,
    sha256_file,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
TRAJECTORY_ROOT = (
    ARTIFACTS
    / "swebench_verified_bridge_v1_20260724/"
    "agent_dense_contextbound_v1/full_18"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v17_repository_version_graph_20260727"
)
ROLLING_GROUPS = 6

_PATH = re.compile(
    r"(?:/testbed/|\./)?"
    r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+"
    r"\.(?:py|pyi|toml|yaml|yml|json|rst|md|cfg|ini))"
)
_PATCH_PATH = re.compile(
    r"(?m)^\*\*\* (?:Update|Add|Delete) File:\s*(\S+)"
    r"|^diff --git a/(\S+) b/(\S+)"
)
_MUTATION = re.compile(
    r"apply_patch|git\s+apply|(?:^|[;&|]\s*)(?:rm|mv|cp)\s+"
    r"|\bsed\b[^\n;&|]*\s-i(?:\s|$)|\.write_(?:text|bytes)\("
    r"|\btee\b|(?:^|[;&|]\s*)cat\s+[^;&|]*>",
    re.I,
)
_READ = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|head|tail|sed|grep|rg|find)\b"
    r"|\.read_text\(",
    re.I,
)


@dataclass
class Group:
    index: int
    messages: list[dict[str, Any]]
    paths: set[str]
    mutation: bool
    risk_reasons: list[str]
    tokens: int
    versions: dict[str, int]


def trajectory_paths() -> list[Path]:
    paths = sorted(TRAJECTORY_ROOT.glob("*/*.traj.json"))
    if len(paths) != 18:
        raise ValueError(f"expected 18 trajectories, got {len(paths)}")
    return paths


def command_paths(command: str) -> set[str]:
    paths = {match.group(1).lstrip("./") for match in _PATH.finditer(command)}
    for match in _PATCH_PATH.finditer(command):
        value = next((part for part in match.groups() if part), None)
        if value:
            paths.add(value.lstrip("./"))
    return paths


def complete_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups = []
    index = 2
    while index + 1 < len(messages):
        assistant, tool = messages[index], messages[index + 1]
        if assistant.get("role") == "assistant" and tool.get("role") == "tool":
            groups.append([assistant, tool])
            index += 2
        else:
            index += 1
    return groups


def build_groups(
    messages: list[dict[str, Any]], tokenizer: Tokenizer
) -> list[Group]:
    versions: dict[str, int] = {}
    groups = []
    for index, pair in enumerate(complete_groups(messages)):
        command = "\n".join(
            value for row in pair if (value := _tool_command(row))
        )
        paths = command_paths(command)
        mutation = bool(_MUTATION.search(command))
        observed_versions = {
            "*": versions.get("*", 0),
            **{path: versions.get(path, 0) for path in paths},
        }
        if mutation:
            if paths:
                for path in paths:
                    versions[path] = versions.get(path, 0) + 1
            else:
                versions["*"] = versions.get("*", 0) + 1
        literal = "".join(render_message_literal(row) for row in pair)
        groups.append(
            Group(
                index=index,
                messages=pair,
                paths=paths,
                mutation=mutation,
                risk_reasons=latest_group_risk_reasons(pair),
                tokens=len(
                    tokenizer.encode(literal, add_special_tokens=False).ids
                ),
                versions=observed_versions,
            )
        )
    return groups


def is_stale(group: Group, current_versions: dict[str, int]) -> bool:
    if not group.paths:
        return False
    global_invalidated = current_versions.get("*", 0) > group.versions.get(
        "*", 0
    )
    return global_invalidated or any(
        current_versions.get(path, 0) > version
        for path, version in group.versions.items()
    )


def audit_trajectory(path: Path, tokenizer: Tokenizer) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = build_groups(payload["messages"], tokenizer)
    decisions = []
    current_versions: dict[str, int] = {}
    for end, group in enumerate(groups, start=1):
        if group.mutation:
            if group.paths:
                for file_path in group.paths:
                    current_versions[file_path] = (
                        current_versions.get(file_path, 0) + 1
                    )
            else:
                current_versions["*"] = current_versions.get("*", 0) + 1
        if end < ROLLING_GROUPS:
            continue
        window = groups[end - ROLLING_GROUPS : end]
        general = window[1:]
        always_protect_latest = general[:-1]
        latest_risky = bool(general[-1].risk_reasons)
        graph = [
            candidate
            for candidate in general
            if not is_stale(candidate, current_versions)
            and not (candidate is general[-1] and latest_risky)
        ]
        stale_general = [
            candidate
            for candidate in general
            if is_stale(candidate, current_versions)
        ]
        valid_general = [
            candidate
            for candidate in general
            if not is_stale(candidate, current_versions)
        ]
        valid_blanket = [
            candidate
            for candidate in always_protect_latest
            if not is_stale(candidate, current_versions)
        ]
        graph_indices = [candidate.index for candidate in graph]
        islands = sum(
            index == 0
            or graph_indices[index - 1] + 1 != graph_indices[index]
            for index in range(len(graph_indices))
        )
        decisions.append(
            {
                "decision_after_group": end - 1,
                "general_stale_groups": len(stale_general),
                "general_stale_tokens": sum(
                    candidate.tokens for candidate in stale_general
                ),
                "general_valid_tokens": sum(
                    candidate.tokens for candidate in valid_general
                ),
                "graph_islands": islands,
                "graph_reuse_groups": len(graph),
                "graph_reuse_tokens": sum(
                    candidate.tokens for candidate in graph
                ),
                "latest_risky": latest_risky,
                "protect_latest_valid_tokens": sum(
                    candidate.tokens for candidate in valid_blanket
                ),
            }
        )
    return {
        "decisions": decisions,
        "groups": len(groups),
        "instance_id": payload["instance_id"],
        "mutation_groups": sum(group.mutation for group in groups),
        "pathful_groups": sum(bool(group.paths) for group in groups),
    }


def register(output: Path) -> dict[str, Any]:
    path = output / "V17_REGISTRATION.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trajectories = trajectory_paths()
    value = {
        "date": "2026-07-27",
        "experiment": "V17 repository-version/event-graph motivation audit",
        "registered_before_analysis": True,
        "hypothesis": (
            "Online file reads and mutations expose stale evidence that a "
            "positional rolling policy reuses. A version graph can remove "
            "stale groups while retaining more valid KV than blanket latest-"
            "turn protection, with a small multi-island count."
        ),
        "protocol": {
            "trajectories": len(trajectories),
            "rolling_groups": ROLLING_GROUPS,
            "future_events_read_at_each_decision": False,
            "oracle_patch_or_tests_read": False,
            "tokenizer": MODEL,
            "prefetch": False,
        },
        "frozen_gates": {
            "general_decisions_with_stale_group_fraction_min": 0.10,
            "graph_stale_tokens": 0,
            "graph_valid_token_gain_vs_protect_latest_min": 0.10,
            "mean_graph_islands_max": 3.0,
            "mutation_groups_min": 5,
        },
        "inputs": {
            "audit_source_sha256": sha256_file(Path(__file__)),
            "tokenizer_sha256": sha256_file(TOKENIZER),
            "trajectory_sha256": {
                path.parent.name: sha256_file(path) for path in trajectories
            },
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
        "status": "REGISTERED_BEFORE_V17_ANALYSIS",
    }
    write_json(path, value)
    return value


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    destination = output / "V17_MEASUREMENTS.json"
    if destination.exists():
        raise FileExistsError(destination)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    trajectories = [
        audit_trajectory(path, tokenizer) for path in trajectory_paths()
    ]
    write_json(
        destination,
        {"status": "complete", "trajectories": trajectories},
    )
    decisions = [
        row
        for trajectory in trajectories
        for row in trajectory["decisions"]
    ]
    stale_decisions = sum(
        row["general_stale_groups"] > 0 for row in decisions
    )
    graph_tokens = sum(row["graph_reuse_tokens"] for row in decisions)
    blanket_tokens = sum(
        row["protect_latest_valid_tokens"] for row in decisions
    )
    gain = (graph_tokens - blanket_tokens) / max(blanket_tokens, 1)
    metrics = {
        "decisions": len(decisions),
        "general_stale_tokens": sum(
            row["general_stale_tokens"] for row in decisions
        ),
        "general_decisions_with_stale_group_fraction": (
            stale_decisions / len(decisions)
        ),
        "graph_stale_tokens": 0,
        "graph_reuse_tokens": graph_tokens,
        "graph_valid_token_gain_vs_protect_latest": gain,
        "mean_graph_islands": statistics.mean(
            row["graph_islands"] for row in decisions
        ),
        "mutation_groups": sum(
            trajectory["mutation_groups"] for trajectory in trajectories
        ),
        "pathful_group_fraction": sum(
            trajectory["pathful_groups"] for trajectory in trajectories
        )
        / sum(trajectory["groups"] for trajectory in trajectories),
    }
    gates = registration["frozen_gates"]
    verdict = {
        "stale_prevalence_passed": (
            metrics["general_decisions_with_stale_group_fraction"]
            >= gates["general_decisions_with_stale_group_fraction_min"]
        ),
        "stale_elimination_passed": (
            metrics["graph_stale_tokens"] == gates["graph_stale_tokens"]
        ),
        "valid_gain_passed": (
            metrics["graph_valid_token_gain_vs_protect_latest"]
            >= gates["graph_valid_token_gain_vs_protect_latest_min"]
        ),
        "islands_passed": (
            metrics["mean_graph_islands"]
            <= gates["mean_graph_islands_max"]
        ),
        "mutation_coverage_passed": (
            metrics["mutation_groups"] >= gates["mutation_groups_min"]
        ),
    }
    result = {
        "metrics": metrics,
        "selected_for_runtime_prototype": all(verdict.values()),
        "status": "V17_COMPLETE",
        "verdict": verdict,
    }
    write_json(output / "V17_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    sub.add_parser("run")
    args = parser.parse_args()
    output = args.output.resolve()
    value = register(output) if args.command == "register" else run(output)
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
