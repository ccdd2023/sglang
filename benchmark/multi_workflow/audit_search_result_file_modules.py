#!/usr/bin/env python3
"""Audit capacity for file-bounded modules inside excluded search results.

The current graph-cold runtime excludes repository search observations as a
whole.  This outcome-blind audit asks whether their already-visible output
contains contiguous, literal file sections that could become natural lossy-KV
islands while retaining the same version and dependency-hot guards.  It opens
no model, accuracy, TTFT, NLL, attention, or KV-deviation outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from benchmark.multi_workflow.bridge_reuse_litellm_model import (
    BridgeReuseLitellmModel,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    _tool_command,
    coding_dependency_relations,
    observed_path_target_guard,
    versioned_observed_path_candidates,
)
from benchmark.multi_workflow.context_bounded_litellm_model import (
    ContextBoundedLitellmModel,
)
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
RUNTIME = RuntimePaths.from_project(PROJECT)
CAMPAIGN = RUNTIME.artifacts / "impactkv_common_agent_baselines_fresh24_20260812"
TRAJECTORIES = (
    CAMPAIGN
    / "runs/sglang_formal/coding_dependency_graph_cold_lcb/full_24"
)
OUTPUT = RUNTIME.artifacts / "impactkv_search_file_module_audit_20260812"
MODEL = Path.home() / "models/Qwen2.5-Coder-7B-Instruct"
ROLLING_GROUPS = 6
MIN_TOKENS = 32
COPY_CAP = 4096
SEARCH = re.compile(r"(?:^|[;&|]\s*)(?:rg|grep|find)\b", re.I)
GREP_LINE = re.compile(
    r"^(?P<path>(?:/testbed/|\./)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\."
    r"(?:py|pyi|toml|yaml|yml|json|rst|md|cfg|ini|txt))(?::|-)(?P<line>\d+)(?::|-)"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_path(value: str) -> str:
    for prefix in ("/testbed/", "./"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def search_sections(group: list[dict[str, Any]], tokenizer: Tokenizer) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for message in group:
        if message.get("role") != "tool":
            continue
        current_path: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_path, current_lines
            if current_path is None or not current_lines:
                current_path = None
                current_lines = []
                return
            text = "\n".join(current_lines)
            tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
            sections.append(
                {
                    "path": current_path,
                    "lines": len(current_lines),
                    "tokens": tokens,
                    "eligible_size": MIN_TOKENS <= tokens <= COPY_CAP,
                }
            )
            current_path = None
            current_lines = []

        for line in str(message.get("content") or "").splitlines():
            match = GREP_LINE.match(line)
            path = normalize_path(match.group("path")) if match else None
            if path != current_path:
                flush()
            if path is not None:
                current_path = path
                current_lines.append(line)
        flush()
    return sections


def prompt_tokens_by_request(trajectory: dict[str, Any]) -> dict[int, int]:
    values = {}
    for message in trajectory.get("messages") or ():
        treatment = (message.get("extra") or {}).get("reuse_treatment") or {}
        if treatment.get("request_index") is not None:
            values[int(treatment["request_index"])] = int(
                treatment.get("prompt_tokens") or 0
            )
    return values


def audit() -> dict[str, Any]:
    tokenizer_path = MODEL / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    render = object.__new__(BridgeReuseLitellmModel)
    render._tokenizer = tokenizer
    counters: Counter[str] = Counter()
    task_sets: dict[str, set[str]] = {
        key: set()
        for key in (
            "excluded_search",
            "file_sections",
            "eligible_file_sections",
            "version_valid_next_target",
            "dependency_cold_next_target",
        )
    }
    section_lengths: list[int] = []
    opportunities: list[dict[str, Any]] = []
    for path in sorted(TRAJECTORIES.glob("*/*.traj.json")):
        trajectory = read_json(path)
        instance_id = str(trajectory["instance_id"])
        groups = ContextBoundedLitellmModel._turn_groups(
            trajectory["messages"][2:]
        )
        prompt_tokens = prompt_tokens_by_request(trajectory)
        # Request q has q-1 completed groups.  Source planning deliberately
        # removes the group that will roll out before q+1.
        for request_index in range(1, len(groups)):
            completed = groups[:request_index]
            selected = completed[-ROLLING_GROUPS:]
            retained = selected if len(selected) < ROLLING_GROUPS else selected[1:]
            broad, decision = versioned_observed_path_candidates(retained)
            for candidate, evidence in zip(
                broad, decision["candidate_evidence"], strict=True
            ):
                group = retained[int(evidence["group_index"])]
                commands = "\n".join(
                    value for message in group if (value := _tool_command(message))
                )
                if not SEARCH.search(commands):
                    continue
                counters["excluded_search_observations"] += 1
                task_sets["excluded_search"].add(instance_id)
                sections = search_sections(group, tokenizer)
                counters["literal_file_sections"] += len(sections)
                task_sets["file_sections"].update(
                    [instance_id] if sections else []
                )
                eligible_sections = [row for row in sections if row["eligible_size"]]
                counters["eligible_size_file_sections"] += len(eligible_sections)
                if eligible_sections:
                    task_sets["eligible_file_sections"].add(instance_id)
                    section_lengths.extend(int(row["tokens"]) for row in eligible_sections)

                next_selected = groups[: request_index + 1][-ROLLING_GROUPS:]
                pending = {
                    "source_group_sha256": evidence["group_sha256"],
                    "source_observation_sha256": evidence["observation_sha256"],
                    "source_paths": evidence["paths"],
                    "source_symbols": evidence["symbols"],
                    "repository_scope_dependency": evidence["path_provenance"][
                        "repository_scope_dependency"
                    ],
                }
                guard = observed_path_target_guard(pending, next_selected)
                if not guard["target_evidence_valid"]:
                    continue
                counters["version_valid_next_target"] += 1
                task_sets["version_valid_next_target"].add(instance_id)
                source_index = int(guard["source_group_index"])
                relations = coding_dependency_relations(
                    source_paths=set(evidence["paths"]),
                    source_symbols=set(evidence["symbols"]),
                    later_groups=next_selected[source_index + 1 :],
                )
                if relations:
                    continue
                counters["dependency_cold_next_target"] += 1
                task_sets["dependency_cold_next_target"].add(instance_id)
                if not eligible_sections:
                    continue
                literal = "".join(render._render_message_literal(message) for message in candidate)
                opportunities.append(
                    {
                        "instance_id": instance_id,
                        "source_request_index": request_index,
                        "target_request_index": request_index + 1,
                        "target_prompt_tokens": prompt_tokens.get(request_index + 1),
                        "whole_search_tokens": len(
                            tokenizer.encode(literal, add_special_tokens=False).ids
                        ),
                        "paths": evidence["paths"],
                        "eligible_file_sections": eligible_sections,
                        "repository_scope_dependency": evidence["path_provenance"][
                            "repository_scope_dependency"
                        ],
                    }
                )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "outcome-blind offline capacity audit",
        "counts": dict(counters),
        "tasks": {key: len(value) for key, value in task_sets.items()},
        "eligible_section_tokens": {
            "min": min(section_lengths) if section_lengths else None,
            "median": statistics.median(section_lengths) if section_lengths else None,
            "max": max(section_lengths) if section_lengths else None,
        },
        "next_request_cold_opportunities_with_sections": len(opportunities),
        "opportunities": opportunities,
        "interpretation": (
            "A positive count licenses implementation of exact file-section token "
            "localization under the existing version/dependency guards; it does not "
            "establish accuracy or speed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    tokenizer_path = MODEL / "tokenizer.json"
    registration = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_SEARCH_MODULE_CAPACITY_AUDIT",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Do currently excluded search outputs contain literal contiguous file "
            "sections that survive the next rolling request and remain dependency-cold?"
        ),
        "selection": (
            "all version-valid search observations in the frozen LCB Fresh24 "
            "trajectories; no task outcome or TTFT selection"
        ),
        "module_boundary": (
            "maximal contiguous lines sharing an online-visible grep path prefix; "
            "no fixed token slicing"
        ),
        "size_window_tokens": [MIN_TOKENS, COPY_CAP],
        "keep_gate": (
            "at least four next-request dependency-cold opportunities across at "
            "least two tasks, each with an eligible literal file section"
        ),
        "inputs": {
            "trajectory_root": str(TRAJECTORIES),
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": sha256(tokenizer_path),
        },
        "forbidden_outcomes": [
            "official resolved",
            "patch correctness",
            "TTFT",
            "NLL",
            "KV deviation",
            "attention",
        ],
    }
    registration_path = args.output / "REGISTRATION.json"
    if registration_path.is_file():
        prior = read_json(registration_path)
        for key in registration:
            if key != "registered_at_utc" and prior.get(key) != registration[key]:
                raise ValueError(f"registration mismatch: {key}")
    else:
        write_json(registration_path, registration)
    result = audit()
    write_json(args.output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
