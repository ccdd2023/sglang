#!/usr/bin/env python3
"""Preregister grounded, version-valid tool-observation reuse for V40."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from tokenizers import Tokenizer

from benchmark.multi_workflow import (
    motivate_v33_state_transition_target as v33a,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    critical_coding_event_reasons,
    is_successful_readonly_evidence,
    repository_paths,
)
from benchmark.multi_workflow.context_bounded_litellm_model import (
    ContextBoundedLitellmModel,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v40_grounded_observation_motivation_20260728"
)
AUDIT = (
    ARTIFACTS
    / "impactkv_v39_v38_equivalence_audit_20260728"
    / "V39_V38_EQUIVALENCE_AUDIT.json"
)
POLICY = Path(__file__).with_name("coding_reuse_policy.py")
TOKENIZER = Path(
    "/home/gfy/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/tokenizer.json"
)
ROLLING_GROUPS = 6
COPY_CAP = 4096
MIN_TOKENS = 128


def _render_message_literal(message: dict[str, Any]) -> str:
    message = copy.deepcopy(message)
    role = message["role"]
    if role == "assistant" and message.get("tool_calls"):
        value = "<|im_start|>assistant\n"
        if message.get("content"):
            value += str(message["content"]).strip() + "\n"
        for wrapped_call in message["tool_calls"]:
            call = wrapped_call.get("function", wrapped_call)
            value += f"<tool_call>\n<function={call['name']}>\n"
            arguments = call.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            for name, argument in arguments.items():
                value += f"<parameter={name}>{argument}</parameter>\n"
            value += "</function>\n</tool_call>\n"
        return value + "<|im_end|>\n"
    if role == "tool":
        return (
            "<|im_start|>user\n<tool_response>\n"
            + str(message.get("content") or "")
            + "\n</tool_response><|im_end|>\n"
        )
    return (
        f"<|im_start|>{role}\n"
        + str(message.get("content") or "")
        + "<|im_end|>\n"
    )


def _mutation_invalidates(
    source_group: Sequence[dict[str, Any]],
    later_groups: Sequence[Sequence[dict[str, Any]]],
) -> bool:
    source_paths = repository_paths(source_group)
    for group in later_groups:
        if (
            "repository_mutation_command"
            not in critical_coding_event_reasons(group)
        ):
            continue
        changed_paths = repository_paths(group)
        if not source_paths or not changed_paths:
            return True
        if not source_paths.isdisjoint(changed_paths):
            return True
    return False


def select_grounded_observation(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
    tokenizer: Tokenizer,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Select the largest successful read-only tool observation still valid."""

    candidates: list[dict[str, Any]] = []
    invalidated = 0
    read_only = 0
    for index, group in enumerate(retained_groups):
        if not is_successful_readonly_evidence(group):
            continue
        read_only += 1
        if _mutation_invalidates(group, retained_groups[index + 1 :]):
            invalidated += 1
            continue
        tool_messages = [
            message for message in group if message.get("role") == "tool"
        ]
        for message in tool_messages:
            token_count = len(
                tokenizer.encode(_render_message_literal(message)).ids
            )
            if token_count >= MIN_TOKENS:
                candidates.append(
                    {
                        "group_index": index,
                        "token_count": min(token_count, COPY_CAP),
                        "uncapped_token_count": token_count,
                        "repository_paths": sorted(repository_paths(group)),
                    }
                )
    selected = (
        max(
            candidates,
            key=lambda row: (
                row["token_count"],
                row["group_index"],
            ),
        )
        if candidates
        else None
    )
    return selected, {
        "read_only_observations": read_only,
        "version_invalidated_observations": invalidated,
        "eligible_observations": len(candidates),
    }


def register(output: Path) -> dict[str, Any]:
    path = output / "V40_MOTIVATION_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trajectories = v33a._trajectories()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V40_MOTIVATION_ANALYSIS",
        "motivation": (
            "V39 falsified request-level commit-phase Dense latching: V38 "
            "and General produced the same official outcomes and final "
            "patches on all six tasks. Coding-agent prompts contain a more "
            "specific distinction that General ignores: tool observations "
            "are externally grounded repository facts, while assistant "
            "reasoning and tool calls are context-sensitive decisions. Test "
            "whether copying only a still-version-valid, successful read-only "
            "tool observation retains enough online capacity to warrant an "
            "implementation."
        ),
        "candidate": {
            "name": "coding_grounded_observation_island_v40",
            "source": (
                "largest successful read-only tool observation among the five "
                "groups retained by the next rolling request"
            ),
            "excluded": [
                "all assistant reasoning and tool-call tokens",
                "execution, validation, failure, diff, and mutation groups",
                "observations invalidated by a later repository mutation",
            ],
            "one_contiguous_island": True,
            "copy_cap_tokens": COPY_CAP,
            "minimum_tokens": MIN_TOKENS,
            "target": "same exact observation literal in the next prompt",
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "analysis": {
            "same_21_frozen_dense_trajectories_as_v33": True,
            "only_completed_groups_before_each_candidate_request": True,
            "task_outcomes_read": False,
            "rolling_groups": ROLLING_GROUPS,
            "capacity_is_not_accuracy": True,
        },
        "frozen_gates": {
            "v32r_tasks_with_source_min": 2,
            "full18_tasks_with_source_min": 12,
            "requests_with_source_min": 80,
            "source_request_rate_min": 0.30,
            "median_selected_tokens_min": 256,
            "selected_vs_general_token_fraction_min": 0.20,
            "version_invalidated_observations_min": 1,
            "assistant_tokens_selected": 0,
        },
        "inputs": {
            "v39_equivalence_audit": str(AUDIT),
            "v39_equivalence_audit_sha256": sha256(AUDIT),
            "policy_sha256": sha256(POLICY),
            "tokenizer_sha256": sha256(TOKENIZER),
            "script_sha256": sha256(Path(__file__)),
            "trajectory_sha256": {
                str(path): sha256(path)
                for paths in trajectories.values()
                for path in paths
            },
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(path, value)
    return value


def _measure(path: Path, tokenizer: Tokenizer) -> dict[str, Any]:
    trajectory = read_json(path)
    calls = int(trajectory["info"]["model_stats"]["api_calls"])
    groups = ContextBoundedLitellmModel._turn_groups(
        trajectory["messages"][2:]
    )
    rows = []
    for completed_index in range(ROLLING_GROUPS, min(calls, len(groups) + 1)):
        rolling = groups[
            completed_index - ROLLING_GROUPS : completed_index
        ]
        retained = rolling[1:]
        selected, diagnostics = select_grounded_observation(
            retained, tokenizer
        )
        general_tokens = min(
            COPY_CAP,
            len(
                tokenizer.encode(
                    "".join(
                        _render_message_literal(message)
                        for group in retained
                        for message in group
                    )
                ).ids
            ),
        )
        rows.append(
            {
                "request_index": completed_index + 1,
                "selected": selected,
                "general_tokens": general_tokens,
                **diagnostics,
            }
        )
    selected_rows = [row for row in rows if row["selected"] is not None]
    return {
        "instance_id": trajectory["instance_id"],
        "calls": calls,
        "eligible_target_requests": len(rows),
        "requests_with_source": len(selected_rows),
        "selected_tokens": [
            int(row["selected"]["token_count"]) for row in selected_rows
        ],
        "general_tokens_on_selected_requests": [
            int(row["general_tokens"]) for row in selected_rows
        ],
        "version_invalidated_observations": sum(
            row["version_invalidated_observations"] for row in rows
        ),
        "read_only_observations_considered": sum(
            row["read_only_observations"] for row in rows
        ),
        "reached": bool(selected_rows),
    }


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    cohorts = {
        name: [_measure(path, tokenizer) for path in paths]
        for name, paths in v33a._trajectories().items()
    }
    rows = [row for cohort in cohorts.values() for row in cohort]
    eligible = sum(row["eligible_target_requests"] for row in rows)
    reached = sum(row["requests_with_source"] for row in rows)
    selected_tokens = [
        value for row in rows for value in row["selected_tokens"]
    ]
    general_tokens = [
        value
        for row in rows
        for value in row["general_tokens_on_selected_requests"]
    ]
    invalidated = sum(
        row["version_invalidated_observations"] for row in rows
    )
    aggregate = {
        "tasks": len(rows),
        "tasks_with_source": sum(row["reached"] for row in rows),
        "eligible_target_requests": eligible,
        "requests_with_source": reached,
        "source_request_rate": reached / eligible,
        "median_selected_tokens": statistics.median(selected_tokens),
        "selected_tokens": sum(selected_tokens),
        "general_tokens_on_same_requests": sum(general_tokens),
        "selected_vs_general_token_fraction": (
            sum(selected_tokens) / sum(general_tokens)
        ),
        "version_invalidated_observations": invalidated,
        "assistant_tokens_selected": 0,
    }
    frozen = registration["frozen_gates"]
    gates = {
        "v32r_tasks_with_source_min": sum(
            row["reached"] for row in cohorts["v32r"]
        )
        >= frozen["v32r_tasks_with_source_min"],
        "full18_tasks_with_source_min": sum(
            row["reached"] for row in cohorts["full18"]
        )
        >= frozen["full18_tasks_with_source_min"],
        "requests_with_source_min": reached
        >= frozen["requests_with_source_min"],
        "source_request_rate_min": aggregate["source_request_rate"]
        >= frozen["source_request_rate_min"],
        "median_selected_tokens_min": aggregate["median_selected_tokens"]
        >= frozen["median_selected_tokens_min"],
        "selected_vs_general_token_fraction_min": (
            aggregate["selected_vs_general_token_fraction"]
            >= frozen["selected_vs_general_token_fraction_min"]
        ),
        "version_invalidated_observations_min": invalidated
        >= frozen["version_invalidated_observations_min"],
        "assistant_tokens_selected": (
            aggregate["assistant_tokens_selected"]
            == frozen["assistant_tokens_selected"]
        ),
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V40_MOTIVATION"
            if all(gates.values())
            else "FAIL_V40_MOTIVATION"
        ),
        "registration_sha256": sha256(
            output / "V40_MOTIVATION_REGISTRATION.json"
        ),
        "cohorts": cohorts,
        "aggregate": aggregate,
        "gate_outcomes": gates,
        "decision": (
            "Implement V40 and run a preregistered paired mechanism canary."
            if all(gates.values())
            else "Reject V40 before GPU work."
        ),
        "warning": (
            "This outcome-independent capacity result does not establish "
            "functional accuracy or speed superiority."
        ),
    }
    write_json(output / "V40_MOTIVATION_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("register", "run"), nargs="?", default="run"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = (
        register(args.output)
        if args.command == "register"
        else run(args.output)
    )
    print(
        {
            "status": value["status"],
            "aggregate": value.get("aggregate"),
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
