"""Auditable coding-event policies for bridge KV reuse.

The original ``coding_aware`` arm protects the newest completed interaction on
every request.  The policies in this module keep General reuse as the default
and protect that interaction only when its command or observation carries a
concrete software-engineering risk signal.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence, TypeVar


T = TypeVar("T")

_RETURN_CODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.I)
_FAILURE_MARKERS = (
    "<exception>",
    "traceback (most recent call last)",
    "syntaxerror",
    "indentationerror",
    "assertionerror",
    "segmentation fault",
    "test failures",
    "tests failed",
    " failed,",
    " error:",
)
_MUTATION_MARKERS = (
    ".write_text(",
    ".write_bytes(",
    "apply_patch",
    "git apply",
    "git checkout ",
    "git restore ",
)
_SHELL_MUTATION = re.compile(
    r"(?:^|&&\s*|;\s*|\|\|\s*)(?:rm|mv|cp)\s+(?:-[^\s]+\s+)*",
    re.I,
)
_INPLACE_MUTATION = re.compile(
    r"\bsed\b[^\n;&|]*\s-i(?:\s|$)|\btee\b",
    re.I,
)
_OPEN_WRITE_MUTATION = re.compile(
    r"\bopen\(\s*[\"'][^\"']+[\"']\s*,\s*[\"'][wax+]",
    re.I,
)
_SEARCH_COMMAND = re.compile(
    r"(?:^|&&\s*|;\s*|\|\s*|\|\|\s*)"
    r"(?:grep|rg|find)\b",
    re.I,
)
_EXECUTION_OR_STATE_COMMAND = re.compile(
    r"\b(?:python\d*|pytest|tox|unittest|bash|sh)\b"
    r"|\bgit\s+(?:apply|diff|restore|checkout)\b"
    r"|\bmake\s+(?:test|check)\b",
    re.I,
)
_CONCRETE_PYTHON_READ = re.compile(
    r"(?:^|&&\s*|;\s*|\|\|\s*)"
    r"(?:cat|head|tail)\s+(?:-[^\s]+\s+)*[^\s|;]+\.py\b"
    r"|Path\(\s*[\"'][^\"']+\.py[\"']\s*\)\.read_text\(",
    re.I,
)
_READONLY_EVIDENCE_COMMAND = re.compile(
    r"(?:^|&&|;|\|\|?)\s*"
    r"(?:rg|grep|find|sed|cat|head|tail)\b",
    re.I,
)
_REPOSITORY_PATH = re.compile(
    r"(?:/testbed/|\./)?"
    r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+"
    r"\.(?:py|pyi|toml|yaml|yml|json|rst|md|cfg|ini))"
)
_PATCH_PATH = re.compile(
    r"(?m)^\*\*\* (?:Update|Add|Delete) File:\s*(\S+)"
    r"|^diff --git a/(\S+) b/(\S+)"
)


def _tool_command(message: dict[str, Any]) -> str:
    for wrapped in message.get("tool_calls") or ():
        call = wrapped.get("function", wrapped)
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return arguments
        if isinstance(arguments, dict) and arguments.get("command") is not None:
            return str(arguments["command"])
    return ""


def repository_paths(group: Sequence[dict[str, Any]]) -> set[str]:
    """Extract online-visible repository paths from a completed tool group."""

    command = "\n".join(
        value for message in group if (value := _tool_command(message))
    )
    paths = {
        match.group(1).lstrip("./")
        for match in _REPOSITORY_PATH.finditer(command)
    }
    for match in _PATCH_PATH.finditer(command):
        value = next((part for part in match.groups() if part), None)
        if value:
            paths.add(value.lstrip("./"))
    return paths


def _group_weight(group: Sequence[dict[str, Any]]) -> int:
    return sum(
        len(str(message.get("content") or ""))
        + len(json.dumps(message.get("tool_calls") or (), sort_keys=True))
        for message in group
    )


def select_version_graph_groups(
    selected_groups: Sequence[T],
    *,
    protect_latest: bool = True,
) -> tuple[list[T], dict[str, Any]]:
    """Select the largest contiguous file-version-valid reuse island.

    The first group will roll out of the next request.  Within the retained
    five, a later mutation invalidates earlier observations of the same file.
    An unlocalized mutation conservatively invalidates every earlier pathful
    group.  The newest risky event remains Dense.  No future event, oracle
    patch, or evaluator result is consulted.
    """

    retained = list(selected_groups[1:])
    paths = [repository_paths(group) for group in retained]
    risks = [latest_group_risk_reasons(group) for group in retained]
    mutations = [
        "repository_mutation_command" in reasons for reasons in risks
    ]
    stale: set[int] = set()
    for mutation_index, mutation in enumerate(mutations):
        if not mutation:
            continue
        changed = paths[mutation_index]
        for earlier in range(mutation_index):
            if paths[earlier] and (
                not changed or not paths[earlier].isdisjoint(changed)
            ):
                stale.add(earlier)
    protected_latest = bool(protect_latest and retained and risks[-1])
    eligible = [
        index
        for index in range(len(retained))
        if index not in stale
        and not (protected_latest and index == len(retained) - 1)
    ]
    runs: list[list[int]] = []
    for index in eligible:
        if not runs or runs[-1][-1] + 1 != index:
            runs.append([index])
        else:
            runs[-1].append(index)
    selected = (
        max(
            runs,
            key=lambda run: (
                sum(_group_weight(retained[index]) for index in run),
                run[-1],
            ),
        )
        if runs
        else []
    )
    return [retained[index] for index in selected], {
        "mode": "repository_version_graph_longest_island",
        "retained_groups_after_roll": len(retained),
        "repository_pathful_groups": sum(bool(value) for value in paths),
        "repository_mutation_groups": sum(mutations),
        "stale_group_indices": sorted(stale),
        "stale_groups": len(stale),
        "latest_group_protected": protected_latest,
        "latest_guard_enabled": protect_latest,
        "risk_reasons": risks[-1] if protected_latest else [],
        "eligible_islands": len(runs),
        "selected_group_indices": selected,
        "selected_groups": len(selected),
    }


def latest_group_risk_reasons(
    group: Sequence[dict[str, Any]],
) -> list[str]:
    """Return deterministic reasons for recomputing a completed coding turn."""

    reasons: list[str] = []
    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    command_lower = commands.lower()
    if any(marker in command_lower for marker in _MUTATION_MARKERS):
        reasons.append("repository_mutation_command")
    elif (
        _SHELL_MUTATION.search(commands)
        or _INPLACE_MUTATION.search(commands)
        or _OPEN_WRITE_MUTATION.search(commands)
    ):
        reasons.append("repository_mutation_command")

    observations = "\n".join(
        str(message.get("content") or "")
        for message in group
        if message.get("role") == "tool"
    )
    return_codes = [int(value) for value in _RETURN_CODE.findall(observations)]
    if any(value != 0 for value in return_codes):
        reasons.append("nonzero_tool_returncode")
    observation_lower = observations.lower()
    if any(marker in observation_lower for marker in _FAILURE_MARKERS):
        reasons.append("failure_diagnostic")
    if "diff --git " in observation_lower:
        reasons.append("repository_diff_observed")
    return reasons


def is_low_value_search_miss(
    group: Sequence[dict[str, Any]],
) -> bool:
    """Identify a failed read-only search, not an executable coding failure."""

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    if not _SEARCH_COMMAND.search(commands):
        return False
    command_lower = commands.lower()
    if (
        _EXECUTION_OR_STATE_COMMAND.search(commands)
        or any(marker in command_lower for marker in _MUTATION_MARKERS)
        or _SHELL_MUTATION.search(commands)
        or _INPLACE_MUTATION.search(commands)
    ):
        return False
    observations = "\n".join(
        str(message.get("content") or "")
        for message in group
        if message.get("role") == "tool"
    )
    return any(
        int(value) != 0 for value in _RETURN_CODE.findall(observations)
    )


def is_high_value_executable_failure(
    group: Sequence[dict[str, Any]],
) -> bool:
    """Identify a failed executable/reproducer worth retaining as memory."""

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    if not _EXECUTION_OR_STATE_COMMAND.search(commands):
        return False
    if is_low_value_search_miss(group):
        return False
    observations = "\n".join(
        str(message.get("content") or "")
        for message in group
        if message.get("role") == "tool"
    )
    return_codes = [int(value) for value in _RETURN_CODE.findall(observations)]
    return any(value != 0 for value in return_codes) or any(
        marker in observations.lower() for marker in _FAILURE_MARKERS
    )


def critical_coding_event_reasons(
    group: Sequence[dict[str, Any]],
) -> list[str]:
    """Return narrow online events that justify one Dense target.

    Broad nonzero-return and failure markers previously over-triggered on
    harmless search misses.  V31 abstains only after a repository mutation or
    observed diff, or after a real executable/test failure.
    """

    broad = latest_group_risk_reasons(group)
    reasons = [
        reason
        for reason in broad
        if reason
        in ("repository_mutation_command", "repository_diff_observed")
    ]
    if is_high_value_executable_failure(group):
        reasons.append("executable_failure")
    return list(dict.fromkeys(reasons))


def is_concrete_source_read(
    group: Sequence[dict[str, Any]],
) -> bool:
    """Identify a substantial successful read of a concrete Python source."""

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    if not _CONCRETE_PYTHON_READ.search(commands):
        return False
    if re.search(
        r"(?:^|/)(?:tests?|testing)(?:/|_)|(?:^|/)test_[^/]+\.py\b",
        commands,
        re.I,
    ):
        return False
    observations = "\n".join(
        str(message.get("content") or "")
        for message in group
        if message.get("role") == "tool"
    )
    return (
        "<returncode>0</returncode>" in observations
        and len(observations) >= 400
    )


def is_successful_readonly_evidence(
    group: Sequence[dict[str, Any]],
) -> bool:
    """Identify a substantial, successful read-only coding observation.

    This classifier is deliberately mechanical.  It is used only to widen a
    contiguous copy budget when the wider span can save meaningful prefill; it
    never drops history or treats the observation as an accuracy oracle.
    """

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    if not _READONLY_EVIDENCE_COMMAND.search(commands):
        return False
    command_lower = commands.lower()
    if (
        _EXECUTION_OR_STATE_COMMAND.search(commands)
        or any(marker in command_lower for marker in _MUTATION_MARKERS)
        or _SHELL_MUTATION.search(commands)
        or _INPLACE_MUTATION.search(commands)
    ):
        return False
    observations = "\n".join(
        str(message.get("content") or "")
        for message in group
        if message.get("role") == "tool"
    )
    return_codes = [int(value) for value in _RETURN_CODE.findall(observations)]
    return (
        bool(return_codes)
        and all(value == 0 for value in return_codes)
        and len(observations) >= 400
    )


def is_successful_executable_evidence(
    group: Sequence[dict[str, Any]],
) -> bool:
    """Identify a successful execution or focused validation observation."""

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    if not _EXECUTION_OR_STATE_COMMAND.search(commands):
        return False
    command_lower = commands.lower()
    if (
        any(marker in command_lower for marker in _MUTATION_MARKERS)
        or _SHELL_MUTATION.search(commands)
        or _INPLACE_MUTATION.search(commands)
        or _OPEN_WRITE_MUTATION.search(commands)
    ):
        return False
    observations = "\n".join(
        str(message.get("content") or "")
        for message in group
        if message.get("role") == "tool"
    )
    return_codes = [int(value) for value in _RETURN_CODE.findall(observations)]
    return bool(return_codes) and all(value == 0 for value in return_codes)


def coding_state_transition_target_reasons(
    groups: Sequence[Sequence[dict[str, Any]]],
) -> list[str]:
    """Return online-visible reasons to veto reuse on the current request.

    Raw transitions include critical state changes plus entry into successful
    read or execution phases.  A two-interaction cooldown then admits at most
    one target veto in any three consecutive completed interactions.
    """

    if not groups:
        return []
    def evidence_phase(
        group: Sequence[dict[str, Any]],
    ) -> str | None:
        if is_successful_readonly_evidence(group):
            return "readonly_evidence"
        if is_successful_executable_evidence(group):
            return "successful_execution"
        return None

    def raw_reasons(index: int) -> list[str]:
        critical = critical_coding_event_reasons(groups[index])
        if critical:
            return critical
        latest_phase = evidence_phase(groups[index])
        if latest_phase is None:
            return []
        previous_phase = (
            evidence_phase(groups[index - 1]) if index > 0 else None
        )
        if latest_phase == previous_phase:
            return []
        return [f"{latest_phase}_phase_transition"]

    latest = raw_reasons(len(groups) - 1)
    if not latest:
        return []
    cooldown_start = max(0, len(groups) - 3)
    if any(
        raw_reasons(index)
        for index in range(cooldown_start, len(groups) - 1)
    ):
        return []
    return latest


def select_failure_memory_groups(
    groups: Sequence[Sequence[dict[str, Any]]],
    *,
    recent_count: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Keep recent history plus the newest older executable failure."""

    if recent_count <= 0:
        raise ValueError("recent_count must be positive")
    recent_start = max(0, len(groups) - recent_count)
    recent = [list(group) for group in groups[recent_start:]]
    candidates = [
        (index, group)
        for index, group in enumerate(groups[:recent_start])
        if is_high_value_executable_failure(group)
    ]
    if not candidates:
        return recent, {
            "memory_anchor_present": False,
            "memory_anchor_source_index": None,
            "memory_anchor_risk_reasons": [],
        }
    anchor_index, anchor = candidates[-1]
    return [list(anchor), *recent], {
        "memory_anchor_present": True,
        "memory_anchor_source_index": anchor_index,
        "memory_anchor_risk_reasons": latest_group_risk_reasons(anchor),
    }


def select_reuse_groups(
    arm: str,
    selected_groups: Sequence[T],
    *,
    latest_group_messages: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[T], dict[str, Any]]:
    """Select a contiguous source block and emit an auditable decision."""

    retained = list(selected_groups[1:])
    decision: dict[str, Any] = {
        "arm": arm,
        "retained_groups_after_roll": len(retained),
        "latest_group_protected": False,
        "risk_reasons": [],
    }
    if arm == "coding_memory_v5":
        anchor_present = len(selected_groups) > 6
        # The selected prompt is [optional old failure anchor, recent six].
        # The next request always retains the current newest five recent
        # groups. Copy only that guaranteed contiguous overlap.
        overlap = list(
            selected_groups[2:] if anchor_present else selected_groups[1:]
        )
        decision.update(
            mode="failure_memory_plus_general_8k",
            retained_groups_after_roll=len(overlap),
            memory_anchor_present=anchor_present,
        )
        return overlap, decision
    if arm in ("general", "general_8k", "general_dual_4k"):
        decision["mode"] = (
            "general_contiguous_8k"
            if arm == "general_8k"
            else "general_dual_contiguous_4k"
            if arm == "general_dual_4k"
            else "general_contiguous"
        )
        return retained, decision
    if arm == "coding_critical_event_abstain_v31":
        reasons = critical_coding_event_reasons(
            latest_group_messages or ()
        )
        decision.update(
            mode=(
                "critical_event_dense_abstain"
                if reasons
                else "critical_event_general_reuse"
            ),
            critical_event=bool(reasons),
            critical_event_reasons=reasons,
        )
        return retained, decision
    if arm in ("coding_evidence_payoff_v7", "coding_dual_v8"):
        evidence = is_successful_readonly_evidence(
            latest_group_messages or ()
        )
        decision.update(
            mode=(
                "evidence_payoff_dual_island"
                if arm == "coding_dual_v8"
                else "evidence_payoff_contiguous"
            ),
            readonly_evidence=evidence,
            marginal_copy_threshold_tokens=1024,
            widened_copy_cap_tokens=6144,
        )
        return retained, decision
    if arm == "coding_version_graph_v17":
        eligible, graph = select_version_graph_groups(selected_groups)
        decision.update(graph)
        return eligible, decision
    if arm in (
        "coding_post_mutation_v19",
        "coding_post_mutation_dual_v20",
        "coding_post_mutation_seam32_v22",
        "coding_post_mutation_target_prefix_v23",
        "coding_post_mutation_payoff_guard_v28",
        "coding_post_mutation_payoff_guard_v29",
        "coding_critical_event_abstain_v31",
    ):
        eligible, graph = select_version_graph_groups(
            selected_groups,
            protect_latest=False,
        )
        graph["mode"] = (
            "post_mutation_dual_island"
            if arm
            in (
                "coding_post_mutation_dual_v20",
                "coding_post_mutation_seam32_v22",
                "coding_post_mutation_target_prefix_v23",
                "coding_post_mutation_payoff_guard_v28",
                "coding_post_mutation_payoff_guard_v29",
            )
            else "post_mutation_contiguous_island"
        )
        decision.update(graph)
        return eligible, decision
    if arm == "coding_source_guard_v6":
        read_index = next(
            (
                index
                for index in range(len(retained) - 1, -1, -1)
                if is_concrete_source_read(retained[index])
            ),
            None,
        )
        reset_by_mutation = (
            read_index is not None
            and any(
                "repository_mutation_command"
                in latest_group_risk_reasons(group)
                for group in retained[read_index + 1 :]
            )
        )
        if read_index is None or reset_by_mutation:
            decision.update(
                mode="source_analysis_guard",
                source_guard_active=False,
                source_guard_reset_by_mutation=reset_by_mutation,
                source_read_index=read_index,
            )
            return retained, decision
        decision.update(
            mode="source_analysis_guard",
            latest_group_protected=True,
            risk_reasons=["concrete_source_analysis"],
            source_guard_active=True,
            source_guard_reset_by_mutation=False,
            source_read_index=read_index,
            protected_groups=len(retained) - read_index,
        )
        return retained[:read_index], decision
    if arm == "coding_aware":
        decision.update(
            mode="always_protect_latest",
            latest_group_protected=True,
            risk_reasons=["fixed_latest_group_guard"],
        )
        return retained[:-1], decision
    if arm not in (
        "coding_failure_v1",
        "coding_phase_v1",
        "coding_adaptive_v2",
        "coding_adaptive_v3",
        "coding_budget_v4",
        "coding_source_guard_v6",
        "coding_evidence_payoff_v7",
        "coding_dual_v8",
        "coding_version_graph_v17",
        "coding_post_mutation_v19",
        "coding_post_mutation_dual_v20",
        "coding_post_mutation_seam32_v22",
        "coding_post_mutation_target_prefix_v23",
        "coding_post_mutation_payoff_guard_v28",
        "coding_post_mutation_payoff_guard_v29",
    ):
        raise ValueError(f"unsupported reuse policy arm: {arm}")

    reasons = latest_group_risk_reasons(latest_group_messages or ())
    ignored_reasons: list[str] = []
    if arm == "coding_failure_v1":
        reasons = [
            reason
            for reason in reasons
            if reason in ("nonzero_tool_returncode", "failure_diagnostic")
        ]
    elif arm in (
        "coding_adaptive_v3",
        "coding_budget_v4",
    ) and is_low_value_search_miss(latest_group_messages or ()):
        ignored_reasons = [
            reason
            for reason in reasons
            if reason in ("nonzero_tool_returncode", "failure_diagnostic")
        ]
        reasons = [
            reason for reason in reasons if reason not in ignored_reasons
        ]
    if arm == "coding_budget_v4":
        decision.update(
            mode="adaptive_search_aware_copy_budget",
            latest_group_protected=False,
            risk_reasons=reasons,
            ignored_risk_reasons=ignored_reasons,
            low_value_search_miss=bool(ignored_reasons),
            risk_budget_limited=bool(reasons),
        )
        return retained, decision

    protect = bool(reasons)
    decision.update(
        mode=(
            "adaptive_search_aware_latest_guard"
            if arm == "coding_adaptive_v3"
            else "adaptive_risk_gated_latest_guard"
            if arm == "coding_adaptive_v2"
            else "risk_gated_latest_guard"
        ),
        latest_group_protected=protect,
        risk_reasons=reasons,
        ignored_risk_reasons=ignored_reasons,
        low_value_search_miss=bool(ignored_reasons),
    )
    return retained[:-1] if protect else retained, decision


def effective_copy_cap(
    arm: str, base_cap: int, decision: dict[str, Any]
) -> int:
    """Use a wider safe-phase block without widening risky hybrid states."""

    if arm == "general_8k":
        return 2 * base_cap
    if arm == "coding_memory_v5":
        return 2 * base_cap
    if arm in ("coding_evidence_payoff_v7", "coding_dual_v8"):
        candidate_tokens = int(decision.get("candidate_tokens", 0))
        threshold = base_cap + int(
            decision.get("marginal_copy_threshold_tokens", 1024)
        )
        if decision.get("readonly_evidence", False) and (
            candidate_tokens >= threshold
        ):
            return int(decision.get("widened_copy_cap_tokens", 6144))
        return base_cap
    if arm == "coding_budget_v4":
        return (
            base_cap
            if decision.get("risk_budget_limited", False)
            else 2 * base_cap
        )
    if (
        arm in ("coding_adaptive_v2", "coding_adaptive_v3")
        and not decision.get("latest_group_protected", False)
    ):
        return 2 * base_cap
    return base_cap


def post_mutation_payoff_guard(
    *,
    request_index: int,
    coding_candidate_tokens: int,
    general_candidate_tokens: int,
    copy_cap: int,
    step_limit: int = 20,
    minimum_future_targets: int = 4,
    payoff_ratio_threshold: float = 0.60,
    exact_prefix_credit_tokens: int = 640,
) -> dict[str, Any]:
    """Choose coding protection only when it can repay its lost middle span."""

    if request_index <= 0 or step_limit <= 0:
        raise ValueError("request_index and step_limit must be positive")
    if min(
        coding_candidate_tokens,
        general_candidate_tokens,
        copy_cap,
        minimum_future_targets,
        exact_prefix_credit_tokens,
    ) < 0:
        raise ValueError("payoff guard token/count values must be non-negative")
    if payoff_ratio_threshold <= 0:
        raise ValueError("payoff_ratio_threshold must be positive")
    coding_capped = min(coding_candidate_tokens, copy_cap)
    general_capped = min(general_candidate_tokens, copy_cap)
    payoff_ratio = (
        (coding_capped + exact_prefix_credit_tokens) / general_capped
        if general_capped
        else 0.0
    )
    future_target_upper_bound = step_limit - request_index
    if future_target_upper_bound < minimum_future_targets:
        mode = "payoff_guard_dense_abstain_late_branch"
    elif payoff_ratio < payoff_ratio_threshold:
        mode = "payoff_guard_general_middle_exact_prefix"
    else:
        mode = "payoff_guard_post_mutation_protected"
    return {
        "mode": mode,
        "step_limit": step_limit,
        "minimum_future_target_upper_bound": minimum_future_targets,
        "future_target_upper_bound": future_target_upper_bound,
        "payoff_ratio_threshold": payoff_ratio_threshold,
        "exact_prefix_credit_tokens": exact_prefix_credit_tokens,
        "coding_candidate_tokens": coding_candidate_tokens,
        "general_candidate_tokens": general_candidate_tokens,
        "coding_candidate_capped_tokens": coding_capped,
        "general_candidate_capped_tokens": general_capped,
        "payoff_ratio": payoff_ratio,
    }
