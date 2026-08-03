"""Auditable coding-event policies for bridge KV reuse.

The original ``coding_aware`` arm protects the newest completed interaction on
every request.  The policies in this module keep General reuse as the default
and protect that interaction only when its command or observation carries a
concrete software-engineering risk signal.
"""

from __future__ import annotations

import hashlib
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
_SHELL_SOURCE_WRITE = re.compile(
    r"\b(?:cat|printf|echo)\b[^\n]*(?:>>|>)\s*"
    r"(?:/testbed/|\./)?[^\s;&|]+"
    r"\.(?:py|pyi|toml|yaml|yml|json|cfg|ini)\b",
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
_FOCUSED_VALIDATION_COMMAND = re.compile(
    r"\b(?:pytest|tox|unittest)\b"
    r"|\bmake\s+(?:test|check)\b"
    r"|\bpython\d*\b",
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
_PYTHON_DECLARATION = re.compile(
    r"(?m)(?:^|<output>)\s*"
    r"(?:async\s+def|def|class)\s+([A-Za-z_]\w*)\b"
)
_PATCH_HUNK_SYMBOL = re.compile(
    r"(?m)^@@[^\n]*\b(?:async\s+def|def|class)\s+([A-Za-z_]\w*)\b"
)
_READ_QUERY_SYMBOL = re.compile(
    r"(?:^|&&\s*|;\s*|\|\|\s*)"
    r"(?:rg|grep)\b(?:\s+--?[^\s]+)*\s+"
    r"[\"']?([A-Za-z_]\w*)[\"']?",
    re.I,
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


def tool_observation_sha256(group: Sequence[dict[str, Any]]) -> str:
    """Return a stable identity for the tool-only evidence in one turn."""

    tool_messages = [
        {
            "role": str(message.get("role") or ""),
            "content": str(message.get("content") or ""),
        }
        for message in group
        if message.get("role") == "tool"
    ]
    payload = json.dumps(
        tool_messages,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _declared_symbols(value: str) -> set[str]:
    return {
        *(_PYTHON_DECLARATION.findall(value)),
        *(_PATCH_HUNK_SYMBOL.findall(value)),
    }


def repository_observation_symbols(
    group: Sequence[dict[str, Any]],
) -> set[str]:
    """Extract answer-blind symbol provenance from a read observation.

    Full or partial Python reads contribute visible ``def``/``class`` names.
    Focused ``rg``/``grep`` commands contribute their identifier query.  The
    extractor deliberately does not infer call graphs or consult repository
    state that was not already visible to the agent.
    """

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    observations = "\n".join(
        str(message.get("content") or "")
        for message in group
        if message.get("role") == "tool"
    )
    symbols = _declared_symbols(observations)
    symbols.update(_READ_QUERY_SYMBOL.findall(commands))
    return symbols


def repository_mutation_symbols(
    group: Sequence[dict[str, Any]],
) -> set[str]:
    """Extract only explicit symbol names from an online-visible mutation."""

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    return _declared_symbols(commands)


def _mutation_effect_on_evidence(
    *,
    source_paths: set[str],
    source_symbols: set[str],
    mutation: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Classify one later mutation against an earlier observation.

    A symbol-disjoint exception is allowed only for one localized source file
    and explicit symbols on both sides.  Every ambiguity preserves V40's
    file-level fail-closed behavior.
    """

    changed_paths = repository_paths(mutation)
    changed_symbols = repository_mutation_symbols(mutation)
    if not source_paths or not changed_paths:
        return {
            "invalidates": True,
            "reason": "unlocalized_repository_mutation",
            "changed_paths": sorted(changed_paths),
            "changed_symbols": sorted(changed_symbols),
        }
    overlap = source_paths & changed_paths
    if not overlap:
        return {
            "invalidates": False,
            "reason": "path_disjoint_mutation",
            "changed_paths": sorted(changed_paths),
            "changed_symbols": sorted(changed_symbols),
        }
    if (
        len(source_paths) == 1
        and len(overlap) == 1
        and source_symbols
        and changed_symbols
        and source_symbols.isdisjoint(changed_symbols)
    ):
        return {
            "invalidates": False,
            "reason": "same_file_symbol_disjoint_mutation",
            "changed_paths": sorted(changed_paths),
            "changed_symbols": sorted(changed_symbols),
        }
    return {
        "invalidates": True,
        "reason": (
            "same_file_symbol_overlap"
            if source_symbols & changed_symbols
            else "same_file_symbol_ambiguous"
        ),
        "changed_paths": sorted(changed_paths),
        "changed_symbols": sorted(changed_symbols),
    }


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


def grounded_observation_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Return version-valid, successful read-only tool observations.

    V40 deliberately excludes assistant reasoning and tool-call tokens.  A
    read observation is also excluded when a later retained group mutates the
    same repository path, or when either side of that mutation cannot be
    localized safely.
    """

    candidates: list[list[dict[str, Any]]] = []
    candidate_group_indices: list[int] = []
    invalidated = 0
    read_only = 0
    for index, group in enumerate(retained_groups):
        if not is_successful_readonly_evidence(group):
            continue
        read_only += 1
        source_paths = repository_paths(group)
        invalid = False
        for later in retained_groups[index + 1 :]:
            if (
                "repository_mutation_command"
                not in critical_coding_event_reasons(later)
            ):
                continue
            changed_paths = repository_paths(later)
            if (
                not source_paths
                or not changed_paths
                or not source_paths.isdisjoint(changed_paths)
            ):
                invalid = True
                break
        if invalid:
            invalidated += 1
            continue
        tool_messages = [
            message for message in group if message.get("role") == "tool"
        ]
        if tool_messages:
            candidates.append(tool_messages)
            candidate_group_indices.append(index)
    return candidates, {
        "mode": "grounded_version_valid_observation_island",
        "retained_groups_after_roll": len(retained_groups),
        "read_only_observations": read_only,
        "version_invalidated_observations": invalidated,
        "eligible_observations": len(candidates),
        "candidate_group_indices": candidate_group_indices,
        "assistant_tokens_selected": 0,
        "latest_group_protected": False,
        "risk_reasons": [],
    }


def versioned_symbol_observation_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Return V45 read observations with explicit versioned provenance.

    V45 retains V40's file-level fail-closed rule, except when both the read
    and a later same-file mutation name explicit, disjoint Python symbols.
    Candidate evidence is emitted in the same order as ``candidates`` so the
    bridge can bind a cached token span to its file-version evidence.
    """

    candidates: list[list[dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    read_only = 0
    invalidated = 0
    invalidation_reasons: dict[str, int] = {}
    symbol_disjoint_preservations = 0
    for index, group in enumerate(retained_groups):
        if not is_successful_readonly_evidence(group):
            continue
        read_only += 1
        source_paths = repository_paths(group)
        source_symbols = repository_observation_symbols(group)
        invalidating_effect: dict[str, Any] | None = None
        candidate_symbol_disjoint = 0
        later_mutations = 0
        for later in retained_groups[index + 1 :]:
            if (
                "repository_mutation_command"
                not in critical_coding_event_reasons(later)
            ):
                continue
            later_mutations += 1
            effect = _mutation_effect_on_evidence(
                source_paths=source_paths,
                source_symbols=source_symbols,
                mutation=later,
            )
            if effect["invalidates"]:
                invalidating_effect = effect
                break
            if effect["reason"] == "same_file_symbol_disjoint_mutation":
                candidate_symbol_disjoint += 1
        if invalidating_effect is not None:
            invalidated += 1
            reason = str(invalidating_effect["reason"])
            invalidation_reasons[reason] = (
                invalidation_reasons.get(reason, 0) + 1
            )
            continue
        tool_messages = [
            message for message in group if message.get("role") == "tool"
        ]
        if not tool_messages:
            continue
        candidates.append(tool_messages)
        evidence.append(
            {
                "group_index": index,
                "paths": sorted(source_paths),
                "symbols": sorted(source_symbols),
                "observation_sha256": tool_observation_sha256(group),
                "later_mutation_groups": later_mutations,
                "symbol_disjoint_mutations": candidate_symbol_disjoint,
            }
        )
        symbol_disjoint_preservations += candidate_symbol_disjoint
    return candidates, {
        "mode": "versioned_symbol_observation_island_v45",
        "retained_groups_after_roll": len(retained_groups),
        "read_only_observations": read_only,
        "version_invalidated_observations": invalidated,
        "version_invalidation_reasons": invalidation_reasons,
        "symbol_disjoint_preservations": symbol_disjoint_preservations,
        "eligible_observations": len(candidates),
        "candidate_group_indices": [item["group_index"] for item in evidence],
        "candidate_evidence": evidence,
        "assistant_tokens_selected": 0,
        "latest_group_protected": False,
        "risk_reasons": [],
    }


def versioned_grounded_observation_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Bind V40's unchanged candidates to evidence for V45 target checks.

    The offline V45 motivation audit found no same-file, symbol-disjoint
    opportunity in the frozen V40 trajectories.  The active V45 arm therefore
    preserves V40's candidate order and adds target-time version validation.
    Path localization is attached to every candidate so the bridge can
    abstain after making the same selection as V40; it must not silently pick
    a different runner-up. Symbol extraction remains telemetry, not an
    admission rule.
    """

    candidates, decision = grounded_observation_candidates(retained_groups)
    evidence = []
    unlocalized = 0
    for candidate, index in zip(
        candidates, decision["candidate_group_indices"], strict=True
    ):
        del candidate
        group = retained_groups[index]
        paths = sorted(repository_paths(group))
        if not paths:
            unlocalized += 1
        evidence.append(
            {
                "group_index": index,
                "paths": paths,
                "symbols": sorted(repository_observation_symbols(group)),
                "observation_sha256": tool_observation_sha256(group),
            }
        )
    return candidates, {
        **decision,
        "mode": "versioned_grounded_observation_guard_v45",
        "candidate_evidence": evidence,
        "unlocalized_candidate_observations": unlocalized,
        "symbol_relaxation_enabled": False,
    }


def versioned_evidence_target_guard(
    pending: dict[str, Any],
    retained_groups: Sequence[Sequence[dict[str, Any]]],
    *,
    allow_symbol_disjoint: bool = True,
) -> dict[str, Any]:
    """Revalidate pending V45 evidence against newly completed mutations.

    Source selection validates only the groups visible on that request.  This
    target-time guard closes the next-request window: it finds the original
    observation in the current rolling context and checks every later write.
    Missing, duplicated, or unlocalized evidence fails closed.
    """

    expected_hash = str(pending.get("source_observation_sha256") or "")
    source_paths = {str(value) for value in pending.get("source_paths") or ()}
    source_symbols = {
        str(value) for value in pending.get("source_symbols") or ()
    }
    matches = [
        index
        for index, group in enumerate(retained_groups)
        if expected_hash and tool_observation_sha256(group) == expected_hash
    ]
    result: dict[str, Any] = {
        "applied": True,
        "target_evidence_valid": False,
        "reason": "source_observation_not_unique",
        "source_group_matches": len(matches),
        "source_group_index": matches[0] if len(matches) == 1 else None,
        "later_mutation_groups": 0,
        "symbol_disjoint_mutations": 0,
        "source_paths": sorted(source_paths),
        "source_symbols": sorted(source_symbols),
    }
    if len(matches) != 1 or not source_paths:
        return result

    source_index = matches[0]
    for later_index, later in enumerate(
        retained_groups[source_index + 1 :], start=source_index + 1
    ):
        if not repository_commit_phase_event(later):
            continue
        result["later_mutation_groups"] += 1
        effect = _mutation_effect_on_evidence(
            source_paths=source_paths,
            source_symbols=source_symbols,
            mutation=later,
        )
        if effect["reason"] == "same_file_symbol_disjoint_mutation":
            result["symbol_disjoint_mutations"] += 1
            if not allow_symbol_disjoint:
                result.update(
                    {
                        "reason": "same_file_symbol_disjoint_not_enabled",
                        "invalidating_group_index": later_index,
                        "changed_paths": effect["changed_paths"],
                        "changed_symbols": effect["changed_symbols"],
                    }
                )
                return result
        if effect["invalidates"]:
            result.update(
                {
                    "reason": effect["reason"],
                    "invalidating_group_index": later_index,
                    "changed_paths": effect["changed_paths"],
                    "changed_symbols": effect["changed_symbols"],
                }
            )
            return result

    result.update(
        {
            "target_evidence_valid": True,
            "reason": "version_evidence_valid",
        }
    )
    return result


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


def is_successful_focused_validation(
    group: Sequence[dict[str, Any]],
) -> bool:
    """Identify a successful focused execution, excluding state-only tools."""

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    return bool(
        _FOCUSED_VALIDATION_COMMAND.search(commands)
        and is_successful_executable_evidence(group)
        and not critical_coding_event_reasons(group)
    )


def is_shell_source_write(
    group: Sequence[dict[str, Any]],
) -> bool:
    """Identify a shell redirection that writes a source/configuration file."""

    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    return bool(_SHELL_SOURCE_WRITE.search(commands))


def repository_commit_phase_event(
    group: Sequence[dict[str, Any]],
) -> bool:
    """Return whether completed online evidence starts the commit phase."""

    return is_shell_source_write(group) or (
        "repository_mutation_command"
        in critical_coding_event_reasons(group)
    )


def coding_patch_lifecycle_target_reasons(
    groups: Sequence[Sequence[dict[str, Any]]],
) -> list[str]:
    """Protect repair, first-validation, and patch-review decisions.

    This is deliberately separate from the older risk classifier so adding
    shell-write coverage cannot silently change any frozen V31--V35 arm.
    """

    if not groups:
        return []
    latest = groups[-1]
    if is_high_value_executable_failure(latest):
        return ["executable_failure_before_repair"]
    if (
        "repository_diff_observed"
        in critical_coding_event_reasons(latest)
    ):
        return ["patch_diff_before_submission_decision"]
    if not is_successful_focused_validation(latest):
        return []
    state_changes = [
        index
        for index, group in enumerate(groups[:-1])
        if is_shell_source_write(group)
        or any(
            reason
            in {"repository_mutation_command", "repository_diff_observed"}
            for reason in critical_coding_event_reasons(group)
        )
    ]
    if not state_changes:
        return []
    latest_change = state_changes[-1]
    if any(
        is_successful_focused_validation(group)
        for group in groups[latest_change + 1 : -1]
    ):
        return []
    return ["first_validation_of_version_before_submit"]


def coding_version_validation_target_reasons(
    groups: Sequence[Sequence[dict[str, Any]]],
) -> list[str]:
    """Protect repair and first-validation decisions for a code version."""

    if not groups:
        return []
    latest = groups[-1]
    if is_high_value_executable_failure(latest):
        return ["executable_failure_before_repair"]
    if not is_successful_focused_validation(latest):
        return []
    state_changes = [
        index
        for index, group in enumerate(groups[:-1])
        if any(
            reason
            in {"repository_mutation_command", "repository_diff_observed"}
            for reason in critical_coding_event_reasons(group)
        )
    ]
    if not state_changes:
        return []
    latest_change = state_changes[-1]
    if any(
        is_successful_focused_validation(group)
        for group in groups[latest_change + 1 : -1]
    ):
        return []
    return ["first_validation_of_version_before_submit"]


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
    if arm in (
        "general",
        "general_8k",
        "general_dual_4k",
        "coding_state_transition_target_v33b",
        "coding_critical_current_target_v34",
        "coding_version_validation_target_v35b",
        "coding_patch_lifecycle_target_v37",
        "coding_commit_phase_dense_v38",
    ):
        decision["mode"] = (
            "general_contiguous_8k"
            if arm == "general_8k"
            else "general_dual_contiguous_4k"
            if arm == "general_dual_4k"
            else "state_transition_general_source"
            if arm == "coding_state_transition_target_v33b"
            else "critical_current_target_general_source"
            if arm == "coding_critical_current_target_v34"
            else "version_validation_target_general_source"
            if arm == "coding_version_validation_target_v35b"
            else "patch_lifecycle_target_general_source"
            if arm == "coding_patch_lifecycle_target_v37"
            else "commit_phase_exploration_general_source"
            if arm == "coding_commit_phase_dense_v38"
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
