"""Auditable coding-event policies for bridge KV reuse.

The original ``coding_aware`` arm protects the newest completed interaction on
every request.  The policies in this module keep General reuse as the default
and protect that interaction only when its command or observation carries a
concrete software-engineering risk signal.
"""

from __future__ import annotations

import ast
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
_OBSERVED_REPOSITORY_FILE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"((?:/testbed/|\./|a/|b/)?"
    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"
    r"\.(?:py|pyi|toml|yaml|yml|json|rst|md|cfg|ini|txt))\b"
)
_EXPLICIT_CODING_SYMBOL = re.compile(
    r"`([A-Za-z_]\w*)`|\b(?:class|def|function|method)\s+([A-Za-z_]\w*)\b"
)
_REPOSITORY_SCOPE_SEARCH = re.compile(
    r"(?:^|&&\s*|;\s*|\|\|\s*)find\s+"
    r"|\bgrep\b[^\n;&|]*(?:\s-[A-Za-z]*[rR][A-Za-z]*\b|--recursive\b)",
    re.I,
)
_SEARCH_RESULT_FILE_LINE = re.compile(
    r"^(?P<path>(?:/testbed/|\./)?[A-Za-z0-9_.-]+"
    r"(?:/[A-Za-z0-9_.-]+)*\."
    r"(?:py|pyi|toml|yaml|yml|json|rst|md|cfg|ini|txt))"
    r"(?::|-)(?P<line>\d+)(?::|-)"
)
_TOOL_LITERAL_PREFIX = "<|im_start|>user\n<tool_response>\n"
_TOOL_LITERAL_SUFFIX = "\n</tool_response><|im_end|>\n"

# Frozen from the 24-case SGLang natural repository-code stage experiment in
# ``impactkv_natural_module_attention_20260808/.../stage_overhead_code_only_r2``.
# The response variable is paired median Dense TTFT minus reuse TTFT.  The
# predictor approximates prefill attention work rather than using one fixed
# island-token threshold:
#
#   saving_ms = slope * (island_tokens * target_prompt_tokens / 10_000)
#               - fixed_overhead_ms
#
# This is an exploratory engineering model (24 observations, R^2=0.8750), not
# an accuracy proxy and not an independently confirmed speed claim.
NATURAL_CODE_COST_FIXED_OVERHEAD_MS = 14.66811245
NATURAL_CODE_COST_WORK_SLOPE_MS = 0.13169242
NATURAL_CODE_COST_CALIBRATION_CASES = 24
NATURAL_CODE_COST_CALIBRATION_R2 = 0.8750207389619562

# Frozen before running any new dependency-graph accuracy experiment.  These
# values come from ``calibrate_dependency_graph_lcb.py`` over all 56 completed
# exact-prompt targets.  Five folds keep whole agent tasks together.  The
# online score adds the cross-validated residual 10th percentile, making the
# admission threshold deliberately more conservative than a positive mean.
DEPENDENCY_GRAPH_LCB_WORK_SLOPE_MS = 0.15728623490986118
DEPENDENCY_GRAPH_LCB_INTERCEPT_MS = 0.25435619580085245
DEPENDENCY_GRAPH_LCB_RESIDUAL_Q10_MS = -78.79832246051551
DEPENDENCY_GRAPH_LCB_CALIBRATION_TARGETS = 56
DEPENDENCY_GRAPH_LCB_CALIBRATION_TASKS = 7
DEPENDENCY_GRAPH_LCB_CALIBRATION_R2 = 0.8745374212745528

_OUTPUT_BLOCK = re.compile(r"<output>(.*?)</output>", re.I | re.S)
_AMBIGUOUS_UNQUALIFIED_SYMBOL = re.compile(r"^__\w+__$")


def natural_code_reuse_cost_estimate(
    *, island_tokens: int, target_prompt_tokens: int
) -> dict[str, Any]:
    """Estimate cache-ready TTFT benefit for one natural code module.

    The policy admits any strictly positive prediction.  Source build is
    intentionally excluded because online sources are materialized by the
    preceding real agent request; no synthetic prefetch request is allowed.
    """

    if island_tokens <= 0 or target_prompt_tokens <= 0:
        raise ValueError("token counts must be positive")
    attention_work_10k = island_tokens * target_prompt_tokens / 10_000
    saving_ms = (
        NATURAL_CODE_COST_WORK_SLOPE_MS * attention_work_10k
        - NATURAL_CODE_COST_FIXED_OVERHEAD_MS
    )
    return {
        "model": "natural_code_attention_work_linear_v1",
        "island_tokens": island_tokens,
        "target_prompt_tokens": target_prompt_tokens,
        "attention_work_token2": island_tokens * target_prompt_tokens,
        "predicted_cache_ready_saving_ms": saving_ms,
        "reuse_admitted": saving_ms > 0,
        "calibration_cases": NATURAL_CODE_COST_CALIBRATION_CASES,
        "calibration_r2": NATURAL_CODE_COST_CALIBRATION_R2,
        "source_build_included": False,
    }


def dependency_graph_lcb_cost_estimate(
    *, island_tokens: int, target_prompt_tokens: int
) -> dict[str, Any]:
    """Return the frozen conservative TTFT admission score for one island."""

    if island_tokens <= 0 or target_prompt_tokens <= 0:
        raise ValueError("token counts must be positive")
    attention_work_10k = island_tokens * target_prompt_tokens / 10_000
    predicted = (
        DEPENDENCY_GRAPH_LCB_WORK_SLOPE_MS * attention_work_10k
        + DEPENDENCY_GRAPH_LCB_INTERCEPT_MS
    )
    lower_bound = predicted + DEPENDENCY_GRAPH_LCB_RESIDUAL_Q10_MS
    return {
        "model": "dependency_graph_attention_work_lcb_v1",
        "island_tokens": island_tokens,
        "target_prompt_tokens": target_prompt_tokens,
        "attention_work_token2": island_tokens * target_prompt_tokens,
        "predicted_cache_ready_saving_ms": predicted,
        "residual_q10_ms": DEPENDENCY_GRAPH_LCB_RESIDUAL_Q10_MS,
        "lower_bound_cache_ready_saving_ms": lower_bound,
        "reuse_admitted": lower_bound > 0,
        "calibration_targets": DEPENDENCY_GRAPH_LCB_CALIBRATION_TARGETS,
        "calibration_tasks": DEPENDENCY_GRAPH_LCB_CALIBRATION_TASKS,
        "calibration_r2": DEPENDENCY_GRAPH_LCB_CALIBRATION_R2,
        "task_grouped_folds": 5,
        "source_build_included": False,
    }


def dependency_graph_mean_cost_estimate(
    *, island_tokens: int, target_prompt_tokens: int
) -> dict[str, Any]:
    """Use the same frozen graph calibration but admit positive mean benefit.

    This is the single-variable counterfactual to
    :func:`dependency_graph_lcb_cost_estimate`: candidate extraction,
    dependency protection, regression coefficients, and one-island limit stay
    fixed; only the deliberately conservative residual-Q10 subtraction is
    removed from admission.  It remains a speed gate and never acts as an
    accuracy proxy.
    """

    if island_tokens <= 0 or target_prompt_tokens <= 0:
        raise ValueError("token counts must be positive")
    attention_work_10k = island_tokens * target_prompt_tokens / 10_000
    predicted = (
        DEPENDENCY_GRAPH_LCB_WORK_SLOPE_MS * attention_work_10k
        + DEPENDENCY_GRAPH_LCB_INTERCEPT_MS
    )
    return {
        "model": "dependency_graph_attention_work_mean_v1",
        "island_tokens": island_tokens,
        "target_prompt_tokens": target_prompt_tokens,
        "attention_work_token2": island_tokens * target_prompt_tokens,
        "predicted_cache_ready_saving_ms": predicted,
        "reuse_admitted": predicted > 0,
        "calibration_targets": DEPENDENCY_GRAPH_LCB_CALIBRATION_TARGETS,
        "calibration_tasks": DEPENDENCY_GRAPH_LCB_CALIBRATION_TASKS,
        "calibration_r2": DEPENDENCY_GRAPH_LCB_CALIBRATION_R2,
        "task_grouped_folds": 5,
        "source_build_included": False,
    }


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


def coding_group_sha256(group: Sequence[dict[str, Any]]) -> str:
    """Return an identity for one online-visible assistant/tool turn."""

    payload = json.dumps(
        list(group),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_observed_repository_path(value: str) -> str:
    value = value.strip()
    if value.startswith("/testbed/"):
        return value[len("/testbed/") :]
    if value.startswith("./"):
        return value[2:]
    if value.startswith(("a/", "b/")):
        return value[2:]
    return value


def observed_repository_path_provenance(
    group: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Extract literal path dependencies from one visible coding turn.

    V45 originally localized evidence from the command alone.  V46 also uses
    paths printed by that command, for example ``find`` results or diff
    headers.  Recursive searches depend on the repository scope rather than
    only on files that happened to match, so any subsequent repository write
    invalidates them.
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
    command_paths = repository_paths(group)
    literal_paths = {
        _normalize_observed_repository_path(match.group(1))
        for match in _OBSERVED_REPOSITORY_FILE.finditer(
            commands + "\n" + observations
        )
    }
    paths = {value for value in command_paths | literal_paths if value}
    return {
        "paths": sorted(paths),
        "command_paths": sorted(command_paths),
        "observation_added_paths": sorted(paths - command_paths),
        "repository_scope_dependency": bool(
            _REPOSITORY_SCOPE_SEARCH.search(commands)
        ),
    }


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


def _tool_observation_output(
    group: Sequence[dict[str, Any]],
) -> str:
    """Return only command output already present in the serialized prompt."""

    outputs: list[str] = []
    for message in group:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        matches = _OUTPUT_BLOCK.findall(content)
        outputs.extend(matches if matches else [content])
    return "\n".join(outputs)


def _python_module_for_path(path: str) -> str:
    normalized = _normalize_observed_repository_path(path)
    if normalized.endswith(".pyi"):
        normalized = normalized[:-4]
    elif normalized.endswith(".py"):
        normalized = normalized[:-3]
    else:
        return ""
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _dotted_ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_ast_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class _VisiblePythonGraph(ast.NodeVisitor):
    """Collect a small answer-blind dependency graph from visible Python."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.declared_qualified: set[str] = set()
        self.declared_symbols: set[str] = set()
        self.import_aliases: dict[str, str] = {}
        self.import_targets: set[str] = set()
        self.referenced_names: set[str] = set()
        self.called_symbols: set[str] = set()

    def _visit_declaration(self, node: ast.AST, name: str) -> None:
        self.declared_symbols.add(name)
        self.declared_qualified.add(".".join([*self.scope, name]))
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_declaration(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_declaration(node, node.name)

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_declaration(node, node.name)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.import_aliases[local] = alias.name
            self.import_targets.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        prefix = "." * node.level + (node.module or "")
        for alias in node.names:
            target = f"{prefix}.{alias.name}" if prefix else alias.name
            local = alias.asname or alias.name
            self.import_aliases[local] = target
            self.import_targets.add(target)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.referenced_names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        dotted = _dotted_ast_name(node)
        if dotted:
            self.referenced_names.add(dotted)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        dotted = _dotted_ast_name(node.func)
        if dotted:
            self.called_symbols.add(dotted)
        self.generic_visit(node)


def _resolve_visible_alias(value: str, aliases: dict[str, str]) -> str:
    first, separator, rest = value.partition(".")
    target = aliases.get(first)
    if target is None:
        return value
    return target + (separator + rest if separator else "")


def visible_python_dependency_graph(
    *, path: str, group: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Build a one-file graph solely from code already shown to the agent.

    Partial ``sed`` windows are common and may not parse as a module.  Such
    observations retain lexical/path guards rather than consulting the hidden
    checkout or pretending the missing syntax is known.
    """

    module = _python_module_for_path(path)
    result: dict[str, Any] = {
        "path": _normalize_observed_repository_path(path),
        "module": module,
        "parse_status": "non_python",
        "declared_symbols": [],
        "qualified_symbols": [],
        "import_aliases": {},
        "import_targets": [],
        "referenced_names": [],
        "called_symbols": [],
    }
    if not module:
        return result
    code = _tool_observation_output(group)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError) as error:
        result.update(
            parse_status="lexical_fallback",
            parse_error=type(error).__name__,
            declared_symbols=sorted(repository_observation_symbols(group)),
        )
        return result

    visitor = _VisiblePythonGraph()
    visitor.visit(tree)
    resolved_references = {
        _resolve_visible_alias(value, visitor.import_aliases)
        for value in visitor.referenced_names
    }
    resolved_calls = {
        _resolve_visible_alias(value, visitor.import_aliases)
        for value in visitor.called_symbols
    }
    qualified = set(visitor.declared_qualified)
    qualified.update(
        f"{module}.{value}" for value in visitor.declared_qualified if module
    )
    result.update(
        parse_status="parsed",
        declared_symbols=sorted(visitor.declared_symbols),
        qualified_symbols=sorted(qualified),
        import_aliases=dict(sorted(visitor.import_aliases.items())),
        import_targets=sorted(visitor.import_targets),
        referenced_names=sorted(resolved_references),
        called_symbols=sorted(resolved_calls),
    )
    return result


def _usable_unqualified_symbol(symbol: str) -> bool:
    return bool(symbol) and not _AMBIGUOUS_UNQUALIFIED_SYMBOL.fullmatch(symbol)


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


def versioned_observed_path_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Return read-only observations with online-visible path provenance.

    Direct reads retain V45's file-version check. Directory-wide ``find`` or
    recursive ``grep`` results are treated as repository-scope evidence and
    fail closed after any later repository mutation. Assistant reasoning and
    future groups are never selected.
    """

    candidates: list[list[dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    read_only = 0
    invalidated = 0
    unlocalized = 0
    observed_path_candidates = 0
    repository_scope_candidates = 0
    for index, group in enumerate(retained_groups):
        if not is_successful_readonly_evidence(group):
            continue
        read_only += 1
        provenance = observed_repository_path_provenance(group)
        source_paths = set(provenance["paths"])
        if not source_paths:
            unlocalized += 1
            continue
        source_symbols = repository_observation_symbols(group)
        invalidating_reason: str | None = None
        for later in retained_groups[index + 1 :]:
            if not repository_commit_phase_event(later):
                continue
            if provenance["repository_scope_dependency"]:
                invalidating_reason = "repository_scope_mutated"
                break
            effect = _mutation_effect_on_evidence(
                source_paths=source_paths,
                source_symbols=source_symbols,
                mutation=later,
            )
            if effect["invalidates"] or effect["reason"] == (
                "same_file_symbol_disjoint_mutation"
            ):
                invalidating_reason = str(effect["reason"])
                break
        if invalidating_reason is not None:
            invalidated += 1
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
                "group_sha256": coding_group_sha256(group),
                "paths": sorted(source_paths),
                "symbols": sorted(source_symbols),
                "observation_sha256": tool_observation_sha256(group),
                "path_provenance": provenance,
            }
        )
        if provenance["observation_added_paths"]:
            observed_path_candidates += 1
        if provenance["repository_scope_dependency"]:
            repository_scope_candidates += 1
    return candidates, {
        "mode": "observed_path_version_guard_v46",
        "retained_groups_after_roll": len(retained_groups),
        "read_only_observations": read_only,
        "version_invalidated_observations": invalidated,
        "unlocalized_candidate_observations": unlocalized,
        "eligible_observations": len(candidates),
        "candidate_group_indices": [item["group_index"] for item in evidence],
        "candidate_evidence": evidence,
        "observation_path_candidates": observed_path_candidates,
        "repository_scope_candidates": repository_scope_candidates,
        "assistant_tokens_selected": 0,
        "latest_group_protected": False,
        "risk_reasons": [],
        "symbol_relaxation_enabled": False,
    }


def natural_repository_code_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Return version-valid, single-file direct-read code observations.

    This is the online counterpart of the natural-module motivation study.
    Repository searches are excluded because their module boundary and
    dependency scope are qualitatively different; assistant interpretation,
    commands, test output, and mutation feedback are never candidates.  A
    direct result naming more than one repository file is rejected rather
    than inventing a fixed token split inside an ambiguous tool payload.
    """

    broad_candidates, broad = versioned_observed_path_candidates(
        retained_groups
    )
    candidates: list[list[dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    excluded_search = 0
    excluded_ambiguous_files = 0
    for candidate, item in zip(
        broad_candidates, broad["candidate_evidence"], strict=True
    ):
        group = retained_groups[int(item["group_index"])]
        commands = "\n".join(
            command
            for message in group
            if (command := _tool_command(message))
        )
        if _SEARCH_COMMAND.search(commands) or item["path_provenance"][
            "repository_scope_dependency"
        ]:
            excluded_search += 1
            continue
        if len(item["paths"]) != 1:
            excluded_ambiguous_files += 1
            continue
        candidates.append(candidate)
        evidence.append({**item, "module_type": "repository_code"})
    return candidates, {
        **broad,
        "mode": "natural_repository_code_version_guard_cost_v1",
        "eligible_observations_before_module_filter": broad[
            "eligible_observations"
        ],
        "eligible_observations": len(candidates),
        "candidate_group_indices": [item["group_index"] for item in evidence],
        "candidate_evidence": evidence,
        "excluded_repository_searches": excluded_search,
        "excluded_ambiguous_multifile_results": excluded_ambiguous_files,
        "selected_module_type": "repository_code",
    }


def _literal_search_file_sections(
    tool_messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return maximal literal file-line sections and candidate-char offsets."""

    rows: list[dict[str, Any]] = []
    literal_cursor = 0
    for message in tool_messages:
        content = str(message.get("content") or "")
        content_literal_start = literal_cursor + len(_TOOL_LITERAL_PREFIX)
        current_path: str | None = None
        current_start: int | None = None
        current_end: int | None = None

        def flush() -> None:
            nonlocal current_path, current_start, current_end
            if current_path is not None and current_start is not None and current_end is not None:
                section = content[current_start:current_end]
                rows.append(
                    {
                        "path": current_path,
                        "text": section,
                        "candidate_char_start": content_literal_start + current_start,
                        "candidate_char_end": content_literal_start + current_end,
                    }
                )
            current_path = None
            current_start = None
            current_end = None

        cursor = 0
        for line in content.splitlines(keepends=True):
            match = _SEARCH_RESULT_FILE_LINE.match(line)
            path = (
                _normalize_observed_repository_path(match.group("path"))
                if match
                else None
            )
            if path != current_path:
                flush()
            if path is not None:
                if current_start is None:
                    current_start = cursor
                current_path = path
                current_end = cursor + len(line)
            cursor += len(line)
        flush()
        literal_cursor += len(_TOOL_LITERAL_PREFIX) + len(content) + len(
            _TOOL_LITERAL_SUFFIX
        )
    return rows


def search_file_section_dependency_cold_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Select version-valid dependency-cold literal file sections in searches.

    Boundaries come exclusively from path-prefixed lines already visible in a
    successful ``grep``/``rg``/``find`` observation.  Each section is bound to
    one file, so an unrelated repository mutation does not invalidate it.
    Target-time group identity and observation hashes are still revalidated.
    """

    candidates: list[list[dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    search_groups = 0
    literal_sections = 0
    version_invalidated = 0
    dependency_hot = 0
    for index, group in enumerate(retained_groups):
        if not is_successful_readonly_evidence(group):
            continue
        commands = "\n".join(
            value for message in group if (value := _tool_command(message))
        )
        if not _SEARCH_COMMAND.search(commands):
            continue
        tool_messages = [
            dict(message) for message in group if message.get("role") == "tool"
        ]
        if not tool_messages:
            continue
        search_groups += 1
        sections = _literal_search_file_sections(tool_messages)
        literal_sections += len(sections)
        for section in sections:
            source_paths = {str(section["path"])}
            synthetic = [{"role": "tool", "content": str(section["text"])}]
            source_symbols = repository_observation_symbols(synthetic)
            invalid = False
            for later in retained_groups[index + 1 :]:
                if not repository_commit_phase_event(later):
                    continue
                effect = _mutation_effect_on_evidence(
                    source_paths=source_paths,
                    source_symbols=source_symbols,
                    mutation=later,
                )
                if effect["invalidates"] or effect["reason"] == (
                    "same_file_symbol_disjoint_mutation"
                ):
                    invalid = True
                    break
            if invalid:
                version_invalidated += 1
                continue
            graph = visible_python_dependency_graph(
                path=str(section["path"]), group=synthetic
            )
            symbols = source_symbols | set(graph.get("declared_symbols") or ())
            relations = coding_dependency_graph_relations(
                source_paths=source_paths,
                source_symbols=symbols,
                source_graph=graph,
                later_groups=retained_groups[index + 1 :],
            )
            if relations:
                dependency_hot += 1
                continue
            candidates.append(tool_messages)
            evidence.append(
                {
                    "group_index": index,
                    "group_sha256": coding_group_sha256(group),
                    "paths": [str(section["path"])],
                    "symbols": sorted(symbols),
                    "observation_sha256": tool_observation_sha256(group),
                    "path_provenance": {
                        "paths": [str(section["path"])],
                        "command_paths": [],
                        "observation_added_paths": [str(section["path"])],
                        "repository_scope_dependency": False,
                    },
                    "module_type": "repository_search_file_section",
                    "dependency_graph": graph,
                    "dependency_hot": False,
                    "dependency_relations": [],
                    "candidate_char_start": int(section["candidate_char_start"]),
                    "candidate_char_end": int(section["candidate_char_end"]),
                    "section_characters": len(str(section["text"])),
                }
            )
    return candidates, {
        "mode": "search_file_section_dependency_cold_mean_v1",
        "retained_groups_after_roll": len(retained_groups),
        "search_groups": search_groups,
        "literal_file_sections": literal_sections,
        "version_invalidated_sections": version_invalidated,
        "dependency_hot_sections_protected": dependency_hot,
        "dependency_cold_sections": len(candidates),
        "eligible_observations": len(candidates),
        "candidate_group_indices": [item["group_index"] for item in evidence],
        "candidate_evidence": evidence,
        "selected_module_type": "repository_search_file_section",
        "dependency_direction": "graph_hot_recompute_graph_cold_lossy_copy",
        "natural_boundary": "literal_contiguous_file_prefixed_lines",
        "assistant_tokens_selected": 0,
        "hidden_repository_scan": False,
    }


def coding_dependency_relations(
    *,
    source_paths: set[str],
    source_symbols: set[str],
    later_groups: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Find online-visible consumers of one earlier code observation.

    This is intentionally a protection signal.  A later turn that names the
    same file or an explicit symbol from the observed code makes the source
    dependency-hot, so its old K/V must not be copied into the current prompt.
    The detector uses only text and tool calls already visible before the
    target generation; it never consults Attention, a Dense answer, or task
    outcomes.
    """

    relations: list[dict[str, Any]] = []
    for later_index, group in enumerate(later_groups):
        assistant_text = "\n".join(
            str(message.get("content") or "")
            for message in group
            if message.get("role") == "assistant"
        )
        visible_text = "\n".join(
            str(message.get("content") or "")
            for message in group
        )
        later_paths = set(observed_repository_path_provenance(group)["paths"])
        later_paths.update(
            _normalize_observed_repository_path(match.group(1))
            for match in _OBSERVED_REPOSITORY_FILE.finditer(assistant_text)
        )
        later_symbols = repository_observation_symbols(group)
        later_symbols.update(
            next(part for part in match.groups() if part)
            for match in _EXPLICIT_CODING_SYMBOL.finditer(visible_text)
        )
        # A focused command often names a symbol without a ``def`` or
        # backticks.  Match only symbols actually declared in the source to
        # avoid treating arbitrary identifier-like words as dependencies.
        for symbol in source_symbols:
            if re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", visible_text):
                later_symbols.add(symbol)
        path_overlap = sorted(source_paths & later_paths)
        symbol_overlap = sorted(source_symbols & later_symbols)
        if not path_overlap and not symbol_overlap:
            continue
        relations.append(
            {
                "later_group_offset": later_index,
                "exact_paths": path_overlap,
                "shared_symbols": symbol_overlap,
                "assistant_interpretation_grounding": bool(
                    assistant_text and (path_overlap or symbol_overlap)
                ),
            }
        )
    return relations


def cold_natural_repository_code_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Keep only version-valid code observations with no visible consumer.

    ``natural_repository_code_candidates`` establishes module validity.  This
    second stage changes the ranking direction justified by the Hot/Cold
    physical-splice study: dependency-hot code is protected (recomputed), and
    only dependency-cold code can enter the lossy KV pool.
    """

    broad_candidates, broad = natural_repository_code_candidates(retained_groups)
    candidates: list[list[dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    hot = 0
    for candidate, item in zip(
        broad_candidates, broad["candidate_evidence"], strict=True
    ):
        source_index = int(item["group_index"])
        relations = coding_dependency_relations(
            source_paths=set(item["paths"]),
            source_symbols=set(item["symbols"]),
            later_groups=retained_groups[source_index + 1 :],
        )
        if relations:
            hot += 1
            continue
        candidates.append(candidate)
        evidence.append(
            {
                **item,
                "dependency_hot": False,
                "dependency_relations": [],
            }
        )
    return candidates, {
        **broad,
        "mode": "natural_repository_code_dependency_cold_cost_v1",
        "eligible_observations_before_dependency_guard": broad[
            "eligible_observations"
        ],
        "eligible_observations": len(candidates),
        "candidate_group_indices": [item["group_index"] for item in evidence],
        "candidate_evidence": evidence,
        "dependency_hot_observations_protected": hot,
        "dependency_cold_observations": len(candidates),
        "dependency_direction": "hot_recompute_cold_lossy_copy",
    }


def coding_dependency_target_guard(
    pending: dict[str, Any],
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Revalidate version evidence, then veto dependency-hot KV copies."""

    result = observed_path_target_guard(pending, retained_groups)
    result.update(
        dependency_guard_applied=True,
        dependency_direction="hot_recompute_cold_lossy_copy",
        dependency_hot=None,
        dependency_relations=[],
    )
    if not result["target_evidence_valid"]:
        return result
    source_index = int(result["source_group_index"])
    relations = coding_dependency_relations(
        source_paths=set(pending.get("source_paths") or ()),
        source_symbols=set(pending.get("source_symbols") or ()),
        later_groups=retained_groups[source_index + 1 :],
    )
    hot = bool(relations)
    result.update(
        dependency_hot=hot,
        dependency_relations=relations,
        target_evidence_valid=not hot,
        reason=(
            "coding_dependency_hot_protected"
            if hot
            else "coding_dependency_cold_version_valid"
        ),
    )
    return result


def coding_dependency_graph_relations(
    *,
    source_paths: set[str],
    source_symbols: set[str],
    source_graph: dict[str, Any],
    later_groups: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Find visible one-hop path/import/call consumers of source code.

    Unlike the earlier flat-symbol detector, cross-file dunder names such as
    ``__init__`` and ``__call__`` cannot independently make an observation
    hot.  Qualified names and import aliases are resolved from visible Python
    ASTs.  Syntax-incomplete windows fall back to conservative lexical
    matching, and the hidden repository is never scanned.
    """

    source_module = str(source_graph.get("module") or "")
    source_qualified = {
        str(value) for value in source_graph.get("qualified_symbols") or ()
    }
    usable_symbols = {
        symbol for symbol in source_symbols if _usable_unqualified_symbol(symbol)
    }
    relations: list[dict[str, Any]] = []
    for later_index, group in enumerate(later_groups):
        provenance = observed_repository_path_provenance(group)
        later_paths = set(provenance["paths"])
        exact_paths = sorted(source_paths & later_paths)

        later_graphs = [
            visible_python_dependency_graph(path=path, group=group)
            for path in sorted(later_paths)
            if _python_module_for_path(path)
        ]
        references = {
            str(value)
            for graph in later_graphs
            for value in graph.get("referenced_names") or ()
        }
        calls = {
            str(value)
            for graph in later_graphs
            for value in graph.get("called_symbols") or ()
        }
        import_targets = {
            str(value)
            for graph in later_graphs
            for value in graph.get("import_targets") or ()
        }
        qualified_matches = sorted(source_qualified & (references | calls))
        import_matches = sorted(
            target
            for target in import_targets
            if source_module
            and (target == source_module or target.startswith(source_module + "."))
        )
        direct_call_matches = sorted(
            usable_symbols
            & {value.rsplit(".", 1)[-1] for value in calls}
        )

        commands = "\n".join(
            command
            for message in group
            if (command := _tool_command(message))
        )
        assistant_text = "\n".join(
            str(message.get("content") or "")
            for message in group
            if message.get("role") == "assistant"
        )
        lexical_text = "\n".join(
            (commands, assistant_text, _tool_observation_output(group))
        )
        lexical_matches = sorted(
            symbol
            for symbol in usable_symbols
            if re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", lexical_text)
        )

        relation_kinds: list[str] = []
        if exact_paths:
            relation_kinds.append("exact_path")
        if qualified_matches:
            relation_kinds.append("qualified_symbol")
        if import_matches:
            relation_kinds.append("one_hop_import")
        if direct_call_matches:
            relation_kinds.append("direct_call")
        if lexical_matches and not (
            qualified_matches or import_matches or direct_call_matches
        ):
            relation_kinds.append("lexical_fallback")
        if not relation_kinds:
            continue
        relations.append(
            {
                "later_group_offset": later_index,
                "relation_kinds": relation_kinds,
                "exact_paths": exact_paths,
                "qualified_symbol_matches": qualified_matches,
                "import_matches": import_matches,
                "direct_call_matches": direct_call_matches,
                "lexical_symbol_matches": lexical_matches,
                "later_parse_statuses": [
                    str(graph["parse_status"]) for graph in later_graphs
                ],
            }
        )
    return relations


def dependency_graph_cold_repository_code_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Select version-valid code with no visible one-hop consumer."""

    broad_candidates, broad = natural_repository_code_candidates(retained_groups)
    candidates: list[list[dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    hot = 0
    relation_kind_counts: dict[str, int] = {}
    parse_status_counts: dict[str, int] = {}
    for candidate, item in zip(
        broad_candidates, broad["candidate_evidence"], strict=True
    ):
        source_index = int(item["group_index"])
        source_path = str(item["paths"][0])
        graph = visible_python_dependency_graph(
            path=source_path,
            group=retained_groups[source_index],
        )
        status = str(graph["parse_status"])
        parse_status_counts[status] = parse_status_counts.get(status, 0) + 1
        source_symbols = set(item["symbols"]) | set(
            graph.get("declared_symbols") or ()
        )
        relations = coding_dependency_graph_relations(
            source_paths=set(item["paths"]),
            source_symbols=source_symbols,
            source_graph=graph,
            later_groups=retained_groups[source_index + 1 :],
        )
        if relations:
            hot += 1
            for relation in relations:
                for kind in relation["relation_kinds"]:
                    relation_kind_counts[kind] = (
                        relation_kind_counts.get(kind, 0) + 1
                    )
            continue
        candidates.append(candidate)
        evidence.append(
            {
                **item,
                "symbols": sorted(source_symbols),
                "dependency_graph": graph,
                "dependency_hot": False,
                "dependency_relations": [],
            }
        )
    return candidates, {
        **broad,
        "mode": "dependency_graph_cold_lcb_single_island_v1",
        "eligible_observations_before_dependency_guard": broad[
            "eligible_observations"
        ],
        "eligible_observations": len(candidates),
        "candidate_group_indices": [item["group_index"] for item in evidence],
        "candidate_evidence": evidence,
        "dependency_hot_observations_protected": hot,
        "dependency_cold_observations": len(candidates),
        "dependency_relation_kind_counts": relation_kind_counts,
        "source_parse_status_counts": parse_status_counts,
        "dependency_direction": "graph_hot_recompute_graph_cold_lossy_copy",
        "hidden_repository_scan": False,
    }


def coding_dependency_graph_target_guard(
    pending: dict[str, Any],
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Revalidate file version and the visible graph before physical copy."""

    result = observed_path_target_guard(pending, retained_groups)
    result.update(
        dependency_graph_guard_applied=True,
        dependency_direction="graph_hot_recompute_graph_cold_lossy_copy",
        dependency_hot=None,
        dependency_relations=[],
    )
    if not result["target_evidence_valid"]:
        return result
    source_graph = pending.get("source_dependency_graph")
    if not isinstance(source_graph, dict):
        result.update(
            target_evidence_valid=False,
            reason="source_dependency_graph_missing",
        )
        return result
    source_index = int(result["source_group_index"])
    relations = coding_dependency_graph_relations(
        source_paths=set(pending.get("source_paths") or ()),
        source_symbols=set(pending.get("source_symbols") or ()),
        source_graph=source_graph,
        later_groups=retained_groups[source_index + 1 :],
    )
    hot = bool(relations)
    result.update(
        dependency_hot=hot,
        dependency_relations=relations,
        target_evidence_valid=not hot,
        reason=(
            "coding_dependency_graph_hot_protected"
            if hot
            else "coding_dependency_graph_cold_version_valid"
        ),
    )
    return result


def observed_path_target_guard(
    pending: dict[str, Any],
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Revalidate one V46 observation by group identity and path version."""

    expected_group = str(pending.get("source_group_sha256") or "")
    expected_observation = str(
        pending.get("source_observation_sha256") or ""
    )
    source_paths = {
        str(value) for value in pending.get("source_paths") or ()
    }
    source_symbols = {
        str(value) for value in pending.get("source_symbols") or ()
    }
    group_matches = [
        index
        for index, group in enumerate(retained_groups)
        if expected_group and coding_group_sha256(group) == expected_group
    ]
    result: dict[str, Any] = {
        "applied": True,
        "target_evidence_valid": False,
        "reason": "source_group_identity_not_unique",
        "source_group_matches": len(group_matches),
        "source_group_index": (
            group_matches[0] if len(group_matches) == 1 else None
        ),
        "later_mutation_groups": 0,
        "source_paths": sorted(source_paths),
        "source_symbols": sorted(source_symbols),
        "repository_scope_dependency": bool(
            pending.get("repository_scope_dependency", False)
        ),
    }
    if len(group_matches) != 1 or not source_paths:
        return result

    source_index = group_matches[0]
    if tool_observation_sha256(retained_groups[source_index]) != (
        expected_observation
    ):
        result["reason"] = "source_group_observation_mismatch"
        return result
    for later_index, later in enumerate(
        retained_groups[source_index + 1 :], start=source_index + 1
    ):
        if not repository_commit_phase_event(later):
            continue
        result["later_mutation_groups"] += 1
        if result["repository_scope_dependency"]:
            result.update(
                {
                    "reason": "repository_scope_mutated",
                    "invalidating_group_index": later_index,
                }
            )
            return result
        effect = _mutation_effect_on_evidence(
            source_paths=source_paths,
            source_symbols=source_symbols,
            mutation=later,
        )
        if effect["reason"] == "same_file_symbol_disjoint_mutation":
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
            "reason": "observed_path_version_evidence_valid",
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
