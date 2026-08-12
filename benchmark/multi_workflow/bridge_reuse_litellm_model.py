"""mini-SWE-agent model wrapper for native rolling-history KV reuse.

The wrapper gives Dense, general reuse, and coding-aware reuse the same
deterministic six-interaction history window.  Reuse selection is written to a
local version-3 sidecar before each request; no HTTP field may select KV spans.
The preceding real agent request is materialized as the source, so there is no
synthetic prefetch or replay request.
"""

from __future__ import annotations

import ast
import copy
import fcntl
import hashlib
import itertools
import json
import os
import re
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Literal

import litellm
import requests
from jinja2.utils import htmlsafe_json_dumps
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from pydantic import Field, model_validator

from benchmark.multi_workflow.coding_reuse_policy import (
    coding_dependency_graph_target_guard,
    coding_dependency_target_guard,
    coding_group_sha256,
    coding_patch_lifecycle_target_reasons,
    coding_state_transition_target_reasons,
    coding_version_validation_target_reasons,
    cold_natural_repository_code_candidates,
    critical_coding_event_reasons,
    dependency_graph_cold_repository_code_candidates,
    dependency_graph_lcb_cost_estimate,
    dependency_graph_mean_cost_estimate,
    effective_copy_cap,
    grounded_observation_candidates,
    natural_code_reuse_cost_estimate,
    natural_repository_code_candidates,
    observed_path_target_guard,
    post_mutation_payoff_guard,
    repository_commit_phase_event,
    select_failure_memory_groups,
    select_reuse_groups,
    search_file_section_dependency_cold_candidates,
    versioned_evidence_target_guard,
    versioned_grounded_observation_candidates,
    versioned_observed_path_candidates,
)
from benchmark.multi_workflow.context_bounded_litellm_model import (
    ContextBoundedLitellmModel,
    ContextBoundedLitellmModelConfig,
)
ROLLING_NOTICE = (
    '<history_compaction dropped_turn_groups="{dropped}">'
    "Earlier interaction details were omitted to stay within the rolling "
    "history budget. Repository state persists; the most recent complete "
    "interactions follow."
    "</history_compaction>"
)
_MODEL_INSTANCE_COUNTER = itertools.count(1)
DENSE_REUSE_ARMS = ("dense", "coding_memory_dense_v5")
MEMORY_ARMS = ("coding_memory_dense_v5", "coding_memory_v5")
TARGET_VETO_ARMS = (
    "coding_state_transition_target_v33b",
    "coding_critical_current_target_v34",
    "coding_version_validation_target_v35b",
    "coding_patch_lifecycle_target_v37",
    "coding_commit_phase_dense_v38",
)
NATURAL_CODE_COST_ARMS = (
    "coding_natural_code_cost",
    "coding_dependency_cold_cost",
    "coding_dependency_graph_cold_lcb",
    "coding_dependency_graph_cold_mean",
    "coding_search_file_section_mean",
)
DEPENDENCY_GRAPH_ARMS = (
    "coding_dependency_graph_cold_lcb",
    "coding_dependency_graph_cold_mean",
    "coding_search_file_section_mean",
)


def token_ids_hash(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(
            int(token_id).to_bytes(8, byteorder="little", signed=True)
        )
    return digest.hexdigest()


def native_generate_payload(
    *,
    backend: str | None,
    session_id: str,
    request_index: int,
    prompt_text_sha256: str,
    input_ids: list[int],
    segments: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
) -> dict[str, Any]:
    """Build the common request while preserving each native API's sampling schema."""
    payload = {
        "schema_version": 1,
        "backend": backend,
        "session_id": session_id,
        "request_index": request_index,
        "prompt_text_sha256": prompt_text_sha256,
        "input_ids": input_ids,
        "input_ids_sha256": token_ids_hash(input_ids),
        "segments": segments,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
    }
    # CacheBlend/KVCOMM expose the common fields above directly.  Stock
    # SGLang's /generate endpoint reads generation controls only from the
    # nested sampling_params object; without this translation it silently
    # falls back to the model's temperature=0.7 generation config.
    if str(backend or "").startswith("sglang-"):
        payload["sampling_params"] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }
    return payload


class BridgeReuseLitellmModelConfig(ContextBoundedLitellmModelConfig):
    reuse_arm: Literal[
        "dense",
        "general",
        "general_8k",
        "coding_aware",
        "coding_failure_v1",
        "coding_phase_v1",
        "coding_adaptive_v2",
        "coding_adaptive_v3",
        "coding_budget_v4",
        "coding_memory_dense_v5",
        "coding_memory_v5",
        "coding_source_guard_v6",
        "coding_evidence_payoff_v7",
        "general_dual_4k",
        "coding_dual_v8",
        "coding_version_graph_v17",
        "coding_post_mutation_v19",
        "coding_post_mutation_dual_v20",
        "coding_post_mutation_seam32_v22",
        "coding_post_mutation_target_prefix_v23",
        "coding_post_mutation_payoff_guard_v28",
        "coding_post_mutation_payoff_guard_v29",
        "coding_critical_event_abstain_v31",
        "coding_state_transition_target_v33b",
        "coding_critical_current_target_v34",
        "coding_version_validation_target_v35b",
        "coding_patch_lifecycle_target_v37",
        "coding_commit_phase_dense_v38",
        "coding_grounded_observation_island_v40",
        "coding_versioned_evidence_guard_v45",
        "coding_observed_path_pool_v46",
        "coding_natural_code_cost",
        "coding_dependency_cold_cost",
        "coding_dependency_graph_cold_lcb",
        "coding_dependency_graph_cold_mean",
        "coding_search_file_section_mean",
    ] = "dense"
    rolling_history_groups: int = Field(default=6, ge=4)
    reuse_copy_cap: int = Field(default=4096, ge=128)
    reuse_min_tokens: int = Field(default=128, ge=32)
    reuse_manifest_path: Path | None = None
    reuse_client_ledger_path: Path | None = None
    native_backend_url: str | None = None
    native_backend_name: str | None = None
    recover_unparsed_output_with_notice: bool = False

    @model_validator(mode="after")
    def validate_reuse_paths(self):
        if self.reuse_arm not in DENSE_REUSE_ARMS:
            if self.reuse_manifest_path is None:
                raise ValueError("reuse_manifest_path is required for reuse arms")
            if self.reuse_client_ledger_path is None:
                raise ValueError(
                    "reuse_client_ledger_path is required for reuse arms"
                )
        return self


def find_sublist(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    first = needle[0]
    return [
        start
        for start in range(len(haystack) - len(needle) + 1)
        if haystack[start] == first
        and haystack[start : start + len(needle)] == needle
    ]


def capped_tail(
    token_ids: list[int], start: int, cap: int
) -> tuple[list[int], int]:
    if len(token_ids) <= cap:
        return token_ids, start
    offset = len(token_ids) - cap
    return token_ids[offset:], start + offset


def close_litellm_sync_stream(stream: Any) -> None:
    """Release the HTTP response retained by LiteLLM's sync stream wrapper.

    ``CustomStreamWrapper`` exposes only an async ``aclose`` method even when
    its ``completion_stream`` is synchronous.  Leaving that underlying stream
    referenced after normal exhaustion accumulates CLOSE_WAIT sockets and can
    eventually deadlock the shared httpx connection pool on long agent runs.
    """

    completion_stream = getattr(stream, "completion_stream", None)
    if completion_stream is None:
        return
    try:
        close = getattr(completion_stream, "close", None)
        if callable(close):
            close()
    finally:
        stream.completion_stream = None


def apply_current_target_veto(
    *,
    arm: str,
    selected_groups: list[list[dict[str, Any]]],
    target: dict[str, Any] | None,
    releases: list[str],
    commit_phase_latched: bool = False,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    """Apply the V33B online target guard without consulting future output."""

    if arm == "coding_state_transition_target_v33b":
        reasons = coding_state_transition_target_reasons(selected_groups)
    elif arm == "coding_critical_current_target_v34":
        reasons = critical_coding_event_reasons(
            selected_groups[-1] if selected_groups else ()
        )
    elif arm == "coding_version_validation_target_v35b":
        reasons = coding_version_validation_target_reasons(selected_groups)
    elif arm == "coding_patch_lifecycle_target_v37":
        reasons = coding_patch_lifecycle_target_reasons(selected_groups)
    elif arm == "coding_commit_phase_dense_v38" and commit_phase_latched:
        reasons = ["repository_commit_phase_latched"]
    else:
        reasons = []
    vetoed = bool(reasons and target is not None)
    if vetoed:
        releases = list(
            dict.fromkeys([*releases, str(target["source_id"])])
        )
        target = None
    return target, releases, {
        "target_vetoed": vetoed,
        "target_veto_reasons": reasons,
    }


class BridgeReuseLitellmModel(ContextBoundedLitellmModel):
    """Run an agent with a shared rolling-compaction policy and local KV plans."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # ContextBoundedLitellmModel instantiated its base config class.  Parse
        # the same values into the stricter treatment config.
        self.config = BridgeReuseLitellmModelConfig(**kwargs)
        # mini-SWE-agent constructs a fresh model wrapper for each benchmark
        # instance in the same worker process.  Session/request counters
        # therefore are not globally unique within the append-only manifest.
        self._instance_nonce = (
            f"p{os.getpid()}-m{next(_MODEL_INSTANCE_COUNTER)}"
        )
        self._request_index = 0
        self._session_index = 0
        self._last_message_count = 0
        self._pending_source: dict[str, Any] | None = None
        self._pending_sources: dict[str, dict[str, Any]] = {}
        self._commit_phase_latched = False
        self._last_stream_stats: dict[str, Any] = {}
        if self.config.reuse_arm not in DENSE_REUSE_ARMS:
            # Workers are frozen to one.  A new wrapper therefore marks a
            # benchmark-instance boundary: sources left without any target by
            # the preceding instance can no longer have a future consumer.
            self._atomic_sidecar_update(release_orphaned_sources=True)

    def _render_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        prompt = self._render_prompt(messages)
        return self._tokenizer.encode(
            prompt, add_special_tokens=False
        ).ids

    def _render_prompt(self, messages: list[dict[str, Any]]) -> str:
        return self._chat_template.render(
            messages=self._template_messages(messages),
            tools=[BASH_TOOL],
            add_generation_prompt=True,
        )

    def _native_backend_segments(
        self,
        messages: list[dict[str, Any]],
        prompt_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Locate reusable rolling observations without changing target IDs."""

        candidates: list[dict[str, Any]] = []
        cursor = 0
        for message_index, message in enumerate(messages):
            if message_index < 2 or message.get("role") != "tool":
                continue
            literal = self._render_message_literal(message)
            ids = self._tokenizer.encode(
                literal, add_special_tokens=False
            ).ids
            matches = [start for start in find_sublist(prompt_ids, ids) if start >= cursor]
            if not matches:
                continue
            start = matches[0]
            end = start + len(ids)
            cursor = end
            candidates.append(
                {
                    "segment_id": hashlib.sha256(
                        f"tool:{message_index}:".encode()
                        + bytes.fromhex(token_ids_hash(ids))
                    ).hexdigest()[:24],
                    "message_index": message_index,
                    "role": "tool",
                    "start": start,
                    "end": end,
                    "length": len(ids),
                    "token_ids_sha256": token_ids_hash(ids),
                }
            )
        return candidates

    @staticmethod
    def _attach_embedded_tool_call(
        message: Any,
        call_id: str,
        recover_unparsed_output_with_notice: bool = False,
    ) -> None:
        if message.tool_calls or not isinstance(message.content, str):
            return
        content = message.content
        function_match = re.search(
            r"<tool_call>\s*<function=(?P<name>[^>]+)>\s*"
            r"<parameter=command>(?P<command>.*?)</parameter>\s*"
            r"</function>\s*</tool_call>",
            content,
            flags=re.DOTALL,
        )
        json_match = re.search(
            r"<tool_call>\s*(?P<call>\{.*?\})\s*</tool_call>",
            content,
            flags=re.DOTALL,
        )
        name = None
        arguments: dict[str, Any] | None = None
        start = None
        if function_match is not None:
            name = function_match.group("name").strip()
            arguments = {"command": function_match.group("command")}
            start = function_match.start()
        elif json_match is not None:
            try:
                value = json.loads(json_match.group("call"))
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                name = value.get("name")
                arguments = value.get("arguments")
                start = json_match.start()
        if name is None:
            # Qwen2.5-Coder sometimes emits the requested tool-call JSON in a
            # Markdown fence (or directly in prose) instead of wrapping it in
            # <tool_call>.  Decode balanced JSON objects rather than using a
            # non-greedy regex, which truncates the nested ``arguments`` map.
            decoder = json.JSONDecoder()
            for candidate_start, character in enumerate(content):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(content[candidate_start:])
                except json.JSONDecodeError:
                    continue
                if not isinstance(candidate, dict):
                    continue
                function = candidate.get("function", candidate)
                if not isinstance(function, dict):
                    continue
                candidate_name = function.get("name")
                candidate_arguments = function.get("arguments")
                if (
                    candidate_name == "bash"
                    and isinstance(candidate_arguments, dict)
                    and isinstance(candidate_arguments.get("command"), str)
                ):
                    name = candidate_name
                    arguments = candidate_arguments
                    start = candidate_start
                    break
        if name is None:
            # Shell commands commonly contain backslashes that are legal in a
            # shell string but invalid JSON escapes (observed: ``\;`` and
            # ``\'``). ``ast.literal_eval`` safely accepts the resulting
            # JSON-like dict without executing code. Limit candidates to a
            # whole fence/object and still require the exact bash schema.
            literal_candidates: list[tuple[int, str]] = []
            for fenced in re.finditer(
                r"```(?:json|bash|sh|shell)?\s*\n(?P<value>.*?)```",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            ):
                literal_candidates.append(
                    (fenced.start(), fenced.group("value").strip())
                )
            first_brace = content.find("{")
            last_brace = content.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                literal_candidates.append(
                    (first_brace, content[first_brace : last_brace + 1])
                )
            for candidate_start, candidate_text in literal_candidates:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        candidate = ast.literal_eval(candidate_text)
                except (SyntaxError, ValueError):
                    continue
                if not isinstance(candidate, dict):
                    continue
                function = candidate.get("function", candidate)
                if not isinstance(function, dict):
                    continue
                candidate_name = function.get("name")
                candidate_arguments = function.get("arguments")
                if (
                    candidate_name == "bash"
                    and isinstance(candidate_arguments, dict)
                    and isinstance(candidate_arguments.get("command"), str)
                ):
                    name = candidate_name
                    arguments = candidate_arguments
                    start = candidate_start
                    break
        if name is None:
            # A second observed Qwen fallback is a literal shell fence.  It is
            # still an unambiguous invocation of our only exposed tool.  Take
            # one fence as one action so the common max-one-call contract is
            # preserved even if the model narrates several future steps.  If
            # it first emits a standalone ``cd /testbed`` plan step, prefer
            # the first substantive fence because every tool call already
            # starts in /testbed and shell working directories do not persist.
            shell_matches = list(re.finditer(
                r"```(?:bash|sh|shell)\s*\n(?P<command>.*?)```",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            ))
            shell_match = next(
                (
                    match
                    for match in shell_matches
                    if match.group("command").strip()
                    not in {"cd /testbed", "cd -- /testbed"}
                ),
                shell_matches[0] if shell_matches else None,
            )
            if shell_match is not None and shell_match.group("command").strip():
                name = "bash"
                arguments = {"command": shell_match.group("command").strip()}
                start = shell_match.start()
        if (
            (not name or not isinstance(arguments, dict) or start is None)
            and recover_unparsed_output_with_notice
            and content.strip()
        ):
            # mini-SWE-agent does not retain an invalid assistant message; it
            # appends another FormatError user message instead.  Qwen2.5 can
            # therefore repeat the same prose-only output until the entire
            # call budget is exhausted.  Preserve the model's own text as
            # reasoning and execute only a non-mutating, task-independent
            # notice.  This breaks the transport-format loop without
            # inventing a repository search, edit, or solution for the model.
            name = "bash"
            arguments = {
                "command": (
                    "printf '%s\\n' 'NOTICE: the preceding assistant text "
                    "contained no executable tool call and changed nothing. "
                    "Your next response must contain exactly one non-empty "
                    "bash tool call; put the intended shell command inside "
                    "the command field.'"
                )
            }
            start = len(content)
        if not name or not isinstance(arguments, dict) or start is None:
            return
        command = arguments.get("command")
        if name == "bash" and isinstance(command, str):
            normalized = re.sub(r"\s+", " ", command.strip()).rstrip(";")
            if normalized in {"cd /testbed", "cd -- /testbed"}:
                # EnrootEnvironment already prefixes every action with
                # ``cd /testbed``.  Returning an explicit observation breaks
                # the observed silent no-op loop without inventing a
                # task-specific search or mutating the repository.
                arguments = {
                    "command": (
                        "pwd; printf '%s\\n' 'NOTICE: every action already "
                        "starts in /testbed. Do not issue standalone cd. "
                        "Next inspect narrowly with rg -n or sed -n.'"
                    )
                }
            elif re.search(
                r"(?:^|(?:&&|\|\||;|\|)\s*)(?:sudo\s+)?"
                r"(?:apt(?:-get)?|pip(?:3)?\s+install)\b",
                normalized,
            ):
                # SWE-bench containers are deliberately offline.  A failed
                # package install was observed to consume the entire 32-call
                # agent budget.  Return a backend-neutral observation without
                # inventing a task-specific search or repository mutation.
                arguments = {
                    "command": (
                        "printf '%s\\n' 'NOTICE: this container is offline; "
                        "package installation is unavailable. Do not retry "
                        "apt, sudo, or pip install. Use existing grep/find/sed "
                        "tools instead.'"
                    )
                }
        message.tool_calls = [
            litellm.ChatCompletionMessageToolCall(
                id=call_id,
                type="function",
                function={"name": str(name), "arguments": json.dumps(arguments)},
            )
        ]
        reasoning = content[:start].strip()
        message.content = reasoning or None

    @staticmethod
    def _render_message_literal(message: dict[str, Any]) -> str:
        message = copy.deepcopy(message)
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            value = "<|im_start|>assistant"
            if message.get("content"):
                value += "\n" + str(message["content"])
            for wrapped_call in message["tool_calls"]:
                call = wrapped_call.get("function", wrapped_call)
                arguments = call.get("arguments") or {}
                # ``_render_prompt`` first normalizes LiteLLM's serialized
                # argument string into the JSON object consumed by the frozen
                # Qwen2.5 template.  Span localization must render that same
                # object; otherwise the group literal contains an extra quoted
                # JSON string and can never occur in the actual prompt tokens.
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_arguments": arguments}
                value += (
                    '\n<tool_call>\n{"name": "'
                    + str(call["name"])
                    + '", "arguments": '
                    + str(htmlsafe_json_dumps(arguments))
                    + "}\n</tool_call>"
                )
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

    def _rolling_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], dict[str, Any]]:
        base = copy.deepcopy(messages[:2])
        groups = self._turn_groups(messages[2:])
        memory: dict[str, Any] = {}
        if self.config.reuse_arm in MEMORY_ARMS:
            selected, memory = select_failure_memory_groups(
                groups,
                recent_count=self.config.rolling_history_groups,
            )
            dropped = len(groups) - len(selected)
        else:
            dropped = max(
                0, len(groups) - self.config.rolling_history_groups
            )
            selected = groups[dropped:]
        if dropped == 0:
            output = [*base]
            for group in selected:
                output.extend(copy.deepcopy(group))
            return output, selected, {
                "applied": False,
                "total_groups": len(groups),
                "dropped_groups": 0,
                "retained_groups": len(selected),
                **memory,
            }
        notice = {
            "role": "user",
            "content": ROLLING_NOTICE.format(dropped=dropped),
        }
        output = [*base, notice]
        for group in selected:
            output.extend(copy.deepcopy(group))
        return output, selected, {
            "applied": True,
            "total_groups": len(groups),
            "dropped_groups": dropped,
            "retained_groups": len(selected),
            **memory,
        }

    def _atomic_sidecar_update(
        self,
        *,
        sources: list[dict[str, Any]] = [],
        cases: list[dict[str, Any]] = [],
        release_source_ids: list[str] = [],
        release_orphaned_sources: bool = False,
    ) -> None:
        path = self.config.reuse_manifest_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(path.read_text(encoding="utf-8"))
            value.setdefault("sources", []).extend(sources)
            value.setdefault("cases", []).extend(cases)
            releases = value.setdefault("release_source_ids", [])
            if release_orphaned_sources:
                referenced_sources = {
                    str(case["source_id"])
                    for case in value["cases"]
                    if case.get("source_id") is not None
                }
                orphaned_sources = (
                    str(source["source_id"])
                    for source in value["sources"]
                    if str(source["source_id"]) not in referenced_sources
                )
                for source_id in orphaned_sources:
                    if source_id not in releases:
                        releases.append(source_id)
            for source_id in release_source_ids:
                if source_id not in releases:
                    releases.append(source_id)
            temporary = path.with_suffix(
                path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            temporary.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def _record_client(self, row: dict[str, Any]) -> None:
        path = self.config.reuse_client_ledger_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "pid": os.getpid(),
            "model_instance_nonce": self._instance_nonce,
            "reuse_arm": self.config.reuse_arm,
            "session_index": self._session_index,
            **row,
        }
        with path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)

    def _new_session_if_needed(self, messages: list[dict[str, Any]]) -> None:
        if len(messages) <= 2 or len(messages) < self._last_message_count:
            release = [
                str(source["source_id"])
                for source in getattr(self, "_pending_sources", {}).values()
            ]
            if self._pending_source is not None:
                release.append(str(self._pending_source["source_id"]))
            if (
                release
                and self.config.reuse_arm not in DENSE_REUSE_ARMS
            ):
                self._atomic_sidecar_update(release_source_ids=release)
            self._pending_source = None
            self._pending_sources = {}
            self._commit_phase_latched = False
            self._session_index += 1
            self._request_index = 0
        self._last_message_count = len(messages)

    def _target_case(
        self,
        target_ids: list[int],
        selected_groups: list[list[dict[str, Any]]] | None = None,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        pending = self._pending_source
        self._last_target_evidence_guard = {
            "applied": False,
            "target_evidence_valid": None,
            "reason": "not_v45_or_no_pending_source",
        }
        if pending is None:
            return None, []
        if self.config.reuse_arm == "coding_versioned_evidence_guard_v45":
            self._last_target_evidence_guard = versioned_evidence_target_guard(
                pending,
                selected_groups or [],
                allow_symbol_disjoint=False,
            )
            if not self._last_target_evidence_guard[
                "target_evidence_valid"
            ]:
                self._record_client(
                    {
                        "event": "pending_source_version_invalidated",
                        "source_id": pending["source_id"],
                        "evidence_guard": self._last_target_evidence_guard,
                    }
                )
                return None, [str(pending["source_id"])]
        positions = find_sublist(target_ids, pending["segment_ids"])
        if len(positions) != 1:
            self._record_client(
                {
                    "event": "pending_source_not_reusable",
                    "source_id": pending["source_id"],
                    "matches": len(positions),
                }
            )
            return None, [str(pending["source_id"])]
        target_start = positions[0]
        length = len(pending["segment_ids"])
        if target_start <= 0 or target_start + length >= len(target_ids):
            return None, [str(pending["source_id"])]
        case_id = (
            f"{self._instance_nonce}-s{self._session_index}-"
            f"q{self._request_index}-"
            f"{self.config.reuse_arm}"
        )
        return {
            "case_id": case_id,
            "source_id": pending["source_id"],
            "content_hash": pending["content_hash"],
            "length": length,
            "policy_label": self.config.reuse_arm,
            "segment_token_hash": pending["segment_token_hash"],
            "source_prefix_token_hash": pending["source_prefix_token_hash"],
            "source_prompt_hash": pending["source_prompt_hash"],
            "source_start": pending["source_start"],
            "target_prefix_token_hash": token_ids_hash(
                target_ids[:target_start]
            ),
            "target_prompt_hash": token_ids_hash(target_ids),
            "target_start": target_start,
            "target_uses": 1,
        }, []

    def _group_token_span(
        self,
        prompt_ids: list[int],
        group: list[dict[str, Any]],
    ) -> tuple[int, int] | None:
        literal = "".join(
            self._render_message_literal(message) for message in group
        )
        group_ids = self._tokenizer.encode(
            literal, add_special_tokens=False
        ).ids
        positions = find_sublist(prompt_ids, group_ids)
        if len(positions) != 1:
            return None
        return positions[0], positions[0] + len(group_ids)

    @staticmethod
    def _v46_pool_key(row: dict[str, Any]) -> str:
        return ":".join(
            (
                str(row["source_group_sha256"]),
                ",".join(row["source_paths"]),
                str(row["segment_token_hash"]),
            )
        )

    def _v46_target_cases(
        self,
        *,
        prompt_ids: list[int],
        selected_groups: list[list[dict[str, Any]]],
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        pool = dict(getattr(self, "_pending_sources", {}))
        target_rows: list[dict[str, Any]] = []
        releases: list[str] = []
        guards: list[dict[str, Any]] = []
        for key, handle in list(pool.items()):
            if self.config.reuse_arm in DEPENDENCY_GRAPH_ARMS:
                base_guard = coding_dependency_graph_target_guard(
                    handle, selected_groups
                )
            elif self.config.reuse_arm == "coding_dependency_cold_cost":
                base_guard = coding_dependency_target_guard(
                    handle, selected_groups
                )
            else:
                base_guard = observed_path_target_guard(
                    handle, selected_groups
                )
            guard = {
                "pool_key": key,
                "source_id": handle["source_id"],
                **base_guard,
            }
            # Keep the same object in telemetry so later span/cost admission
            # updates are visible to the caller.
            guards.append(guard)
            if not guard["target_evidence_valid"]:
                releases.append(str(handle["source_id"]))
                pool.pop(key, None)
                continue
            group = selected_groups[int(guard["source_group_index"])]
            group_span = self._group_token_span(prompt_ids, group)
            if group_span is None:
                releases.append(str(handle["source_id"]))
                pool.pop(key, None)
                guard.update(
                    target_evidence_valid=False,
                    reason="target_group_token_span_not_unique",
                )
                continue
            positions = find_sublist(prompt_ids, handle["segment_ids"])
            left, right = group_span
            inside = [
                start
                for start in positions
                if start >= left
                and start + len(handle["segment_ids"]) <= right
            ]
            if len(inside) != 1:
                releases.append(str(handle["source_id"]))
                pool.pop(key, None)
                guard.update(
                    target_evidence_valid=False,
                    reason="target_segment_not_unique_inside_group",
                )
                continue
            target_start = inside[0]
            if (
                target_start <= 0
                or target_start + len(handle["segment_ids"])
                >= len(prompt_ids)
            ):
                releases.append(str(handle["source_id"]))
                pool.pop(key, None)
                guard.update(
                    target_evidence_valid=False,
                    reason="target_segment_not_strictly_middle",
                )
                continue
            cost = (
                dependency_graph_lcb_cost_estimate(
                    island_tokens=len(handle["segment_ids"]),
                    target_prompt_tokens=len(prompt_ids),
                )
                if self.config.reuse_arm
                == "coding_dependency_graph_cold_lcb"
                else dependency_graph_mean_cost_estimate(
                    island_tokens=len(handle["segment_ids"]),
                    target_prompt_tokens=len(prompt_ids),
                )
                if self.config.reuse_arm in (
                    "coding_dependency_graph_cold_mean",
                    "coding_search_file_section_mean",
                )
                else natural_code_reuse_cost_estimate(
                    island_tokens=len(handle["segment_ids"]),
                    target_prompt_tokens=len(prompt_ids),
                )
            )
            if self.config.reuse_arm in NATURAL_CODE_COST_ARMS:
                guard.update(
                    reuse_admitted=cost["reuse_admitted"],
                    admission_reason=(
                        "lower_bound_cache_ready_saving_positive"
                        if cost["reuse_admitted"]
                        and self.config.reuse_arm
                        == "coding_dependency_graph_cold_lcb"
                        else "predicted_cache_ready_saving_positive"
                        if cost["reuse_admitted"]
                        else "lower_bound_cache_ready_saving_nonpositive"
                        if self.config.reuse_arm
                        == "coding_dependency_graph_cold_lcb"
                        else "predicted_cache_ready_saving_nonpositive"
                    ),
                    cost_estimate=cost,
                )
                if not cost["reuse_admitted"]:
                    # Keep a valid source in the small pool: a later, longer
                    # prompt can cross the measured break-even point.
                    continue
            target_rows.append(
                {**handle, "target_start": target_start, "cost_estimate": cost}
            )

        selected: list[dict[str, Any]] = []
        intervals: list[tuple[int, int]] = []
        for row in sorted(
            target_rows,
            key=lambda value: (
                float(
                    value["cost_estimate"].get(
                        "lower_bound_cache_ready_saving_ms",
                        value["cost_estimate"][
                            "predicted_cache_ready_saving_ms"
                        ],
                    )
                ),
                value["source_request_index"],
            ),
            reverse=True,
        ):
            start = int(row["target_start"])
            end = start + len(row["segment_ids"])
            if any(start < right and left < end for left, right in intervals):
                continue
            selected.append(row)
            intervals.append((start, end))
            max_target_islands = (
                1
                if self.config.reuse_arm in DEPENDENCY_GRAPH_ARMS
                else 3
            )
            if len(selected) >= max_target_islands:
                break
        selected.sort(key=lambda row: row["target_start"])

        target_prompt_hash = token_ids_hash(prompt_ids)
        target_group_id = (
            f"{self._instance_nonce}-s{self._session_index}-"
            f"q{self._request_index}-v46-{target_prompt_hash[:12]}"
        )
        cases = []
        for island_index, pending in enumerate(selected):
            target_start = int(pending["target_start"])
            cases.append(
                {
                    "case_id": f"{target_group_id}-i{island_index}",
                    "target_group_id": target_group_id,
                    "source_id": pending["source_id"],
                    "content_hash": pending["content_hash"],
                    "length": len(pending["segment_ids"]),
                    "policy_label": self.config.reuse_arm,
                    "segment_token_hash": pending["segment_token_hash"],
                    "source_prefix_token_hash": pending[
                        "source_prefix_token_hash"
                    ],
                    "source_prompt_hash": pending["source_prompt_hash"],
                    "source_start": pending["source_start"],
                    "target_prefix_token_hash": token_ids_hash(
                        prompt_ids[:target_start]
                    ),
                    "target_prompt_hash": target_prompt_hash,
                    "target_start": target_start,
                    "target_uses": 1,
                    **(
                        {"cost_estimate": pending["cost_estimate"]}
                        if self.config.reuse_arm in NATURAL_CODE_COST_ARMS
                        else {}
                    ),
                }
            )
        return cases, list(dict.fromkeys(releases)), guards, pool

    def _v46_future_sources(
        self,
        *,
        prompt_ids: list[int],
        selected_groups: list[list[dict[str, Any]]],
        pool: dict[str, dict[str, Any]],
        protected_source_ids: set[str] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        list[str],
        dict[str, Any],
    ]:
        retained = (
            selected_groups
            if len(selected_groups) < self.config.rolling_history_groups
            else selected_groups[1:]
        )
        if self.config.reuse_arm == "coding_search_file_section_mean":
            candidates, decision = search_file_section_dependency_cold_candidates(
                retained
            )
        elif self.config.reuse_arm in DEPENDENCY_GRAPH_ARMS:
            candidates, decision = (
                dependency_graph_cold_repository_code_candidates(retained)
            )
        elif self.config.reuse_arm == "coding_dependency_cold_cost":
            candidates, decision = cold_natural_repository_code_candidates(retained)
        elif self.config.reuse_arm == "coding_natural_code_cost":
            candidates, decision = natural_repository_code_candidates(retained)
        else:
            candidates, decision = versioned_observed_path_candidates(retained)
        group_hashes = [coding_group_sha256(group) for group in retained]
        proposed: dict[str, dict[str, Any]] = {}
        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        source_prompt_hash = token_ids_hash(prompt_ids)
        for candidate_index, (candidate, evidence) in enumerate(
            zip(candidates, decision["candidate_evidence"], strict=True)
        ):
            group = retained[int(evidence["group_index"])]
            group_hash = str(evidence["group_sha256"])
            if group_hashes.count(group_hash) != 1:
                skip("source_group_identity_not_unique")
                continue
            group_span = self._group_token_span(prompt_ids, group)
            if group_span is None:
                skip("source_group_token_span_not_unique")
                continue
            literal = "".join(
                self._render_message_literal(message)
                for message in candidate
            )
            encoded = self._tokenizer.encode(literal, add_special_tokens=False)
            literal_ids = encoded.ids
            section_start = evidence.get("candidate_char_start")
            section_end = evidence.get("candidate_char_end")
            localized_source_start: int | None = None
            if section_start is not None and section_end is not None:
                literal_positions = find_sublist(prompt_ids, literal_ids)
                left, right = group_span
                literal_inside = [
                    start
                    for start in literal_positions
                    if start >= left and start + len(literal_ids) <= right
                ]
                offsets = list(getattr(encoded, "offsets", ()))
                if len(literal_inside) != 1 or len(offsets) != len(literal_ids):
                    skip("source_section_parent_not_unique_or_offsets_absent")
                    continue
                local_indices = [
                    index
                    for index, (start, end) in enumerate(offsets)
                    if end > int(section_start) and start < int(section_end)
                ]
                if not local_indices:
                    skip("source_section_has_no_tokens")
                    continue
                local_left = local_indices[0]
                local_right = local_indices[-1] + 1
                segment_ids = literal_ids[local_left:local_right]
                localized_source_start = literal_inside[0] + local_left
            else:
                segment_ids = literal_ids
            if len(segment_ids) < self.config.reuse_min_tokens:
                skip("source_below_minimum_tokens")
                continue
            if localized_source_start is None:
                positions = find_sublist(prompt_ids, segment_ids)
                left, right = group_span
                inside = [
                    start
                    for start in positions
                    if start >= left and start + len(segment_ids) <= right
                ]
                if len(inside) != 1:
                    skip("source_segment_not_unique_inside_group")
                    continue
                localized_source_start = inside[0]
            segment_ids, source_start = capped_tail(
                segment_ids,
                localized_source_start,
                self.config.reuse_copy_cap,
            )
            if (
                source_start <= 0
                or source_start + len(segment_ids) >= len(prompt_ids)
            ):
                skip("source_segment_not_strictly_middle")
                continue
            segment_token_hash = token_ids_hash(segment_ids)
            source_id = (
                f"{self._instance_nonce}-s{self._session_index}-"
                f"q{self._request_index}-{self.config.reuse_arm}-"
                f"i{candidate_index}-{source_prompt_hash[:12]}"
            )
            content_hash = hashlib.sha256(
                (
                    self.config.reuse_arm
                    + ":"
                    + source_prompt_hash
                    + ":"
                    + segment_token_hash
                    + ":"
                    + group_hash
                ).encode()
            ).hexdigest()
            source = {
                "source_id": source_id,
                "content_hash": content_hash,
                "length": len(segment_ids),
                "persistent": True,
                "policy_label": self.config.reuse_arm,
                "segment_token_hash": segment_token_hash,
                "source_prefix_token_hash": token_ids_hash(
                    prompt_ids[:source_start]
                ),
                "source_prompt_hash": source_prompt_hash,
                "source_start": source_start,
            }
            pending = {
                **source,
                "segment_ids": segment_ids,
                "source_group_sha256": group_hash,
                "source_observation_sha256": evidence[
                    "observation_sha256"
                ],
                "source_paths": evidence["paths"],
                "source_symbols": evidence["symbols"],
                **(
                    {
                        "source_dependency_graph": evidence[
                            "dependency_graph"
                        ]
                    }
                    if "dependency_graph" in evidence
                    else {}
                ),
                "repository_scope_dependency": evidence[
                    "path_provenance"
                ]["repository_scope_dependency"],
                "source_request_index": self._request_index,
            }
            key = self._v46_pool_key(pending)
            pending["pool_key"] = key
            if key not in pool:
                proposed[key] = pending

        protected_source_ids = protected_source_ids or set()
        protected_keys = [
            key
            for key, handle in pool.items()
            if str(handle["source_id"]) in protected_source_ids
        ]
        if len(protected_keys) > 3:
            raise ValueError("target references more than three V46 sources")
        combined = {**pool, **proposed}
        ranked_unprotected = [
            key
            for key, _ in sorted(
                (
                    (key, handle)
                    for key, handle in combined.items()
                    if key not in protected_keys
                ),
                key=lambda item: (
                    len(item[1]["segment_ids"]),
                    item[1]["source_request_index"],
                ),
                reverse=True,
            )
        ]
        ranked_keys = [
            *protected_keys,
            *ranked_unprotected[: 3 - len(protected_keys)],
        ]
        keep = set(ranked_keys)
        releases = [
            str(handle["source_id"])
            for key, handle in pool.items()
            if key not in keep
        ]
        next_pool = {key: combined[key] for key in ranked_keys}
        added = [
            {
                key: value
                for key, value in proposed[pool_key].items()
                if key
                not in {
                    "pool_key",
                    "repository_scope_dependency",
                    "segment_ids",
                    "source_group_sha256",
                    "source_observation_sha256",
                    "source_paths",
                    "source_request_index",
                    "source_symbols",
                    "source_dependency_graph",
                }
            }
            for pool_key in ranked_keys
            if pool_key in proposed
        ]
        return added, next_pool, releases, {
            **decision,
            "source_registered": bool(added),
            "sources_registered": len(added),
            "pool_size": len(next_pool),
            "pool_evictions": len(releases),
            "target_protected_sources": len(protected_keys),
            "source_skip_reasons": skipped,
            "max_live_sources": 3,
            "max_target_islands": (
                1
                if self.config.reuse_arm
                in DEPENDENCY_GRAPH_ARMS
                else 3
            ),
        }

    def _future_source(
        self,
        *,
        prompt_ids: list[int],
        selected_groups: list[list[dict[str, Any]]],
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any],
    ]:
        # A future request adds one completed interaction and keeps the latest
        # rolling_history_groups groups.  Therefore selected_groups[0] will be
        # dropped.  General copies all retained history; coding-aware protects
        # the current latest group plus the future newest group.
        if (
            self.config.reuse_arm == "coding_commit_phase_dense_v38"
            and self._commit_phase_latched
        ):
            return None, None, {
                "arm": self.config.reuse_arm,
                "mode": "commit_phase_dense_latched",
                "latest_group_protected": True,
                "risk_reasons": ["repository_commit_phase_latched"],
                "retained_groups_after_roll": max(
                    0, len(selected_groups) - 1
                ),
                "source_registered": False,
                "skip_reason": "repository_commit_phase_latched",
            }
        if len(selected_groups) < self.config.rolling_history_groups:
            return None, None, {
                "arm": self.config.reuse_arm,
                "mode": "insufficient_rolling_history",
                "latest_group_protected": False,
                "risk_reasons": [],
                "retained_groups_after_roll": max(
                    0, len(selected_groups) - 1
                ),
            }
        grounded_encoded: tuple[list[int], list[int]] | None = None
        selected_candidate_evidence: dict[str, Any] | None = None

        def encoded_groups(
            groups: list[list[dict[str, Any]]],
        ) -> tuple[list[int], list[int]]:
            literal = "".join(
                self._render_message_literal(message)
                for group in groups
                for message in group
            )
            ids = self._tokenizer.encode(
                literal, add_special_tokens=False
            ).ids
            return ids, find_sublist(prompt_ids, ids)

        if self.config.reuse_arm in (
            "coding_grounded_observation_island_v40",
            "coding_versioned_evidence_guard_v45",
        ):
            if (
                self.config.reuse_arm
                == "coding_versioned_evidence_guard_v45"
            ):
                candidates, decision = (
                    versioned_grounded_observation_candidates(
                        selected_groups[1:]
                    )
                )
            else:
                candidates, decision = grounded_observation_candidates(
                    selected_groups[1:]
                )
            encoded_candidates = [
                (index, candidate, *encoded_groups([candidate]))
                for index, candidate in enumerate(candidates)
            ]
            eligible_candidates = [
                row
                for row in encoded_candidates
                if len(row[2]) >= self.config.reuse_min_tokens
                and len(row[3]) == 1
            ]
            if not eligible_candidates:
                return None, None, {
                    **decision,
                    "source_registered": False,
                    "skip_reason": (
                        "no_unique_version_valid_observation_at_minimum_size"
                    ),
                }
            selected = max(
                eligible_candidates,
                key=lambda row: (
                    min(len(row[2]), self.config.reuse_copy_cap),
                    decision["candidate_group_indices"][row[0]],
                ),
            )
            eligible = [selected[1]]
            grounded_encoded = (selected[2], selected[3])
            decision.update(
                selected_candidate_index=selected[0],
                selected_group_index=decision[
                    "candidate_group_indices"
                ][selected[0]],
                selected_uncapped_tokens=len(selected[2]),
            )
            if self.config.reuse_arm == (
                "coding_versioned_evidence_guard_v45"
            ):
                selected_candidate_evidence = decision[
                    "candidate_evidence"
                ][selected[0]]
                decision["selected_candidate_evidence"] = (
                    selected_candidate_evidence
                )
                if not selected_candidate_evidence["paths"]:
                    return None, None, {
                        **decision,
                        "source_registered": False,
                        "skip_reason": "selected_observation_unlocalized",
                    }
        else:
            eligible, decision = select_reuse_groups(
                self.config.reuse_arm,
                selected_groups,
                latest_group_messages=selected_groups[-1],
            )
        if decision["mode"] == "critical_event_dense_abstain":
            return None, None, {
                **decision,
                "source_registered": False,
                "skip_reason": "critical_coding_event",
            }

        segment_ids, positions = (
            grounded_encoded
            if grounded_encoded is not None
            else encoded_groups(eligible)
        )
        if self.config.reuse_arm in (
            "coding_post_mutation_payoff_guard_v28",
            "coding_post_mutation_payoff_guard_v29",
        ):
            general_groups = [list(group) for group in selected_groups[1:]]
            general_ids, general_positions = encoded_groups(general_groups)
            guard = post_mutation_payoff_guard(
                request_index=self._request_index,
                coding_candidate_tokens=len(segment_ids),
                general_candidate_tokens=len(general_ids),
                copy_cap=self.config.reuse_copy_cap,
                payoff_ratio_threshold=(
                    1.20
                    if self.config.reuse_arm
                    == "coding_post_mutation_payoff_guard_v29"
                    else 0.60
                ),
            )
            decision.update(guard)
            if guard["mode"] == "payoff_guard_dense_abstain_late_branch":
                return None, None, {
                    **decision,
                    "source_registered": False,
                    "skip_reason": "insufficient_future_target_payoff",
                }
            if guard["mode"] == "payoff_guard_general_middle_exact_prefix":
                eligible = general_groups
                segment_ids = general_ids
                positions = general_positions
                decision.update(
                    coding_protection_active=False,
                )
            else:
                decision.update(
                    coding_protection_active=True,
                )
        decision["candidate_tokens"] = len(segment_ids)
        if len(segment_ids) < self.config.reuse_min_tokens or len(positions) != 1:
            return (
                None,
                None,
                {
                    **decision,
                    "source_registered": False,
                    "skip_reason": "segment_not_unique_or_too_short",
                    "eligible_groups": len(eligible),
                    "segment_tokens": len(segment_ids),
                    "matches": len(positions),
                },
            )
        copy_cap = effective_copy_cap(
            self.config.reuse_arm,
            self.config.reuse_copy_cap,
            decision,
        )
        segment_ids, source_start = capped_tail(
            segment_ids, positions[0], copy_cap
        )
        if source_start <= 0 or source_start + len(segment_ids) >= len(prompt_ids):
            return (
                None,
                None,
                {
                    **decision,
                    "source_registered": False,
                    "skip_reason": "span_not_strictly_middle",
                },
            )
        source_prompt_hash = token_ids_hash(prompt_ids)
        segment_token_hash = token_ids_hash(segment_ids)
        source_id = (
            f"{self._instance_nonce}-s{self._session_index}-"
            f"q{self._request_index}-"
            f"{self.config.reuse_arm}-{source_prompt_hash[:12]}"
        )
        content_hash = hashlib.sha256(
            (
                self.config.reuse_arm
                + ":"
                + source_prompt_hash
                + ":"
                + segment_token_hash
            ).encode()
        ).hexdigest()
        source = {
            "source_id": source_id,
            "content_hash": content_hash,
            "length": len(segment_ids),
            "policy_label": self.config.reuse_arm,
            "segment_token_hash": segment_token_hash,
            "source_prefix_token_hash": token_ids_hash(
                prompt_ids[:source_start]
            ),
            "source_prompt_hash": source_prompt_hash,
            "source_start": source_start,
        }
        pending = {**source, "segment_ids": segment_ids}
        if selected_candidate_evidence is not None:
            pending.update(
                source_observation_sha256=selected_candidate_evidence[
                    "observation_sha256"
                ],
                source_paths=selected_candidate_evidence["paths"],
                source_symbols=selected_candidate_evidence["symbols"],
            )
        return (
            source,
            pending,
            {
                **decision,
                "source_registered": True,
                "eligible_groups": len(eligible),
                "selected_tokens": len(segment_ids),
                "effective_copy_cap": copy_cap,
            },
        )

    def _query(self, messages: list[dict[str, str]], **kwargs: Any):
        if self.config.native_backend_url:
            prompt = self._render_prompt(messages)
            prompt_ids = self._tokenizer.encode(
                prompt, add_special_tokens=False
            ).ids
            options = self.config.model_kwargs | kwargs
            max_new_tokens = int(
                options.get("max_tokens", options.get("max_new_tokens", 2048))
            )
            payload = native_generate_payload(
                backend=self.config.native_backend_name,
                session_id=f"{self._instance_nonce}-s{self._session_index}",
                request_index=self._request_index,
                prompt_text_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                input_ids=prompt_ids,
                segments=self._native_backend_segments(messages, prompt_ids),
                max_new_tokens=max_new_tokens,
                temperature=float(options.get("temperature", 0.0)),
                repetition_penalty=float(options.get("repetition_penalty", 1.0)),
            )
            started = time.perf_counter()
            response = requests.post(
                self.config.native_backend_url.rstrip("/") + "/generate",
                json=payload,
                timeout=float(options.get("timeout", 900)),
            )
            response.raise_for_status()
            value = response.json()
            if value.get("input_ids_sha256") != payload["input_ids_sha256"]:
                raise RuntimeError("native backend changed the target token IDs")
            text = str(value.get("text") or "")
            completion_ids = value.get("output_ids") or []
            model_response = litellm.ModelResponse(
                model=self.config.native_backend_name or "native-kv-backend",
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": value.get("finish_reason", "stop"),
                    }
                ],
                usage={
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(completion_ids),
                    "total_tokens": len(prompt_ids) + len(completion_ids),
                },
            )
            self._attach_embedded_tool_call(
                model_response.choices[0].message,
                (
                    f"call_{self._instance_nonce}_s{self._session_index}_"
                    f"q{self._request_index}"
                ),
                getattr(self.config, "recover_unparsed_output_with_notice", False),
            )
            self._last_stream_stats = {
                "ttft_seconds": (
                    float(value["ttft_ms"]) / 1000
                    if value.get("ttft_ms") is not None
                    else None
                ),
                "stream_elapsed_seconds": time.perf_counter() - started,
                "stream_chunks": None,
                "native_backend": self.config.native_backend_name,
                "native_backend_metrics": {
                    key: value.get(key)
                    for key in (
                        "cache_build_ms",
                        "preprocess_ms",
                        "reused_k_tokens",
                        "reused_v_tokens",
                        "recomputed_tokens",
                        "fallback_reason",
                        "physical_reuse",
                    )
                },
                "messages_sha256": hashlib.sha256(
                    json.dumps(
                        messages,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "input_ids_sha256": payload["input_ids_sha256"],
                "reusable_segments": payload["segments"],
            }
            return model_response
        started = time.perf_counter()
        options = self.config.model_kwargs | kwargs
        options.pop("stream", None)
        stream = litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL],
            stream=True,
            **options,
        )
        chunks = []
        ttft_seconds = None
        try:
            for chunk in stream:
                chunks.append(chunk)
                choices = getattr(chunk, "choices", None) or []
                if ttft_seconds is None and choices:
                    delta = choices[0].delta
                    meaningful = (
                        getattr(delta, "content", None)
                        or getattr(delta, "tool_calls", None)
                        or getattr(delta, "reasoning_content", None)
                    )
                    if meaningful:
                        ttft_seconds = time.perf_counter() - started
        finally:
            close_litellm_sync_stream(stream)
        finished = time.perf_counter()
        response = litellm.stream_chunk_builder(
            chunks,
            messages=messages,
            start_time=started,
            end_time=finished,
        )
        if response is None:
            raise RuntimeError("empty LiteLLM stream")
        message = response.choices[0].message
        self._attach_embedded_tool_call(
            message,
            (
                f"call_{self._instance_nonce}_s{self._session_index}_"
                f"q{self._request_index}"
            ),
            getattr(self.config, "recover_unparsed_output_with_notice", False),
        )
        self._last_stream_stats = {
            "ttft_seconds": ttft_seconds,
            "stream_elapsed_seconds": finished - started,
            "stream_chunks": len(chunks),
        }
        return response

    def prepare_reuse_query(
        self,
        messages: list[dict[str, Any]],
        *,
        write_sidecar: bool = True,
    ) -> dict[str, Any]:
        """Advance reuse planning without issuing the model request.

        Paired experiments use this split phase to register both treatments
        before either identical target prompt reaches the scheduler.  Ordinary
        single-arm execution still calls both phases synchronously via
        :meth:`query`.
        """

        self._new_session_if_needed(messages)
        self._request_index += 1
        rolling_messages, selected_groups, rolling = self._rolling_messages(
            messages
        )
        if self.config.reuse_arm == "coding_commit_phase_dense_v38":
            self._commit_phase_latched = (
                self._commit_phase_latched
                or any(
                    repository_commit_phase_event(group)
                    for group in selected_groups
                )
            )
        compacted_messages, compaction = self.compact_messages(rolling_messages)
        prompt_ids = self._render_prompt_ids(compacted_messages)

        target = None
        targets: list[dict[str, Any]] = []
        releases: list[str] = []
        source = None
        sources: list[dict[str, Any]] = []
        next_pending = None
        policy_decision = {
            "arm": self.config.reuse_arm,
            "mode": "dense",
            "latest_group_protected": False,
            "risk_reasons": [],
        }
        if self.config.reuse_arm in (
            "coding_observed_path_pool_v46",
            "coding_natural_code_cost",
            "coding_dependency_cold_cost",
            "coding_dependency_graph_cold_lcb",
            "coding_dependency_graph_cold_mean",
            "coding_search_file_section_mean",
        ):
            targets, releases, target_guards, live_pool = (
                self._v46_target_cases(
                    prompt_ids=prompt_ids,
                    selected_groups=selected_groups,
                )
            )
            sources, live_pool, source_releases, policy_decision = (
                self._v46_future_sources(
                    prompt_ids=prompt_ids,
                    selected_groups=selected_groups,
                    pool=live_pool,
                    protected_source_ids={
                        str(case["source_id"]) for case in targets
                    },
                )
            )
            releases = list(
                dict.fromkeys([*releases, *source_releases])
            )
            policy_decision.update(
                arm=self.config.reuse_arm,
                target_evidence_guards=target_guards,
                target_islands=len(targets),
                copied_tokens_planned=sum(
                    int(case["length"]) for case in targets
                ),
            )
            if write_sidecar:
                self._atomic_sidecar_update(
                    sources=sources,
                    cases=targets,
                    release_source_ids=releases,
                )
            self._pending_sources = live_pool
            source = sources[0] if sources else None
            target = targets[0] if targets else None
        elif self.config.reuse_arm not in DENSE_REUSE_ARMS:
            target, releases = self._target_case(
                prompt_ids,
                selected_groups=selected_groups,
            )
            target, releases, target_guard = apply_current_target_veto(
                arm=self.config.reuse_arm,
                selected_groups=selected_groups,
                target=target,
                releases=releases,
                commit_phase_latched=self._commit_phase_latched,
            )
            source, next_pending, policy_decision = self._future_source(
                prompt_ids=prompt_ids,
                selected_groups=selected_groups,
            )
            if (
                self.config.reuse_arm
                == "coding_versioned_evidence_guard_v45"
            ):
                policy_decision["target_evidence_guard"] = (
                    self._last_target_evidence_guard
                )
            if self.config.reuse_arm in TARGET_VETO_ARMS:
                dense_veto_mode = (
                    "commit_phase_target_dense_veto"
                    if self.config.reuse_arm
                    == "coding_commit_phase_dense_v38"
                    else
                    "patch_lifecycle_target_dense_veto"
                    if self.config.reuse_arm
                    == "coding_patch_lifecycle_target_v37"
                    else
                    "version_validation_target_dense_veto"
                    if self.config.reuse_arm
                    == "coding_version_validation_target_v35b"
                    else
                    "critical_current_target_dense_veto"
                    if self.config.reuse_arm
                    == "coding_critical_current_target_v34"
                    else "state_transition_target_dense_veto"
                )
                general_reuse_mode = (
                    "commit_phase_exploration_general_reuse"
                    if self.config.reuse_arm
                    == "coding_commit_phase_dense_v38"
                    else
                    "patch_lifecycle_target_general_reuse"
                    if self.config.reuse_arm
                    == "coding_patch_lifecycle_target_v37"
                    else
                    "version_validation_target_general_reuse"
                    if self.config.reuse_arm
                    == "coding_version_validation_target_v35b"
                    else
                    "critical_current_target_general_reuse"
                    if self.config.reuse_arm
                    == "coding_critical_current_target_v34"
                    else "state_transition_target_general_reuse"
                )
                if (
                    self.config.reuse_arm
                    == "coding_commit_phase_dense_v38"
                    and self._commit_phase_latched
                    and not target_guard["target_vetoed"]
                ):
                    policy_decision.update(**target_guard)
                else:
                    policy_decision.update(
                        mode=(
                            dense_veto_mode
                            if target_guard["target_vetoed"]
                            else general_reuse_mode
                        ),
                        **target_guard,
                    )
            if write_sidecar:
                self._atomic_sidecar_update(
                    sources=[source] if source else [],
                    cases=[target] if target else [],
                    release_source_ids=releases,
                )
            self._pending_source = next_pending

        return {
            "compacted_messages": compacted_messages,
            "compaction": compaction,
            "policy_decision": policy_decision,
            "prompt_ids": prompt_ids,
            "releases": releases,
            "rolling": rolling,
            "source": source,
            "sources": sources or ([source] if source else []),
            "target": target,
            "targets": targets or ([target] if target else []),
        }

    def execute_prepared_reuse_query(
        self,
        prepared: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Issue and record one request produced by ``prepare_reuse_query``."""

        compacted_messages = prepared["compacted_messages"]
        compaction = prepared["compaction"]
        policy_decision = prepared["policy_decision"]
        prompt_ids = prepared["prompt_ids"]
        rolling = prepared["rolling"]
        source = prepared["source"]
        target = prepared["target"]
        sources = prepared.get("sources") or ([source] if source else [])
        targets = prepared.get("targets") or ([target] if target else [])
        started = time.perf_counter()
        # Call the mini-SWE-agent base directly: compacting a second time would
        # change the registered prompt identity and double-increment the local
        # request counter.
        result = LitellmModel.query(self, compacted_messages, **kwargs)
        elapsed = time.perf_counter() - started
        if self.config.native_backend_url:
            executed_hash = self._last_stream_stats.get("input_ids_sha256")
            planned_hash = token_ids_hash(prompt_ids)
            if executed_hash != planned_hash:
                raise RuntimeError(
                    "common native backend prompt differs from the planned prompt: "
                    f"{executed_hash} != {planned_hash}"
                )
        original_tool_calls = result.get("tool_calls") or []
        executed_tool_calls = original_tool_calls[
            : self.config.max_tool_calls_per_response
        ]
        result["tool_calls"] = executed_tool_calls
        result.setdefault("extra", {})["context_compaction"] = {
            "request_index": self._request_index,
            **compaction,
        }
        result["extra"]["tool_call_limit"] = {
            "configured_limit": self.config.max_tool_calls_per_response,
            "original_tool_calls": len(original_tool_calls),
            "executed_tool_calls": len(executed_tool_calls),
            "discarded_tool_calls": len(original_tool_calls)
            - len(executed_tool_calls),
        }
        result["extra"]["request_latency_seconds"] = elapsed
        result.setdefault("extra", {})["rolling_history"] = rolling
        result["extra"]["reuse_treatment"] = {
            "arm": self.config.reuse_arm,
            "request_index": self._request_index,
            "prompt_tokens": len(prompt_ids),
            "target_registered": bool(targets),
            "target_islands": len(targets),
            "source_registered": bool(sources),
            "sources_registered": len(sources),
            "copied_tokens_planned": sum(
                int(case["length"]) for case in targets
            ),
            "reuse_policy_decision": policy_decision,
            "request_elapsed_seconds": elapsed,
            **self._last_stream_stats,
        }
        result["extra"]["pre_reuse_context_compaction"] = compaction
        self._record_client(
            {
                "event": "request_complete",
                **result["extra"]["reuse_treatment"],
                "rolling_history": rolling,
            }
        )
        return result

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        prepared = self.prepare_reuse_query(messages)
        return self.execute_prepared_reuse_query(prepared, **kwargs)
