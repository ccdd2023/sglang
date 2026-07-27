"""A mini-SWE-agent LiteLLM model with deterministic, audited history bounds."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions
from pydantic import Field
from tokenizers import Tokenizer


class ContextBoundedLitellmModelConfig(LitellmModelConfig):
    tokenizer_json_path: Path
    chat_template_path: Path
    prompt_token_limit: int = Field(default=28_000, gt=0)
    max_tool_observation_chars: int = Field(default=6_000, gt=0)
    max_assistant_reasoning_chars: int = Field(default=3_000, gt=0)
    emergency_message_chars: int = Field(default=1_500, gt=0)
    max_tool_calls_per_response: int = Field(default=1, gt=0)


def truncate_middle(value: str, limit: int) -> tuple[str, int]:
    if len(value) <= limit:
        return value, 0
    marker = "\n<content_compacted chars_removed={removed}/>\n"
    removed = len(value) - limit
    rendered_marker = marker.format(removed=removed)
    payload_limit = max(0, limit - len(rendered_marker))
    head = payload_limit // 2
    tail = payload_limit - head
    compacted = value[:head] + rendered_marker + (value[-tail:] if tail else "")
    return compacted, removed


class ContextBoundedLitellmModel(LitellmModel):
    """Keep complete local trajectories while bounding each remote API request."""

    def __init__(self, **kwargs: Any):
        super().__init__(config_class=ContextBoundedLitellmModelConfig, **kwargs)
        self.config: ContextBoundedLitellmModelConfig
        self._tokenizer = Tokenizer.from_file(str(self.config.tokenizer_json_path))
        self._chat_template = Template(
            self.config.chat_template_path.read_text(encoding="utf-8"),
            undefined=StrictUndefined,
        )
        self._request_index = 0

    def _template_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = []
        for source in messages:
            message = {
                key: copy.deepcopy(value)
                for key, value in source.items()
                if key != "extra"
            }
            if message.get("content") is None:
                message["content"] = ""
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        function["arguments"] = {"raw_arguments": arguments}
            prepared.append(message)
        return prepared

    def count_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        prompt = self._chat_template.render(
            messages=self._template_messages(messages),
            tools=[BASH_TOOL],
            add_generation_prompt=True,
        )
        return len(self._tokenizer.encode(prompt, add_special_tokens=False).ids)

    @staticmethod
    def _turn_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "assistant" and current:
                groups.append(current)
                current = []
            current.append(message)
        if current:
            groups.append(current)
        return groups

    def _compact_message(
        self,
        source: dict[str, Any],
        *,
        emergency: bool = False,
    ) -> tuple[dict[str, Any], int]:
        message = copy.deepcopy(source)
        role = message.get("role")
        limit = None
        if role == "tool":
            limit = (
                self.config.emergency_message_chars
                if emergency
                else self.config.max_tool_observation_chars
            )
        elif role == "assistant" and isinstance(message.get("content"), str):
            limit = (
                self.config.emergency_message_chars
                if emergency
                else self.config.max_assistant_reasoning_chars
            )
        if limit is None or not isinstance(message.get("content"), str):
            return message, 0
        message["content"], removed = truncate_middle(message["content"], limit)
        return message, removed

    def compact_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        original_tokens = self.count_prompt_tokens(messages)
        stats: dict[str, Any] = {
            "applied": False,
            "prompt_token_limit": self.config.prompt_token_limit,
            "original_prompt_tokens": original_tokens,
            "compacted_prompt_tokens": original_tokens,
            "original_messages": len(messages),
            "compacted_messages": len(messages),
            "dropped_turn_groups": 0,
            "dropped_messages": 0,
            "inserted_notice_messages": 0,
            "truncated_characters": 0,
        }
        if original_tokens <= self.config.prompt_token_limit:
            return messages, stats

        base = copy.deepcopy(messages[:2])
        groups = self._turn_groups(messages[2:])
        compact_groups: list[list[dict[str, Any]]] = []
        truncated_characters = 0
        for group in groups:
            compact_group = []
            for message in group:
                compact_message, removed = self._compact_message(message)
                compact_group.append(compact_message)
                truncated_characters += removed
            compact_groups.append(compact_group)

        selected: list[list[dict[str, Any]]] = []
        for group in reversed(compact_groups):
            candidate_groups = [group, *selected]
            dropped = len(compact_groups) - len(candidate_groups)
            notice = {
                "role": "user",
                "content": (
                    "<history_compaction "
                    f"dropped_turn_groups=\"{dropped}\">"
                    "Earlier interaction details were omitted to stay within "
                    "the hardware context budget. Repository state persists; "
                    "the most recent complete interactions follow."
                    "</history_compaction>"
                ),
            }
            candidate = [*base, notice]
            for candidate_group in candidate_groups:
                candidate.extend(candidate_group)
            if self.count_prompt_tokens(candidate) <= self.config.prompt_token_limit:
                selected = candidate_groups
            else:
                break

        if not selected and compact_groups:
            emergency_group = []
            for message in compact_groups[-1]:
                compact_message, removed = self._compact_message(
                    message, emergency=True
                )
                emergency_group.append(compact_message)
                truncated_characters += removed
            selected = [emergency_group]

        dropped_groups = len(compact_groups) - len(selected)
        notice = {
            "role": "user",
            "content": (
                "<history_compaction "
                f"dropped_turn_groups=\"{dropped_groups}\">"
                "Earlier interaction details were omitted to stay within the "
                "hardware context budget. Repository state persists; the most "
                "recent complete interactions follow."
                "</history_compaction>"
            ),
        }
        compacted = [*base, notice]
        for group in selected:
            compacted.extend(group)
        compacted_tokens = self.count_prompt_tokens(compacted)
        if compacted_tokens > self.config.prompt_token_limit:
            raise ValueError(
                f"unable to compact prompt to {self.config.prompt_token_limit}: "
                f"{compacted_tokens} tokens remain"
            )

        original_group_message_count = sum(len(group) for group in groups)
        selected_original_message_count = sum(len(group) for group in selected)
        stats.update(
            applied=True,
            compacted_prompt_tokens=compacted_tokens,
            compacted_messages=len(compacted),
            dropped_turn_groups=dropped_groups,
            dropped_messages=(
                original_group_message_count - selected_original_message_count
            ),
            inserted_notice_messages=1,
            truncated_characters=truncated_characters,
        )
        return compacted, stats

    def _parse_actions(self, response: Any) -> list[dict[str, Any]]:
        """Execute a bounded prefix while retaining the raw response for audit."""
        tool_calls = response.choices[0].message.tool_calls or []
        bounded_tool_calls = tool_calls[: self.config.max_tool_calls_per_response]
        return parse_toolcall_actions(
            bounded_tool_calls,
            format_error_template=self.config.format_error_template,
        )

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self._request_index += 1
        compacted_messages, compaction = self.compact_messages(messages)
        started = time.perf_counter()
        result = super().query(compacted_messages, **kwargs)
        elapsed = time.perf_counter() - started
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
        return result
