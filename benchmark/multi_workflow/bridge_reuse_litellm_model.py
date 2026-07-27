"""mini-SWE-agent model wrapper for native rolling-history KV reuse.

The wrapper gives Dense, general reuse, and coding-aware reuse the same
deterministic six-interaction history window.  Reuse selection is written to a
local version-3 sidecar before each request; no HTTP field may select KV spans.
The preceding real agent request is materialized as the source, so there is no
synthetic prefetch or replay request.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import itertools
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import litellm
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from pydantic import Field, model_validator

from benchmark.multi_workflow.coding_reuse_policy import (
    effective_copy_cap,
    post_mutation_payoff_guard,
    select_failure_memory_groups,
    select_reuse_groups,
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


def token_ids_hash(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(
            int(token_id).to_bytes(8, byteorder="little", signed=True)
        )
    return digest.hexdigest()


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
    ] = "dense"
    rolling_history_groups: int = Field(default=6, ge=4)
    reuse_copy_cap: int = Field(default=4096, ge=128)
    reuse_min_tokens: int = Field(default=128, ge=32)
    reuse_manifest_path: Path | None = None
    reuse_client_ledger_path: Path | None = None

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
        self._last_stream_stats: dict[str, Any] = {}
        if self.config.reuse_arm not in DENSE_REUSE_ARMS:
            # Workers are frozen to one.  A new wrapper therefore marks a
            # benchmark-instance boundary: sources left without any target by
            # the preceding instance can no longer have a future consumer.
            self._atomic_sidecar_update(release_orphaned_sources=True)

    def _render_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        prompt = self._chat_template.render(
            messages=self._template_messages(messages),
            tools=[BASH_TOOL],
            add_generation_prompt=True,
        )
        return self._tokenizer.encode(
            prompt, add_special_tokens=False
        ).ids

    @staticmethod
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
            release = []
            if self._pending_source is not None:
                release.append(str(self._pending_source["source_id"]))
            if (
                release
                and self.config.reuse_arm not in DENSE_REUSE_ARMS
            ):
                self._atomic_sidecar_update(release_source_ids=release)
            self._pending_source = None
            self._session_index += 1
            self._request_index = 0
        self._last_message_count = len(messages)

    def _target_case(
        self, target_ids: list[int]
    ) -> tuple[dict[str, Any] | None, list[str]]:
        pending = self._pending_source
        if pending is None:
            return None, []
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
        eligible, decision = select_reuse_groups(
            self.config.reuse_arm,
            selected_groups,
            latest_group_messages=selected_groups[-1],
        )

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

        segment_ids, positions = encoded_groups(eligible)
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
        if not message.tool_calls and isinstance(message.content, str):
            match = re.search(
                r"<tool_call>\s*<function=(?P<name>[^>]+)>\s*"
                r"<parameter=command>(?P<command>.*?)</parameter>\s*"
                r"</function>\s*</tool_call>",
                message.content,
                flags=re.DOTALL,
            )
            if match is not None:
                message.tool_calls = [
                    litellm.ChatCompletionMessageToolCall(
                        id=(
                            f"call_{self._instance_nonce}_"
                            f"s{self._session_index}_"
                            f"q{self._request_index}"
                        ),
                        type="function",
                        function={
                            "name": match.group("name").strip(),
                            "arguments": json.dumps(
                                {"command": match.group("command")}
                            ),
                        },
                    )
                ]
                reasoning = message.content[: match.start()].strip()
                message.content = reasoning or None
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
        compacted_messages, compaction = self.compact_messages(rolling_messages)
        prompt_ids = self._render_prompt_ids(compacted_messages)

        target = None
        releases: list[str] = []
        source = None
        next_pending = None
        policy_decision = {
            "arm": self.config.reuse_arm,
            "mode": "dense",
            "latest_group_protected": False,
            "risk_reasons": [],
        }
        if self.config.reuse_arm not in DENSE_REUSE_ARMS:
            target, releases = self._target_case(prompt_ids)
            source, next_pending, policy_decision = self._future_source(
                prompt_ids=prompt_ids,
                selected_groups=selected_groups,
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
            "target": target,
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
        started = time.perf_counter()
        # Call the mini-SWE-agent base directly: compacting a second time would
        # change the registered prompt identity and double-increment the local
        # request counter.
        result = LitellmModel.query(self, compacted_messages, **kwargs)
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
        result.setdefault("extra", {})["rolling_history"] = rolling
        result["extra"]["reuse_treatment"] = {
            "arm": self.config.reuse_arm,
            "request_index": self._request_index,
            "prompt_tokens": len(prompt_ids),
            "target_registered": target is not None,
            "source_registered": source is not None,
            "copied_tokens_planned": target["length"] if target else 0,
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
