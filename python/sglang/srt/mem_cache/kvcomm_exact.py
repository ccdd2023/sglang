"""Local-manifest executor for auditable middle-span KV-reuse pilots.

This module is intentionally opt-in and integration-only.  Requests cannot
select spans through the HTTP API.  Versions 1 and 2 use a manifest frozen
before server startup.  Version 3 is an append-only local sidecar: an agent
process may register the next source/target identities between requests, while
the scheduler alone reloads and validates them.  Version 1 admits only exact
same-position reuse.  Versions 2 and 3 also admit shifted copy-only reuse and
apply the required RoPE delta to K while copying V unchanged.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.radix_backend import (
    DeviceSegmentMaterializer,
    RadixKVTransferBackend,
    RoPEConfig,
    TargetSlotTransaction,
)
from sglang.srt.mem_cache.kvcomm.types import (
    DenseRange,
    KVReusePlan,
    KVSegmentHandle,
    KVSegmentKey,
    KVTransferStats,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)


class RequestTokenPool(Protocol):
    req_to_token: torch.Tensor

    def write(self, indices: Any, values: torch.Tensor) -> None: ...


class ExactMiddlePhase(str, Enum):
    DENSE_PREFIX = "dense_prefix"
    COPY_READY = "copy_ready"
    DENSE_SUFFIX = "dense_suffix"
    FALLBACK_DENSE = "fallback_dense"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ExactMiddleCase:
    case_id: str
    source_prompt_hash: str
    target_prompt_hash: str
    segment_token_hash: str
    source_prefix_token_hash: str
    target_prefix_token_hash: str
    source_start: int
    target_start: int
    length: int
    content_hash: str
    allow_shifted_copy: bool = False
    allow_target_prefix_bypass: bool = False
    policy_label: str = "general"
    target_uses: int | None = None
    source_id: str | None = None
    ordinary_prefix_reuse: bool | None = None
    reuse_enabled: bool = True

    def __post_init__(self) -> None:
        hashes = (
            self.source_prompt_hash,
            self.target_prompt_hash,
            self.segment_token_hash,
            self.source_prefix_token_hash,
            self.target_prefix_token_hash,
            self.content_hash,
        )
        if not self.case_id or any(not value for value in hashes):
            raise ValueError("exact-middle case identity is incomplete")
        if self.source_start <= 0 or self.target_start <= 0 or self.length <= 0:
            raise ValueError("middle span must have a non-empty dense prefix")
        shifted = (
            self.source_start != self.target_start
            or self.source_prefix_token_hash != self.target_prefix_token_hash
        )
        if shifted and not self.allow_shifted_copy:
            raise ValueError("shifted copy requires manifest version 2")
        if self.allow_target_prefix_bypass and not self.allow_shifted_copy:
            raise ValueError(
                "target-prefix bypass requires shifted-copy support"
            )
        if not self.policy_label:
            raise ValueError("policy_label must be non-empty")
        if self.target_uses is not None and self.target_uses <= 0:
            raise ValueError("target_uses must be positive when configured")

    def key(self, *, model_id: str, cache_dtype: str) -> KVSegmentKey:
        return KVSegmentKey(
            content_hash=self.content_hash,
            token_hash=self.segment_token_hash,
            token_count=self.length,
            model_id=model_id,
            cache_dtype=cache_dtype,
            kind=SegmentKind.MIDDLE,
        )


@dataclass(frozen=True)
class ExactMiddleSource:
    source_id: str
    source_prompt_hash: str
    segment_token_hash: str
    source_prefix_token_hash: str
    source_start: int
    length: int
    content_hash: str
    policy_label: str = "general"

    def __post_init__(self) -> None:
        hashes = (
            self.source_prompt_hash,
            self.segment_token_hash,
            self.source_prefix_token_hash,
            self.content_hash,
        )
        if not self.source_id or any(not value for value in hashes):
            raise ValueError("exact-middle source identity is incomplete")
        if self.source_start <= 0 or self.length <= 0:
            raise ValueError("source span must have a non-empty dense prefix")
        if not self.policy_label:
            raise ValueError("policy_label must be non-empty")

    def key(self, *, model_id: str, cache_dtype: str) -> KVSegmentKey:
        return KVSegmentKey(
            content_hash=self.content_hash,
            token_hash=self.segment_token_hash,
            token_count=self.length,
            model_id=model_id,
            cache_dtype=cache_dtype,
            kind=SegmentKind.MIDDLE,
        )


@dataclass
class ExactMiddleRequestState:
    case: ExactMiddleCase
    source: KVSegmentHandle
    lease: Any
    phase: ExactMiddlePhase = ExactMiddlePhase.DENSE_PREFIX
    fallback_reason: str | None = None
    transfer_stats: KVTransferStats | None = None
    copied_indices: torch.Tensor | None = None
    ordinary_prefix_tokens: int | None = None


class ExactMiddleCanaryController:
    """Execute one pre-registered middle island without prefetch."""

    def __init__(
        self,
        *,
        manager: KVCommManager,
        allocator: Any,
        req_to_token_pool: RequestTokenPool,
        model_id: str,
        cache_dtype: str,
        rope: RoPEConfig,
        cases: Sequence[ExactMiddleCase],
        sources: Sequence[ExactMiddleSource] = (),
        ledger_path: Path | None = None,
        lease_ttl_s: float = 300.0,
        manifest_path: Path | None = None,
        reclaim_device_tokens: Callable[[int], None] | None = None,
        host_overflow_enabled: bool = False,
        ordinary_prefix_reuse_enabled: bool = False,
        ordinary_prefix_repair_tokens: int = 0,
        ordinary_prefix_target_only: bool = False,
    ) -> None:
        if not manager.config.core_enabled:
            raise ValueError("KVCOMM core must be enabled")
        if not model_id or not cache_dtype:
            raise ValueError("model/cache identity must be non-empty")
        if lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be positive")
        if not cases and not sources and manifest_path is None:
            raise ValueError("reuse manifest contains no cases")
        self.manager = manager
        self.allocator = allocator
        self.req_to_token_pool = req_to_token_pool
        self.model_id = model_id
        self.cache_dtype = cache_dtype
        self.rope = rope
        self.ledger_path = ledger_path
        self.lease_ttl_s = lease_ttl_s
        self.reclaim_device_tokens = reclaim_device_tokens
        self.host_overflow_enabled = host_overflow_enabled
        self.ordinary_prefix_reuse_enabled = ordinary_prefix_reuse_enabled
        if ordinary_prefix_repair_tokens < 0:
            raise ValueError(
                "ordinary_prefix_repair_tokens must be non-negative"
            )
        if ordinary_prefix_repair_tokens and not ordinary_prefix_reuse_enabled:
            raise ValueError(
                "ordinary prefix repair requires ordinary prefix reuse"
            )
        if ordinary_prefix_target_only and not ordinary_prefix_reuse_enabled:
            raise ValueError(
                "target-only ordinary prefix requires ordinary prefix reuse"
            )
        self.ordinary_prefix_repair_tokens = ordinary_prefix_repair_tokens
        self.ordinary_prefix_target_only = ordinary_prefix_target_only
        self.materializer = DeviceSegmentMaterializer(
            manager=manager,
            allocator=allocator,
            model_id=model_id,
            cache_dtype=cache_dtype,
        )
        derived_sources = [
            ExactMiddleSource(
                source_id=case.source_id or case.case_id,
                source_prompt_hash=case.source_prompt_hash,
                segment_token_hash=case.segment_token_hash,
                source_prefix_token_hash=case.source_prefix_token_hash,
                source_start=case.source_start,
                length=case.length,
                content_hash=case.content_hash,
                policy_label=case.policy_label,
            )
            for case in cases
            if case.reuse_enabled
        ]
        self._sources = self._group_sources([*derived_sources, *sources])
        self._targets = self._group_targets(cases)
        self._target_case_cursor: dict[str, int] = {}
        self._remaining_target_uses = {
            case.case_id: case.target_uses
            for case in cases
            if case.target_uses is not None
        }
        self._ledger_lock = threading.Lock()
        self._manifest_lock = threading.RLock()
        self._manifest_path = manifest_path
        self._manifest_signature: tuple[int, int, int] | None = None
        self._materialized_sources: dict[str, KVSegmentHandle] = {}
        self._persistent_source_leases: dict[str, Any] = {}

    @property
    def owned_device_tokens(self) -> int:
        return self.materializer.owned_device_tokens

    @classmethod
    def from_manifest(
        cls,
        path: Path,
        *,
        manager: KVCommManager,
        allocator: Any,
        req_to_token_pool: RequestTokenPool,
        model_id: str,
        cache_dtype: str,
        reclaim_device_tokens: Callable[[int], None] | None = None,
    ) -> "ExactMiddleCanaryController":
        value = json.loads(path.read_text(encoding="utf-8"))
        version = int(value.get("version", 0))
        if version not in (1, 2, 3):
            raise ValueError("unsupported reuse manifest version")
        if value.get("model_id") != model_id:
            raise ValueError("reuse manifest model_id mismatch")
        if value.get("cache_dtype") != cache_dtype:
            raise ValueError("reuse manifest cache_dtype mismatch")
        cases = []
        for row in value.get("cases", ()):
            legacy_prefix = row.get("prefix_token_hash")
            source_prefix = row.get("source_prefix_token_hash", legacy_prefix)
            target_prefix = row.get("target_prefix_token_hash", legacy_prefix)
            cases.append(
                ExactMiddleCase(
                    case_id=str(row["case_id"]),
                    source_prompt_hash=str(row["source_prompt_hash"]),
                    target_prompt_hash=str(row["target_prompt_hash"]),
                    segment_token_hash=str(row["segment_token_hash"]),
                    source_prefix_token_hash=str(source_prefix),
                    target_prefix_token_hash=str(target_prefix),
                    source_start=int(row["source_start"]),
                    target_start=int(row["target_start"]),
                    length=int(row["length"]),
                    content_hash=str(row["content_hash"]),
                    allow_shifted_copy=version >= 2,
                    allow_target_prefix_bypass=bool(
                        row.get("allow_target_prefix_bypass", False)
                    ),
                    policy_label=str(row.get("policy_label") or "general"),
                    target_uses=(
                        int(row["target_uses"])
                        if row.get("target_uses") is not None
                        else None
                    ),
                    source_id=(
                        str(row["source_id"])
                        if row.get("source_id") is not None
                        else None
                    ),
                    ordinary_prefix_reuse=(
                        bool(row["ordinary_prefix_reuse"])
                        if row.get("ordinary_prefix_reuse") is not None
                        else None
                    ),
                    reuse_enabled=bool(row.get("reuse_enabled", True)),
                )
            )
        sources = [
            ExactMiddleSource(
                source_id=str(row["source_id"]),
                source_prompt_hash=str(row["source_prompt_hash"]),
                segment_token_hash=str(row["segment_token_hash"]),
                source_prefix_token_hash=str(row["source_prefix_token_hash"]),
                source_start=int(row["source_start"]),
                length=int(row["length"]),
                content_hash=str(row["content_hash"]),
                policy_label=str(row.get("policy_label") or "general"),
            )
            for row in value.get("sources", ())
        ]
        rope = value.get("rope") or {}
        ledger = value.get("ledger_path")
        controller = cls(
            manager=manager,
            allocator=allocator,
            req_to_token_pool=req_to_token_pool,
            model_id=model_id,
            cache_dtype=cache_dtype,
            rope=RoPEConfig(
                rotary_dim=int(rope["rotary_dim"]),
                base=float(rope["base"]),
                is_neox_style=bool(rope["is_neox_style"]),
            ),
            cases=cases,
            sources=sources,
            ledger_path=Path(ledger) if ledger else None,
            lease_ttl_s=float(value.get("lease_ttl_s", 300.0)),
            manifest_path=path if version == 3 else None,
            reclaim_device_tokens=reclaim_device_tokens,
            host_overflow_enabled=bool(
                value.get("host_overflow_enabled", False)
            ),
            ordinary_prefix_reuse_enabled=bool(
                value.get("ordinary_prefix_reuse_enabled", False)
            ),
            ordinary_prefix_repair_tokens=int(
                value.get("ordinary_prefix_repair_tokens", 0)
            ),
            ordinary_prefix_target_only=bool(
                value.get("ordinary_prefix_target_only", False)
            ),
        )
        if version == 3:
            stat = path.stat()
            controller._manifest_signature = (
                stat.st_mtime_ns,
                stat.st_size,
                stat.st_ino,
            )
        return controller

    @staticmethod
    def _group_targets(
        cases: Sequence[ExactMiddleCase],
    ) -> dict[str, list[ExactMiddleCase]]:
        output: dict[str, list[ExactMiddleCase]] = {}
        case_ids: set[str] = set()
        for case in cases:
            if case.case_id in case_ids:
                raise ValueError(f"duplicate reuse case_id: {case.case_id}")
            case_ids.add(case.case_id)
            output.setdefault(case.target_prompt_hash, []).append(case)
        return output

    @staticmethod
    def _group_sources(
        sources: Sequence[ExactMiddleSource],
    ) -> dict[str, list[ExactMiddleSource]]:
        output: dict[str, list[ExactMiddleSource]] = {}
        source_ids: set[str] = set()
        for source in sources:
            previous = output.get(source.source_prompt_hash, [])
            if source in previous:
                continue
            if source.source_id in source_ids:
                raise ValueError(f"duplicate reuse source_id: {source.source_id}")
            output.setdefault(source.source_prompt_hash, []).append(source)
            source_ids.add(source.source_id)
        return output

    @staticmethod
    def _source_from_row(row: Mapping[str, Any]) -> ExactMiddleSource:
        return ExactMiddleSource(
            source_id=str(row["source_id"]),
            source_prompt_hash=str(row["source_prompt_hash"]),
            segment_token_hash=str(row["segment_token_hash"]),
            source_prefix_token_hash=str(row["source_prefix_token_hash"]),
            source_start=int(row["source_start"]),
            length=int(row["length"]),
            content_hash=str(row["content_hash"]),
            policy_label=str(row.get("policy_label") or "general"),
        )

    @staticmethod
    def _case_from_row(
        row: Mapping[str, Any], *, allow_shifted_copy: bool
    ) -> ExactMiddleCase:
        legacy_prefix = row.get("prefix_token_hash")
        return ExactMiddleCase(
            case_id=str(row["case_id"]),
            source_prompt_hash=str(row["source_prompt_hash"]),
            target_prompt_hash=str(row["target_prompt_hash"]),
            segment_token_hash=str(row["segment_token_hash"]),
            source_prefix_token_hash=str(
                row.get("source_prefix_token_hash", legacy_prefix)
            ),
            target_prefix_token_hash=str(
                row.get("target_prefix_token_hash", legacy_prefix)
            ),
            source_start=int(row["source_start"]),
            target_start=int(row["target_start"]),
            length=int(row["length"]),
            content_hash=str(row["content_hash"]),
            allow_shifted_copy=allow_shifted_copy,
            allow_target_prefix_bypass=bool(
                row.get("allow_target_prefix_bypass", False)
            ),
            policy_label=str(row.get("policy_label") or "general"),
            target_uses=(
                int(row["target_uses"])
                if row.get("target_uses") is not None
                else None
            ),
            source_id=(
                str(row["source_id"])
                if row.get("source_id") is not None
                else None
            ),
            ordinary_prefix_reuse=(
                bool(row["ordinary_prefix_reuse"])
                if row.get("ordinary_prefix_reuse") is not None
                else None
            ),
            reuse_enabled=bool(row.get("reuse_enabled", True)),
        )

    def _refresh_manifest(self) -> None:
        path = self._manifest_path
        if path is None:
            return
        try:
            stat = path.stat()
        except FileNotFoundError:
            return
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if self._manifest_signature == signature:
            return
        with self._manifest_lock:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            if self._manifest_signature == signature:
                return
            value = json.loads(path.read_text(encoding="utf-8"))
            if int(value.get("version", 0)) != 3:
                raise ValueError("dynamic reuse sidecar must remain version 3")
            if value.get("model_id") != self.model_id:
                raise ValueError("dynamic reuse manifest model_id changed")
            if value.get("cache_dtype") != self.cache_dtype:
                raise ValueError("dynamic reuse manifest cache_dtype changed")
            if bool(value.get("host_overflow_enabled", False)) != (
                self.host_overflow_enabled
            ):
                raise ValueError(
                    "dynamic reuse manifest host_overflow_enabled changed"
                )
            if bool(value.get("ordinary_prefix_reuse_enabled", False)) != (
                self.ordinary_prefix_reuse_enabled
            ):
                raise ValueError(
                    "dynamic reuse manifest "
                    "ordinary_prefix_reuse_enabled changed"
                )
            if int(value.get("ordinary_prefix_repair_tokens", 0)) != (
                self.ordinary_prefix_repair_tokens
            ):
                raise ValueError(
                    "dynamic reuse manifest "
                    "ordinary_prefix_repair_tokens changed"
                )
            if bool(value.get("ordinary_prefix_target_only", False)) != (
                self.ordinary_prefix_target_only
            ):
                raise ValueError(
                    "dynamic reuse manifest "
                    "ordinary_prefix_target_only changed"
                )

            added_sources = 0
            for row in value.get("sources", ()):
                source = self._source_from_row(row)
                previous = self._sources.get(source.source_prompt_hash, [])
                if source in previous:
                    continue
                if any(
                    old.source_id == source.source_id
                    for sources in self._sources.values()
                    for old in sources
                ):
                    raise ValueError("dynamic source_id was reused")
                self._sources.setdefault(
                    source.source_prompt_hash, []
                ).append(source)
                added_sources += 1

            added_targets = 0
            existing_case_ids = {
                case.case_id
                for cases in self._targets.values()
                for case in cases
            }
            for row in value.get("cases", ()):
                case = self._case_from_row(row, allow_shifted_copy=True)
                previous = self._targets.get(case.target_prompt_hash, [])
                if case in previous:
                    continue
                if case.case_id in existing_case_ids:
                    raise ValueError("dynamic case_id was reused")
                self._targets.setdefault(
                    case.target_prompt_hash, []
                ).append(case)
                existing_case_ids.add(case.case_id)
                if case.target_uses is not None:
                    self._remaining_target_uses[case.case_id] = case.target_uses
                added_targets += 1

            released_sources = 0
            for source_id in value.get("release_source_ids", ()):
                persistent_lease = self._persistent_source_leases.pop(
                    str(source_id), None
                )
                if persistent_lease is not None:
                    self.manager.store.unpin(persistent_lease)
                handle = self._materialized_sources.pop(str(source_id), None)
                if handle is not None and self.manager.store.release(handle):
                    released_sources += 1
            self._manifest_signature = signature
            self._record(
                {
                    "event": "manifest_reloaded",
                    "added_sources": added_sources,
                    "added_targets": added_targets,
                    "released_sources": released_sources,
                }
            )

    @staticmethod
    def _prompt_tokens(req: Any) -> tuple[int, ...]:
        return tuple(int(value) for value in req.origin_input_ids)

    def maybe_materialize_source(self, req: Any) -> KVSegmentHandle | None:
        self._refresh_manifest()
        tokens = self._prompt_tokens(req)
        sources = self._sources.get(token_ids_hash(tokens))
        if not sources:
            return None
        handles = [
            handle
            for source in sources
            if (handle := self._materialize_source(req, tokens, source))
            is not None
        ]
        return handles[0] if handles else None

    def _materialize_source(
        self,
        req: Any,
        tokens: tuple[int, ...],
        source: ExactMiddleSource,
    ) -> KVSegmentHandle | None:
        existing = self._materialized_sources.get(source.source_id)
        if existing is not None:
            return existing
        end = source.source_start + source.length
        if end >= len(tokens):
            raise ValueError("source span is not strictly middle")
        if int(req.kv_committed_len) < end:
            raise ValueError("source KV span is not committed")
        if token_ids_hash(tokens[: source.source_start]) != (
            source.source_prefix_token_hash
        ):
            raise ValueError("source prefix differs from manifest")
        segment = tokens[source.source_start:end]
        if token_ids_hash(segment) != source.segment_token_hash:
            raise ValueError("source segment differs from manifest")
        source_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, source.source_start:end
        ].to(dtype=torch.int64, copy=True)
        started = time.perf_counter()
        flush_deferred_frees = getattr(
            self.allocator, "flush_free_group", None
        )
        if flush_deferred_frees is not None:
            flush_deferred_frees()
        if self.reclaim_device_tokens is not None:
            self.reclaim_device_tokens(len(segment))
        try:
            handle = self.materializer.materialize(
                token_ids=segment,
                source_indices=source_indices,
                source_start=source.source_start,
                content_hash=source.content_hash,
            )
        except MemoryError:
            available = getattr(self.allocator, "available_size", None)
            available_tokens = (
                int(available()) if available is not None else None
            )
            if self.host_overflow_enabled:
                self._record(
                    {
                        "source_id": source.source_id,
                        "event": "source_device_capacity_overflow",
                        "policy_label": source.policy_label,
                        "requested_tokens": source.length,
                        "available_tokens": available_tokens,
                        "owned_device_tokens": self.owned_device_tokens,
                    }
                )
                try:
                    handle = self.materializer.materialize_host(
                        token_ids=segment,
                        source_indices=source_indices,
                        source_start=source.source_start,
                        content_hash=source.content_hash,
                    )
                except (MemoryError, RuntimeError) as error:
                    self._record(
                        {
                            "source_id": source.source_id,
                            "event": "source_materialization_skipped",
                            "policy_label": source.policy_label,
                            "reason": "host_materialization_failed",
                            "error_type": type(error).__name__,
                            "requested_tokens": source.length,
                            "available_tokens": available_tokens,
                            "owned_device_tokens": self.owned_device_tokens,
                        }
                    )
                    return None
                self._materialized_sources[source.source_id] = handle
                self._pin_for_repeated_targets(source, handle)
                self._record(
                    {
                        "source_id": source.source_id,
                        "event": "source_materialized_host",
                        "materialize_ms": (
                            time.perf_counter() - started
                        )
                        * 1000,
                        "policy_label": source.policy_label,
                        "tokens": source.length,
                        **self._lifecycle_counts(),
                    }
                )
                return handle
            self._record(
                {
                    "source_id": source.source_id,
                    "event": "source_materialization_skipped",
                    "policy_label": source.policy_label,
                    "reason": "allocator_capacity",
                    "requested_tokens": source.length,
                    "available_tokens": available_tokens,
                    "owned_device_tokens": self.owned_device_tokens,
                }
            )
            return None
        self._materialized_sources[source.source_id] = handle
        self._pin_for_repeated_targets(source, handle)
        self._record(
            {
                "source_id": source.source_id,
                "event": "source_materialized",
                "materialize_ms": (time.perf_counter() - started) * 1000,
                "policy_label": source.policy_label,
                "tokens": source.length,
                **self._lifecycle_counts(),
            }
        )
        return handle

    def _pin_for_repeated_targets(
        self,
        source: ExactMiddleSource,
        handle: KVSegmentHandle,
    ) -> None:
        repeated = any(
            (case.source_id or case.case_id) == source.source_id
            and (case.target_uses or 0) > 1
            for cases in self._targets.values()
            for case in cases
        )
        if repeated and source.source_id not in self._persistent_source_leases:
            self._persistent_source_leases[source.source_id] = (
                self.manager.store.pin(handle, ttl_s=self.lease_ttl_s)
            )

    def _lifecycle_counts(self) -> dict[str, int]:
        return {
            "store_records": self.manager.store.record_count,
            "store_leases": self.manager.store.lease_count,
            "materialized_sources": len(self._materialized_sources),
            "persistent_source_leases": len(
                self._persistent_source_leases
            ),
        }

    def _next_target_case(
        self,
        prompt_hash: str,
    ) -> tuple[int, ExactMiddleCase] | None:
        cases = self._targets.get(prompt_hash)
        if not cases:
            return None
        cursor = self._target_case_cursor.get(prompt_hash, 0)
        while cursor < len(cases):
            case = cases[cursor]
            remaining = self._remaining_target_uses.get(case.case_id)
            if remaining is None or remaining > 0:
                self._target_case_cursor[prompt_hash] = cursor
                return cursor, case
            cursor += 1
        self._target_case_cursor[prompt_hash] = cursor
        return None

    def _consume_target_without_state(
        self,
        *,
        prompt_hash: str,
        cursor: int,
        case: ExactMiddleCase,
    ) -> None:
        remaining = self._remaining_target_uses.get(case.case_id)
        if remaining is not None:
            self._remaining_target_uses[case.case_id] = max(
                0,
                remaining - 1,
            )
        self._target_case_cursor[prompt_hash] = cursor + 1

    def maybe_attach_target(self, req: Any) -> ExactMiddleRequestState | None:
        self._refresh_manifest()
        if getattr(req, "kvcomm_exact_dispatch_complete", False):
            return None
        existing = getattr(req, "kvcomm_exact_state", None)
        if existing is not None:
            return existing
        tokens = self._prompt_tokens(req)
        prompt_hash = token_ids_hash(tokens)
        selected = self._next_target_case(prompt_hash)
        if selected is None:
            return None
        cursor, case = selected
        end = case.target_start + case.length
        if end >= len(tokens):
            raise ValueError("target span is not strictly middle")
        if token_ids_hash(tokens[: case.target_start]) != (
            case.target_prefix_token_hash
        ):
            raise ValueError("target prefix differs from manifest")
        if token_ids_hash(tokens[case.target_start:end]) != case.segment_token_hash:
            raise ValueError("target segment differs from manifest")
        if not case.reuse_enabled:
            req.kvcomm_exact_dense_control = True
            req.kvcomm_exact_dispatch_complete = True
            self._consume_target_without_state(
                prompt_hash=prompt_hash,
                cursor=cursor,
                case=case,
            )
            self._record(
                {
                    "case_id": case.case_id,
                    "event": "target_dense_control",
                    "policy_label": case.policy_label,
                    **self._lifecycle_counts(),
                }
            )
            return None
        handle = self.manager.store.lookup(
            case.key(model_id=self.model_id, cache_dtype=self.cache_dtype)
        )
        if handle is None:
            req.kvcomm_exact_dispatch_complete = True
            self._consume_target_without_state(
                prompt_hash=prompt_hash,
                cursor=cursor,
                case=case,
            )
            self._record(
                {
                    "case_id": case.case_id,
                    "event": "target_fallback",
                    "policy_label": case.policy_label,
                    "reason": "missing_source",
                    **self._lifecycle_counts(),
                }
            )
            return None
        lease = self.manager.store.pin(handle, ttl_s=self.lease_ttl_s)
        state = ExactMiddleRequestState(case=case, source=handle, lease=lease)
        req.kvcomm_exact_state = state
        cases = self._targets[prompt_hash]
        if cursor < len(cases) - 1:
            self._target_case_cursor[prompt_hash] = cursor + 1
        return state

    def is_target_request(self, req: Any) -> bool:
        self._refresh_manifest()
        if getattr(req, "kvcomm_exact_dispatch_complete", False):
            return False
        return (
            self._next_target_case(
                token_ids_hash(self._prompt_tokens(req))
            )
            is not None
        )

    def ordinary_prefix_match_limit(self, req: Any) -> int | None:
        """Return the safe Radix prefix boundary for a registered target.

        ``None`` means that ordinary Radix matching may inspect the complete
        prompt.  A non-negative value is an exclusive token boundary.  Exact
        middle targets stop at ``target_start`` so a prefix hit can never
        consume the shifted span that the exact controller must copy.
        """

        if not self.ordinary_prefix_reuse_enabled:
            return 0
        self._refresh_manifest()
        if getattr(req, "kvcomm_exact_dispatch_complete", False):
            return 0 if self.ordinary_prefix_target_only else None
        state = getattr(req, "kvcomm_exact_state", None)
        prompt_hash = token_ids_hash(self._prompt_tokens(req))
        cases = self._targets.get(prompt_hash)
        if not cases:
            return 0 if self.ordinary_prefix_target_only else None
        if state is not None:
            case = state.case
        else:
            selected = self._next_target_case(prompt_hash)
            if selected is None:
                return 0 if self.ordinary_prefix_target_only else None
            _, case = selected
        if case.ordinary_prefix_reuse is False:
            return 0
        if case.allow_target_prefix_bypass:
            boundary = case.target_start + case.length
        else:
            boundary = case.target_start
        return max(0, boundary - self.ordinary_prefix_repair_tokens)

    def is_source_request(self, req: Any) -> bool:
        self._refresh_manifest()
        return token_ids_hash(self._prompt_tokens(req)) in self._sources

    def stage_prefix_length(self, req: Any) -> int | None:
        state = self.maybe_attach_target(req)
        if state is None or state.phase != ExactMiddlePhase.DENSE_PREFIX:
            return None
        prefix_len = len(req.prefix_indices)
        if state.ordinary_prefix_tokens is None:
            # This is the prefix present on the first scheduler staging pass,
            # before exact-middle dense chunks are advanced.  It is the
            # auditable ordinary Radix hit, not the eventual dense-prefix
            # length at COPY_READY.
            state.ordinary_prefix_tokens = prefix_len
            self._record(
                {
                    "case_id": state.case.case_id,
                    "event": "target_ordinary_prefix_matched",
                    "ordinary_prefix_tokens": prefix_len,
                    "policy_label": state.case.policy_label,
                }
            )
        case = state.case
        copy_end = case.target_start + case.length
        if case.allow_target_prefix_bypass and prefix_len >= copy_end:
            state.phase = ExactMiddlePhase.DENSE_SUFFIX
            self._record(
                {
                    "case_id": case.case_id,
                    "event": "target_prefix_bypass",
                    "ordinary_prefix_tokens": prefix_len,
                    "policy_label": case.policy_label,
                    "source_copy_tokens_avoided": case.length,
                }
            )
            return None
        if (
            prefix_len > case.target_start
            and not case.allow_target_prefix_bypass
        ):
            self._fallback(req, "prefix_hit_crosses_middle")
            return None
        if prefix_len >= case.target_start:
            state.phase = ExactMiddlePhase.COPY_READY
            return 0
        return case.target_start - prefix_len

    def copy_ready(self, req: Any) -> bool:
        state = getattr(req, "kvcomm_exact_state", None)
        return (
            state is not None
            and state.phase
            in (ExactMiddlePhase.DENSE_PREFIX, ExactMiddlePhase.COPY_READY)
            and (
                len(req.prefix_indices) == state.case.target_start
                or (
                    state.case.allow_target_prefix_bypass
                    and state.case.target_start
                    <= len(req.prefix_indices)
                    < state.case.target_start + state.case.length
                )
            )
        )

    def copy_into_request(self, req: Any) -> KVTransferStats | None:
        state = getattr(req, "kvcomm_exact_state", None)
        if state is None or not self.copy_ready(req):
            return None
        if req.req_pool_idx is None:
            self._fallback(req, "missing_request_pool_slot")
            return None
        case = state.case
        suffix_start = case.target_start + case.length
        copy_start = (
            len(req.prefix_indices)
            if case.allow_target_prefix_bypass
            else case.target_start
        )
        source_offset = copy_start - case.target_start
        copy_length = suffix_start - copy_start
        if copy_length <= 0:
            self._fallback(req, "empty_dynamic_copy")
            return None
        dense = []
        if copy_start:
            dense.append(DenseRange(0, copy_start, "dense_prefix"))
        if suffix_start < len(req.origin_input_ids):
            dense.append(
                DenseRange(
                    suffix_start,
                    len(req.origin_input_ids) - suffix_start,
                    "dense_suffix",
                )
            )
        plan = KVReusePlan(
            target_token_ids=self._prompt_tokens(req),
            copied_spans=(
                TransferSpan(
                    source=state.source,
                    source_offset=source_offset,
                    target_start=copy_start,
                    length=copy_length,
                    rope_delta=case.target_start - case.source_start,
                    chunk_start=copy_start,
                    chunk_length=copy_length,
                ),
            ),
            dense_ranges=tuple(dense),
            require_full_coverage=True,
        )
        started = time.perf_counter()
        dense_calls: list[tuple[int, int, str]] = []
        try:
            if self.reclaim_device_tokens is not None:
                self.reclaim_device_tokens(copy_length)
            with TargetSlotTransaction(self.allocator, copy_length) as transaction:
                target_indices = transaction.indices

                def resolve_target(start: int, length: int) -> torch.Tensor:
                    if start != copy_start or length != copy_length:
                        raise ValueError("unexpected target span")
                    return target_indices

                backend = RadixKVTransferBackend(
                    allocator=self.allocator,
                    target_indices=resolve_target,
                    dense_prefill=lambda start, length, reason: dense_calls.append(
                        (start, length, reason)
                    ),
                    rope=self.rope,
                )
                stats = self.manager.execute(plan, backend)
                if (
                    stats.copied_k_tokens != copy_length
                    or not stats.mechanically_valid
                ):
                    reason = (
                        stats.fallback_reasons[0]
                        if stats.fallback_reasons
                        else "mechanical_validation_failed"
                    )
                    self._fallback(req, reason)
                    return stats
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(copy_start, suffix_start)),
                    target_indices.to(torch.int32),
                )
                committed = transaction.commit()
        except MemoryError:
            self._fallback(req, "target_allocation_capacity")
            return None
        except Exception:
            self._fallback(req, "copy_exception")
            raise

        req.prefix_indices = torch.cat(
            (req.prefix_indices, committed.to(req.prefix_indices.device))
        )
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        state.phase = ExactMiddlePhase.DENSE_SUFFIX
        state.transfer_stats = stats
        state.copied_indices = committed
        self._record(
            {
                "case_id": case.case_id,
                "copy_ms": (time.perf_counter() - started) * 1000,
                "copied_k_tokens": stats.copied_k_tokens,
                "copied_v_tokens": stats.copied_v_tokens,
                "event": "target_copied",
                "fallback_reasons": stats.fallback_reasons,
                "policy_label": case.policy_label,
                "recomputed_tokens": stats.recomputed_tokens,
                "ordinary_prefix_tokens": state.ordinary_prefix_tokens or 0,
                "effective_dense_tokens": max(
                    0,
                    stats.recomputed_tokens
                    - (state.ordinary_prefix_tokens or 0),
                ),
                "rope_delta": case.target_start - case.source_start,
                "rotated_k_tokens": stats.rotated_k_tokens,
                "source_offset": source_offset,
                "source_residency": state.source.residency.value,
                "target_copy_start": copy_start,
            }
        )
        return stats

    def finish_request(self, req: Any) -> None:
        state = getattr(req, "kvcomm_exact_state", None)
        if state is None or state.phase == ExactMiddlePhase.COMPLETE:
            return
        self.manager.store.unpin(state.lease)
        state.phase = ExactMiddlePhase.COMPLETE
        remaining_uses = self._remaining_target_uses.get(state.case.case_id)
        source_released = False
        if remaining_uses is not None:
            remaining_uses -= 1
            self._remaining_target_uses[state.case.case_id] = remaining_uses
            if remaining_uses == 0:
                source_id = state.case.source_id or state.case.case_id
                persistent_lease = self._persistent_source_leases.pop(
                    source_id, None
                )
                if persistent_lease is not None:
                    self.manager.store.unpin(persistent_lease)
                source_released = self.manager.store.release(state.source)
                if source_released:
                    self._materialized_sources.pop(source_id, None)
        self._record(
            {
                "case_id": state.case.case_id,
                "event": "target_complete",
                "fallback_reason": state.fallback_reason,
                "policy_label": state.case.policy_label,
                "remaining_target_uses": remaining_uses,
                "source_released": source_released,
                **self._lifecycle_counts(),
            }
        )

    def _fallback(self, req: Any, reason: str) -> None:
        state = getattr(req, "kvcomm_exact_state", None)
        if state is None:
            return
        state.phase = ExactMiddlePhase.FALLBACK_DENSE
        state.fallback_reason = reason
        self._record(
            {
                "case_id": state.case.case_id,
                "event": "target_fallback",
                "policy_label": state.case.policy_label,
                "reason": reason,
            }
        )

    def _record(self, row: Mapping[str, Any]) -> None:
        if self.ledger_path is None:
            return
        value = {
            "cache_dtype": self.cache_dtype,
            "model_id": self.model_id,
            **dict(row),
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_lock, self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
