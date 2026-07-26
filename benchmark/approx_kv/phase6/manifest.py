from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

from .schema import payload_sha256

REPRESENTATION_PROFILES = {
    "exact_only": {
        "resident_multiplicity": 0,
        "temporary_multiplicity": 0,
        "representation_kinds": (),
    },
    "r0_like": {
        "resident_multiplicity": 1,
        "temporary_multiplicity": 1,
        "representation_kinds": ("canonical_base",),
    },
    "r1_like_k32": {
        "resident_multiplicity": 1,
        "temporary_multiplicity": 2,
        "representation_kinds": ("canonical_base", "repair_state"),
    },
    "r2_like": {
        "resident_multiplicity": 2,
        "temporary_multiplicity": 2,
        "representation_kinds": ("canonical_base", "precomputed_adapter"),
    },
    "r4_like": {
        "resident_multiplicity": 5,
        "temporary_multiplicity": 1,
        "representation_kinds": (
            "canonical_base",
            "anchor",
            "delta",
            "anchor",
            "delta",
        ),
    },
}


class LogicalObjectRole(str, Enum):
    ARCHITECT = "architect"
    CODER = "coder"
    DEBUGGER = "debugger"
    LIVE_FILLER = "live_filler"
    DEAD_FILLER = "dead_filler"


@dataclass(frozen=True)
class FixedLogicalObject:
    object_id: str
    role: LogicalObjectRole
    logical_tokens: int
    active: bool
    retired: bool
    order: int
    dense_cost_ms: float
    recovery_cost_ms: float
    token_ids_sha256: str
    crosses_chunk_boundary: bool
    segment_count: int

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if self.logical_tokens <= 0:
            raise ValueError("logical_tokens must be positive")
        if self.active and self.retired:
            raise ValueError("an object cannot be active and retired")


def fixed_object_token_ids(order: int, logical_tokens: int) -> list[int]:
    return [
        1_000 + ((order * 7_919 + offset * 37) % 30_000)
        for offset in range(logical_tokens)
    ]


def _fixed_object(
    *,
    object_id: str,
    role: LogicalObjectRole,
    logical_tokens: int,
    active: bool,
    retired: bool,
    order: int,
    dense_cost_ms: float,
    recovery_cost_ms: float,
    chunked_prefill_size: int,
    segment_tokens_max: int,
) -> FixedLogicalObject:
    return FixedLogicalObject(
        object_id=object_id,
        role=role,
        logical_tokens=logical_tokens,
        active=active,
        retired=retired,
        order=order,
        dense_cost_ms=dense_cost_ms,
        recovery_cost_ms=recovery_cost_ms,
        token_ids_sha256=payload_sha256(fixed_object_token_ids(order, logical_tokens)),
        crosses_chunk_boundary=logical_tokens > chunked_prefill_size,
        segment_count=math.ceil(logical_tokens / segment_tokens_max),
    )


def build_fixed40_manifest(
    *,
    chunked_prefill_size: int = 1024,
    chunk_source: str = "provisional_worst_case",
) -> dict:
    if chunked_prefill_size <= 0:
        raise ValueError("chunked_prefill_size must be positive")
    if chunk_source not in {"cl2", "provisional_worst_case"}:
        raise ValueError("invalid chunk_source")
    segment_tokens_max = 512
    roles = (
        LogicalObjectRole.ARCHITECT,
        LogicalObjectRole.CODER,
        LogicalObjectRole.DEBUGGER,
        LogicalObjectRole.CODER,
        LogicalObjectRole.DEBUGGER,
    )
    objects: list[FixedLogicalObject] = []
    for index, role in enumerate(roles):
        objects.append(
            _fixed_object(
                object_id=f"workflow-{index:02d}",
                role=role,
                logical_tokens=1024 if index % 2 == 0 else 2048,
                active=True,
                retired=False,
                order=index,
                dense_cost_ms=24.0 if index % 2 else 12.0,
                recovery_cost_ms=2.0,
                chunked_prefill_size=chunked_prefill_size,
                segment_tokens_max=segment_tokens_max,
            )
        )
    for index in range(35):
        retired = index < 12
        objects.append(
            _fixed_object(
                object_id=f"filler-{index:02d}",
                role=(
                    LogicalObjectRole.DEAD_FILLER
                    if retired
                    else LogicalObjectRole.LIVE_FILLER
                ),
                logical_tokens=384 if index % 2 == 0 else 512,
                active=not retired,
                retired=retired,
                order=len(objects),
                dense_cost_ms=3.0 if index % 2 == 0 else 4.0,
                recovery_cost_ms=1.5,
                chunked_prefill_size=chunked_prefill_size,
                segment_tokens_max=segment_tokens_max,
            )
        )
    payload = {
        "schema_version": 2,
        "object_count": len(objects),
        "objects": [asdict(item) for item in objects],
        "workflow_sequence": [
            "architect",
            "coder",
            "debugger",
            "coder",
            "debugger",
        ],
        "body_tokens": 2048,
        "header_tokens": 64,
        "segment_tokens_max": segment_tokens_max,
        "chunked_prefill_size": chunked_prefill_size,
        "chunk_source": chunk_source,
    }
    payload["manifest_sha256"] = payload_sha256(payload)
    return payload
