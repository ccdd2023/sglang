from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class TraceKind(str, Enum):
    RETRY = "retry"
    CYCLE = "cycle"
    LONG_DISTANCE = "long_distance"
    TESTER = "tester"


@dataclass(frozen=True)
class StageInvocation:
    step: int
    role: str
    next_use_step: int | None
    suffix: str


@dataclass(frozen=True)
class PressurePoint:
    active_reusable_tokens: int
    gpu_kv_capacity_tokens: int

    @property
    def ratio(self) -> float:
        if self.gpu_kv_capacity_tokens <= 0:
            raise ValueError("gpu_kv_capacity_tokens must be positive")
        return self.active_reusable_tokens / self.gpu_kv_capacity_tokens


def role_sequence(kind: TraceKind) -> tuple[str, ...]:
    if kind == TraceKind.RETRY:
        return (
            "architect",
            "coder",
            "debugger",
            "coder",
            "debugger",
        )
    if kind == TraceKind.CYCLE:
        return (
            "architect",
            "coder",
            "debugger",
            "architect",
            "coder",
            "debugger",
        )
    if kind == TraceKind.LONG_DISTANCE:
        return (
            "architect",
            "coder",
            "debugger",
            "cold-filler-a",
            "cold-filler-b",
            "coder",
            "debugger",
        )
    if kind == TraceKind.TESTER:
        return (
            "architect",
            "coder",
            "debugger",
            "tester",
        )
    raise ValueError(f"unsupported trace kind: {kind}")


def build_trace(
    kind: TraceKind,
    rounds: int = 1,
) -> tuple[StageInvocation, ...]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    roles = role_sequence(kind) * rounds
    next_use = _next_use_steps(roles)
    return tuple(
        StageInvocation(
            step=step,
            role=role,
            next_use_step=next_use[step],
            suffix=f"trace={kind.value};step={step};role={role}",
        )
        for step, role in enumerate(roles)
    )


def build_interleaved_object_trace(
    *,
    kind: TraceKind,
    rounds: int,
    workflows: int,
    share_roles: bool,
) -> tuple[str, ...]:
    if workflows <= 0:
        raise ValueError("workflows must be positive")
    trace = build_trace(kind, rounds=rounds)
    objects = []
    for invocation in trace:
        for workflow_id in range(workflows):
            objects.append(
                invocation.role
                if share_roles
                else f"workflow-{workflow_id}:{invocation.role}"
            )
    return tuple(objects)


def _next_use_steps(
    roles: Sequence[str],
) -> tuple[int | None, ...]:
    result: list[int | None] = [None] * len(roles)
    next_step: dict[str, int] = {}
    for index in range(len(roles) - 1, -1, -1):
        role = roles[index]
        result[index] = next_step.get(role)
        next_step[role] = index
    return tuple(result)


def next_use_distance(invocation: StageInvocation) -> int | None:
    if invocation.next_use_step is None:
        return None
    return invocation.next_use_step - invocation.step


def deterministic_code(seed: str, blocks: int) -> str:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    lines = []
    for index in range(blocks):
        digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).hexdigest()[:16]
        lines.extend(
            (
                f"def synthetic_{index}_{digest}(value):",
                f"    intermediate = value + {index}",
                "    return intermediate",
                "",
            )
        )
    return "\n".join(lines)


def estimate_active_reusable_tokens(
    *,
    code_tokens: int,
    role_prefix_tokens: int,
    resident_variants: int,
) -> int:
    if code_tokens <= 0:
        raise ValueError("code_tokens must be positive")
    if role_prefix_tokens < 0:
        raise ValueError("role_prefix_tokens must be non-negative")
    if resident_variants <= 0:
        raise ValueError("resident_variants must be positive")
    return (code_tokens + role_prefix_tokens) * resident_variants
