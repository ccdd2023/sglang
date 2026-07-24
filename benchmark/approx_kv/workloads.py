from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence


class CacheObjectKind(str, Enum):
    CANONICAL_BASE = "canonical_base"
    STAGE_VARIANT = "stage_variant"
    ANCHOR = "anchor"
    REPAIR_METADATA = "repair_metadata"


class ReuseClass(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass(frozen=True)
class CacheObject:
    object_id: str
    role: str
    kind: CacheObjectKind
    reuse_class: ReuseClass
    artifact_group: str
    target_prefix_tokens: int
    reusable_prefix_token_ids: tuple[int, ...]
    payload: str
    dense_cost_weight: float
    recovery_cost_weight: float

    @property
    def reusable_prefix_tokens(self) -> int:
        return len(self.reusable_prefix_token_ids)

    def manifest(self) -> dict[str, Any]:
        prefix_bytes = ",".join(
            str(token_id) for token_id in self.reusable_prefix_token_ids
        ).encode("ascii")
        return {
            "object_id": self.object_id,
            "role": self.role,
            "kind": self.kind.value,
            "reuse_class": self.reuse_class.value,
            "artifact_group": self.artifact_group,
            "target_prefix_tokens": self.target_prefix_tokens,
            "reusable_prefix_tokens": self.reusable_prefix_tokens,
            "prefix_token_hash": hashlib.sha256(prefix_bytes).hexdigest(),
            "payload_chars": len(self.payload),
            "payload_hash": hashlib.sha256(self.payload.encode("utf-8")).hexdigest(),
            "dense_cost_weight": self.dense_cost_weight,
            "recovery_cost_weight": self.recovery_cost_weight,
        }


@dataclass(frozen=True)
class PressureSelection:
    objects: tuple[CacheObject, ...]
    active_reusable_tokens: int
    gpu_kv_capacity_tokens: int
    target_ratio: float

    @property
    def actual_ratio(self) -> float:
        if self.gpu_kv_capacity_tokens <= 0:
            raise ValueError("gpu_kv_capacity_tokens must be positive")
        return self.active_reusable_tokens / self.gpu_kv_capacity_tokens


@dataclass(frozen=True)
class TraceInvocation:
    step: int
    phase: str
    object_id: str
    role: str
    occurrence: int
    suffix: str
    next_use_step: int | None
    next_use_distance: int | None
    intervening_unique_prefix_tokens: int | None
    next_use_request_step: int | None = None


def deterministic_code(seed: str, blocks: int) -> str:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    lines: list[str] = []
    for index in range(blocks):
        digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).hexdigest()[:16]
        lines.extend(
            (
                f"def synthetic_{index}_{digest}(value):",
                f"    intermediate = value + {index}",
                f"    checksum = '{digest}'",
                "    return intermediate, checksum",
                "",
            )
        )
    return "\n".join(lines)


def build_messages(
    cache_object: CacheObject,
    suffix: str,
    *,
    cache_salt: str = "measured",
) -> list[dict[str, str]]:
    system = (
        f"cache_salt={cache_salt};"
        f"object={cache_object.object_id};"
        f"role={cache_object.role};"
        f"kind={cache_object.kind.value};"
        f"group={cache_object.artifact_group}\n"
        "Process the fixed code artifact and follow the workflow role."
    )
    user = f"{cache_object.payload}\n\n# dynamic request\n{suffix}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized prompt")
        value = value[0]
    return [int(token_id) for token_id in value]


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    return _normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    )


def common_prefix_token_ids(
    first: Sequence[int],
    second: Sequence[int],
) -> tuple[int, ...]:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return tuple(int(token_id) for token_id in first[:length])


def reusable_prefix_token_ids(
    tokenizer: Any,
    cache_object: CacheObject,
    *,
    cache_salt: str = "measured",
) -> tuple[int, ...]:
    first = tokenize_messages(
        tokenizer,
        build_messages(
            cache_object,
            "invocation=000000;sample=calibration-a",
            cache_salt=cache_salt,
        ),
    )
    second = tokenize_messages(
        tokenizer,
        build_messages(
            cache_object,
            "invocation=999999;sample=calibration-b",
            cache_salt=cache_salt,
        ),
    )
    return common_prefix_token_ids(first, second)


def _calibrate_object(
    tokenizer: Any,
    *,
    object_id: str,
    role: str,
    kind: CacheObjectKind,
    reuse_class: ReuseClass,
    artifact_group: str,
    target_prefix_tokens: int,
    dense_cost_weight: float,
    recovery_cost_weight: float,
    tolerance_tokens: int,
) -> CacheObject:
    if target_prefix_tokens <= 0:
        raise ValueError("target_prefix_tokens must be positive")

    blocks = max(96, target_prefix_tokens // 6)
    source_text = deterministic_code(object_id, blocks)
    source_ids = _normalize_token_ids(
        tokenizer.encode(source_text, add_special_tokens=False)
    )
    if len(source_ids) < target_prefix_tokens:
        raise ValueError(
            f"insufficient calibration tokens for {object_id}: "
            f"{len(source_ids)} < {target_prefix_tokens}"
        )

    def candidate(token_count: int) -> CacheObject:
        payload = tokenizer.decode(
            source_ids[:token_count],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        draft = CacheObject(
            object_id=object_id,
            role=role,
            kind=kind,
            reuse_class=reuse_class,
            artifact_group=artifact_group,
            target_prefix_tokens=target_prefix_tokens,
            reusable_prefix_token_ids=(),
            payload=payload,
            dense_cost_weight=dense_cost_weight,
            recovery_cost_weight=recovery_cost_weight,
        )
        return CacheObject(
            **{
                **draft.__dict__,
                "reusable_prefix_token_ids": reusable_prefix_token_ids(
                    tokenizer,
                    draft,
                ),
            }
        )

    low = 1
    high = len(source_ids)
    best: CacheObject | None = None
    best_source_count = 0
    while low <= high:
        middle = (low + high) // 2
        current = candidate(middle)
        if best is None or abs(
            current.reusable_prefix_tokens - target_prefix_tokens
        ) < abs(best.reusable_prefix_tokens - target_prefix_tokens):
            best = current
            best_source_count = middle
        if current.reusable_prefix_tokens < target_prefix_tokens:
            low = middle + 1
        else:
            high = middle - 1

    for source_count in range(
        max(1, best_source_count - 8),
        min(len(source_ids), best_source_count + 8) + 1,
    ):
        current = candidate(source_count)
        if best is None or abs(
            current.reusable_prefix_tokens - target_prefix_tokens
        ) < abs(best.reusable_prefix_tokens - target_prefix_tokens):
            best = current

    assert best is not None
    error = abs(best.reusable_prefix_tokens - target_prefix_tokens)
    if error > tolerance_tokens:
        raise ValueError(
            f"unable to calibrate {object_id}: target={target_prefix_tokens}, "
            f"actual={best.reusable_prefix_tokens}, tolerance={tolerance_tokens}"
        )
    return best


def build_object_catalog(
    tokenizer: Any,
    *,
    object_count: int = 24,
    target_sizes: Sequence[int] = (512, 1024, 2048, 4096),
    tolerance_tokens: int = 32,
) -> tuple[CacheObject, ...]:
    if object_count < 4:
        raise ValueError("object_count must be at least four")
    if not target_sizes or any(size <= 0 for size in target_sizes):
        raise ValueError("target_sizes must contain positive values")

    roles = (
        "architect",
        "coder",
        "debugger",
        "tester",
        "cold-filler-a",
        "cold-filler-b",
    )
    kinds = tuple(CacheObjectKind)
    reuse_classes = (
        ReuseClass.HOT,
        ReuseClass.WARM,
        ReuseClass.COLD,
        ReuseClass.WARM,
    )
    objects = []
    for index in range(object_count):
        target = int(target_sizes[index % len(target_sizes)])
        dense_cost_weight = target * (1.0 + 0.05 * (index % 5))
        recovery_ratio = 0.08 + 0.04 * (index % 6)
        objects.append(
            _calibrate_object(
                tokenizer,
                object_id=f"phase2-object-{index:02d}",
                role=roles[index % len(roles)],
                kind=kinds[index % len(kinds)],
                reuse_class=reuse_classes[index % len(reuse_classes)],
                artifact_group=f"artifact-{index // 3:02d}",
                target_prefix_tokens=target,
                dense_cost_weight=dense_cost_weight,
                recovery_cost_weight=dense_cost_weight * recovery_ratio,
                tolerance_tokens=tolerance_tokens,
            )
        )
    return tuple(objects)


def unique_prefix_token_count(objects: Iterable[CacheObject]) -> int:
    return unique_token_sequence_count(
        cache_object.reusable_prefix_token_ids for cache_object in objects
    )


def unique_token_sequence_count(
    token_sequences: Iterable[Sequence[int]],
) -> int:
    trie: dict[int, dict] = {}
    unique_tokens = 0
    for token_sequence in token_sequences:
        node = trie
        for token_id in token_sequence:
            child = node.get(token_id)
            if child is None:
                child = {}
                node[token_id] = child
                unique_tokens += 1
            node = child
    return unique_tokens


def trace_physical_token_count(
    tokenizer: Any,
    objects: Sequence[CacheObject],
    trace: Sequence[TraceInvocation],
    *,
    cache_salt: str = "measured",
    sample_kind: str = "measured",
    repeat: int = 0,
) -> int:
    object_map = {cache_object.object_id: cache_object for cache_object in objects}
    prompt_sequences = [
        tokenize_messages(
            tokenizer,
            build_messages(
                object_map[invocation.object_id],
                (
                    f"{invocation.suffix};"
                    f"sample={sample_kind};"
                    f"repeat={repeat:03d}"
                ),
                cache_salt=cache_salt,
            ),
        )
        for invocation in trace
    ]
    # Each request adds one generated token beyond the prompt path.
    return unique_token_sequence_count(prompt_sequences) + len(trace)


def select_objects_for_pressure(
    catalog: Sequence[CacheObject],
    *,
    gpu_kv_capacity_tokens: int,
    target_ratio: float,
    required_object_ids: set[str] | None = None,
) -> PressureSelection:
    if not catalog:
        raise ValueError("catalog must not be empty")
    if gpu_kv_capacity_tokens <= 0:
        raise ValueError("gpu_kv_capacity_tokens must be positive")
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")

    required_object_ids = required_object_ids or set()
    catalog_ids = {cache_object.object_id for cache_object in catalog}
    missing = required_object_ids - catalog_ids
    if missing:
        raise ValueError(f"required objects are absent from catalog: {sorted(missing)}")

    target_tokens = gpu_kv_capacity_tokens * target_ratio
    selected_list = [
        cache_object
        for cache_object in catalog
        if cache_object.object_id in required_object_ids
    ]
    remaining = [
        cache_object
        for cache_object in catalog
        if cache_object.object_id not in required_object_ids
    ]
    if not selected_list:
        selected_list.append(remaining.pop(0))

    while remaining:
        current_tokens = unique_prefix_token_count(selected_list)
        current_error = abs(current_tokens - target_tokens)
        candidate_rows = []
        for cache_object in remaining:
            trial = [*selected_list, cache_object]
            trial_tokens = unique_prefix_token_count(trial)
            candidate_rows.append(
                (
                    abs(trial_tokens - target_tokens),
                    catalog.index(cache_object),
                    cache_object,
                    trial_tokens,
                )
            )
        best_error, _, best_object, best_tokens = min(candidate_rows)
        if best_error >= current_error and current_tokens >= target_tokens:
            break
        selected_list.append(best_object)
        remaining.remove(best_object)
        if best_error >= current_error and best_tokens >= target_tokens:
            break

    selected = tuple(
        cache_object
        for cache_object in catalog
        if cache_object in selected_list
    )
    active_tokens = unique_prefix_token_count(selected)
    return PressureSelection(
        objects=selected,
        active_reusable_tokens=active_tokens,
        gpu_kv_capacity_tokens=gpu_kv_capacity_tokens,
        target_ratio=target_ratio,
    )


def build_workflow_trace(
    objects: Sequence[CacheObject],
) -> tuple[TraceInvocation, ...]:
    if not objects:
        raise ValueError("objects must not be empty")

    by_role: dict[str, list[CacheObject]] = {}
    for cache_object in objects:
        by_role.setdefault(cache_object.role, []).append(cache_object)

    def primary(role: str) -> CacheObject:
        candidates = by_role.get(role)
        if candidates:
            return candidates[0]
        return objects[0]

    phases: list[tuple[str, CacheObject]] = []
    phases.extend(("fill", cache_object) for cache_object in objects)
    phases.extend(
        (
            ("workflow", primary("architect")),
            ("workflow", primary("coder")),
            ("workflow", primary("debugger")),
        )
    )
    phases.extend(
        ("cold-filler", cache_object)
        for cache_object in objects
        if cache_object.reuse_class == ReuseClass.COLD
    )
    phases.extend(
        (
            ("retry", primary("coder")),
            ("retry", primary("debugger")),
        )
    )
    phases.extend(
        ("branch-fanout", cache_object)
        for cache_object in objects
        if cache_object.kind
        in (CacheObjectKind.STAGE_VARIANT, CacheObjectKind.ANCHOR)
    )
    replay_order = sorted(
        objects,
        key=lambda cache_object: (
            {
                ReuseClass.HOT: 0,
                ReuseClass.WARM: 1,
                ReuseClass.COLD: 2,
            }[cache_object.reuse_class],
            -cache_object.reusable_prefix_tokens,
            cache_object.object_id,
        ),
    )
    phases.extend(("replay", cache_object) for cache_object in replay_order)
    phases.extend(
        ("hot-tail", cache_object)
        for cache_object in objects
        if cache_object.reuse_class == ReuseClass.HOT
    )

    occurrences: dict[str, int] = {}
    raw = []
    for step, (phase, cache_object) in enumerate(phases):
        occurrence = occurrences.get(cache_object.object_id, 0)
        occurrences[cache_object.object_id] = occurrence + 1
        raw.append(
            {
                "step": step,
                "phase": phase,
                "object_id": cache_object.object_id,
                "role": cache_object.role,
                "occurrence": occurrence,
                "suffix": (
                    f"invocation={step:06d};"
                    f"phase={phase};"
                    f"occurrence={occurrence:03d}"
                ),
            }
        )

    next_use_by_object: dict[str, int] = {}
    result: list[TraceInvocation] = []
    object_map = {cache_object.object_id: cache_object for cache_object in objects}
    for item in reversed(raw):
        next_use_step = next_use_by_object.get(item["object_id"])
        intervening_tokens = None
        if next_use_step is not None:
            intervening_ids = {
                raw[index]["object_id"]
                for index in range(item["step"] + 1, next_use_step)
            }
            intervening_tokens = unique_prefix_token_count(
                object_map[object_id] for object_id in intervening_ids
            )
        result.append(
            TraceInvocation(
                **item,
                next_use_step=next_use_step,
                next_use_distance=(
                    None if next_use_step is None else next_use_step - item["step"]
                ),
                intervening_unique_prefix_tokens=intervening_tokens,
                next_use_request_step=next_use_step,
            )
        )
        next_use_by_object[item["object_id"]] = item["step"]
    result.reverse()
    return tuple(result)
