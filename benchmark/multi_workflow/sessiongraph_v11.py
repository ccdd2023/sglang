"""Self-contained FileVersion SessionGraphKV V11 policy helpers.

This module is deliberately model-free and result-directory-free.  It accepts
only online-visible replay metadata plus canonical raw-tool provenance and
emits the integer ``head_tokens`` schema consumed by the KVCOMM adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE = "fileversion-sessiongraphkv-v11"
SHUFFLE_SEED = 1729
MAX_COPY_ISLANDS = 4
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
FORBIDDEN_ONLINE_KEYS = frozenset(
    {
        "patch",
        "gold",
        "gold_patch",
        "canonical_solution",
        "test",
        "test_patch",
        "test_outcome",
        "test_outcomes",
        "hidden",
        "hidden_tests",
        "expected_edit",
        "expected_replacement",
    }
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def assert_online_safe(value: object) -> None:
    """Reject sealed answer/test fields recursively on the online path."""
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        bad = keys & FORBIDDEN_ONLINE_KEYS
        bad.update(
            key
            for key in keys
            if key.startswith(("gold_", "hidden_", "expected_"))
        )
        if bad:
            raise ValueError(f"sealed fields in online object: {sorted(bad)}")
        for child in value.values():
            assert_online_safe(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_online_safe(child)


def token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def prompt_hash(token_ids: Sequence[int]) -> str:
    """Canonical, platform-independent prompt-token hash."""
    payload = b"".join(int(value).to_bytes(8, "little", signed=True) for value in token_ids)
    return hashlib.sha256(payload).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def event_index(module_id: str) -> int:
    if not module_id.startswith("event:"):
        return -1
    try:
        return int(module_id.split(":", 1)[1])
    except ValueError:
        return -1


def paths_overlap(left: str, right: str) -> bool:
    left = left.strip("/")
    right = right.strip("/")
    return left == right or left.endswith("/" + right) or right.endswith("/" + left)


def graph_distances(turn: Mapping[str, Any]) -> dict[str, int]:
    dependencies = {
        str(module["module_id"]): tuple(
            str(value) for value in module.get("dependencies", ())
        )
        for module in turn["modules"]
    }
    targets = [
        str(module["module_id"])
        for module in turn["modules"]
        if module["module_type"] == "target"
    ]
    distances = {target: 0 for target in targets}
    queue = deque(targets)
    while queue:
        child = queue.popleft()
        for parent in dependencies.get(child, ()):
            if parent not in distances:
                distances[parent] = distances[child] + 1
                queue.append(parent)
    return distances


def exact_prefix_module_ids(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> set[str]:
    output: set[str] = set()
    for left, right in zip(previous["modules"], current["modules"]):
        if (
            left["module_id"] != right["module_id"]
            or left["content_hash"] != right["content_hash"]
        ):
            break
        output.add(str(right["module_id"]))
    return output


@dataclass(frozen=True)
class CostModel:
    dense_us_per_token: float
    copy_us_per_token: float
    rope_us_per_token: float
    island_fixed_us: float
    cpu_lookup_us: float
    safety_margin_us: float

    def net_saving_us(self, tokens: int, islands: int = 1) -> float:
        if tokens <= 0:
            return -math.inf
        return (
            self.dense_us_per_token * tokens
            - (self.copy_us_per_token + self.rope_us_per_token) * tokens
            - self.island_fixed_us * islands
            - self.cpu_lookup_us
            - self.safety_margin_us
        )


def _islands(module_ids: Iterable[str], positions: Mapping[str, int]) -> list[list[str]]:
    ordered = sorted(set(module_ids), key=positions.__getitem__)
    output: list[list[str]] = []
    for module_id in ordered:
        if not output or positions[module_id] != positions[output[-1][-1]] + 1:
            output.append([module_id])
        else:
            output[-1].append(module_id)
    return output


def canonical_mutations(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, tuple[str, ...] | None]]:
    output: dict[str, dict[str, tuple[str, ...] | None]] = defaultdict(dict)
    for row in rows:
        classification = str(row["classification"])
        if classification == "global_fail_closed":
            paths: tuple[str, ...] | None = None
        elif classification == "resolved_python":
            paths = tuple(
                str(value)
                for value in (*row.get("changed_paths", ()), *row.get("removed_paths", ()))
            )
        else:
            paths = ()
        output[str(row["session_id"])][str(row["event_id"])] = paths
    return dict(output)


def file_version_stable(
    module: Mapping[str, Any],
    resources: Sequence[str],
    mutations: Mapping[str, tuple[str, ...] | None],
) -> tuple[bool, str]:
    if not resources:
        return False, "source_path_unresolved"
    for event_id, changed in mutations.items():
        if event_index(event_id) <= event_index(str(module["module_id"])):
            continue
        if changed is None:
            return False, "later_edit_path_unresolved"
        if any(paths_overlap(view, edit) for view in resources for edit in changed):
            return False, "viewed_file_edited_later"
    return True, "file_version_stable"


def select_fileversion_modules(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    runtime_exact_ids: set[str],
    source_view_resources: Mapping[str, Sequence[str]],
    mutations: Mapping[str, tuple[str, ...] | None],
    cost_model: CostModel,
    max_islands: int = MAX_COPY_ISLANDS,
) -> tuple[list[str], dict[str, str]]:
    """Apply the registered V11 guards and return cost-positive copy modules."""
    assert_online_safe(previous)
    assert_online_safe(current)
    if previous["session_id"] != current["session_id"]:
        raise ValueError("cross-session reuse is forbidden")
    previous_by_id = {str(row["module_id"]): row for row in previous["modules"]}
    prefix = exact_prefix_module_ids(previous, current)
    distances = graph_distances(current)
    current_observation = next(
        (
            str(module["module_id"])
            for module in reversed(current["modules"])
            if module["module_type"]
            in {"tool_output", "test_output", "source_view", "workspace_edit"}
        ),
        None,
    )
    candidates: set[str] = set()
    reasons: dict[str, str] = {}
    positions = {
        str(module["module_id"]): int(module["position"])
        for module in current["modules"]
    }
    by_id = {str(module["module_id"]): module for module in current["modules"]}
    for module_id, module in by_id.items():
        old = previous_by_id.get(module_id)
        distance = distances.get(module_id)
        reason = ""
        candidate_reason = "candidate"
        if module_id in prefix:
            reason = "exact_prefix_baseline"
        elif module["cache_scope"] == "turn":
            reason = "turn_local_dense"
        elif module_id == current_observation:
            reason = "current_observation_dense"
        elif distance is not None and distance < 2:
            reason = "graph_distance_lt_2"
        elif old is None or old["content_hash"] != module["content_hash"]:
            reason = "not_exact"
        elif module_id not in runtime_exact_ids:
            reason = "token_slice_mismatch"
        elif int(module["first_seen_turn"]) >= int(current["turn_id"]):
            reason = "not_seen_before"
        elif int(module["token_span"][1]) - int(module["token_span"][0]) <= 4:
            reason = "runtime_ineligible"
        elif (
            module["cache_scope"] == "workspace"
            and int(module["workspace_version"]) != int(current["workspace_version"])
        ):
            if module["module_type"] != "source_view":
                reason = f"stale_{module['module_type']}"
            else:
                stable, reason = file_version_stable(
                    module, source_view_resources.get(module_id, ()), mutations
                )
                if stable:
                    reason = ""
                    candidate_reason = "file_version_stable"
        tokens = int(module["token_span"][1]) - int(module["token_span"][0])
        if not reason and cost_model.net_saving_us(tokens) <= 0:
            reason = "cost_negative"
        if reason:
            reasons[module_id] = reason
        else:
            candidates.add(module_id)
            reasons[module_id] = candidate_reason
    islands = _islands(candidates, positions)
    ranked = sorted(
        islands,
        key=lambda island: (
            -cost_model.net_saving_us(
                sum(
                    int(by_id[value]["token_span"][1])
                    - int(by_id[value]["token_span"][0])
                    for value in island
                )
            ),
            positions[island[0]],
        ),
    )
    selected = {
        module_id for island in ranked[:max_islands] for module_id in island
    }
    return sorted(selected, key=positions.__getitem__), reasons


def build_label_rows(
    *,
    case_id: str,
    chunks: Sequence[Mapping[str, Any]],
    copied_module_ids: Iterable[str],
) -> dict[str, list[dict[str, Any]]]:
    """Build V11 labels plus exact-budget/exact-island matched controls."""
    ordered = sorted(chunks, key=lambda row: int(row["position"]))
    copied = set(copied_module_ids)
    policy = []
    for chunk in ordered:
        length = int(chunk["chunk_len"])
        module_id = str(chunk["module_id"])
        row = {
            "case_id": case_id,
            "slot_id": str(chunk["slot_id"]),
            "chunk_signature": str(chunk["chunk_signature"]),
            "chunk_len": length,
            "token_hash": str(chunk["token_hash"]),
            "head_tokens": 0 if module_id in copied else length,
            "risk_bucket": (
                "file_version_copy" if module_id in copied else "dense_guard"
            ),
            "graph_distance": None,
            "policy_profile": PROFILE,
        }
        policy.append(row)

    targets = sorted(_copied_island_lengths(policy), reverse=True)

    def control(mode: str) -> list[dict[str, Any]]:
        starts = list(range(len(ordered)))
        if mode == "uniform":
            starts.sort(key=lambda index: (abs((index + 0.5) / len(ordered) - 0.5), index))
        elif mode == "shuffled":
            random.Random(SHUFFLE_SEED).shuffle(starts)
        else:
            starts.sort(
                key=lambda index: (
                    str(ordered[index]["module_type"]),
                    int(ordered[index]["position"]),
                )
            )
        candidates: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
        for target in set(targets):
            options = []
            for start in starts:
                indices, copies, total = [], [], 0
                for index in range(start, len(ordered)):
                    amount = min(target - total, int(ordered[index]["chunk_len"]))
                    indices.append(index)
                    copies.append(amount)
                    total += amount
                    if total == target:
                        options.append((tuple(indices), tuple(copies)))
                        break
            candidates[target] = options
        assignments: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def place(offset: int, occupied: set[int]) -> bool:
            if offset == len(targets):
                return True
            blocked = occupied | {value - 1 for value in occupied} | {
                value + 1 for value in occupied
            }
            for indices, copies in candidates[targets[offset]]:
                if any(index in blocked for index in indices):
                    continue
                assignments.append((indices, copies))
                if place(offset + 1, occupied | set(indices)):
                    return True
                assignments.pop()
            return False

        if targets and not place(0, set()):
            raise ValueError(f"cannot exactly place copy islands {targets} for {mode}")
        copy_by_index = [0] * len(ordered)
        for indices, copies in assignments:
            for index, amount in zip(indices, copies, strict=True):
                copy_by_index[index] = amount
        profile = f"{mode}-matched-fileversion-v11"
        rows = [
            {
                **{key: row[key] for key in (
                    "case_id", "slot_id", "chunk_signature", "chunk_len",
                    "token_hash", "graph_distance",
                )},
                "head_tokens": int(row["chunk_len"]) - copy_by_index[index],
                "risk_bucket": f"{mode}_matched",
                "policy_profile": profile,
            }
            for index, row in enumerate(policy)
        ]
        if _copied_island_lengths(rows) != _copied_island_lengths(policy):
            raise AssertionError("matched control changed copy-island lengths")
        return rows

    return {
        "fileversion": policy,
        "uniform": control("uniform"),
        "shuffled": control("shuffled"),
        "type_only": control("type_only"),
    }


def _copied_island_lengths(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    output: list[int] = []
    running = 0
    for row in rows:
        copied = int(row["chunk_len"]) - int(row["head_tokens"])
        if copied:
            running += copied
        elif running:
            output.append(running)
            running = 0
    if running:
        output.append(running)
    return sorted(output)
