"""Pure policy helpers for ProbeHead StateSensitivityKV V12.

The module deliberately contains no model loading and no result-directory
discovery.  A reference executor supplies observed head-KV deviations; these
helpers turn them into deterministic copy/dense decisions and matched
controls.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from benchmark.multi_workflow.sessiongraph_v11 import CostModel


PROFILE = "probehead-statesensitivitykv-v12"
HEAD_CANDIDATES = (8, 16, 32, 64)
MAX_COPY_ISLANDS = 4
SHUFFLE_SEED = 1729
JS_LIMIT = 1e-3
MIN_PROMPT_COPY_FRACTION = 0.15
MIN_HARM_REDUCTION = 0.30
MAX_PROBE_P95_MS = 2.0
BOOTSTRAP_ITERATIONS = 10_000


@dataclass(frozen=True)
class ProbeCandidate:
    session_id: str
    turn_id: int
    module_id: str
    source_start: int
    target_start: int
    length: int
    prompt_tokens: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.module_id:
            raise ValueError("probe candidate identity must be non-empty")
        if self.turn_id < 1:
            raise ValueError("probe candidates must belong to a later turn")
        if min(self.source_start, self.target_start) < 0 or self.length <= 0:
            raise ValueError("probe candidate spans must be positive")
        if self.target_start + self.length > self.prompt_tokens:
            raise ValueError("probe candidate exceeds the target prompt")


@dataclass(frozen=True)
class ProbeDecision:
    candidate: ProbeCandidate
    head_tokens: int
    probe_score: float
    copied_tokens: int
    island_index: int | None
    reason: str

    @property
    def accepted(self) -> bool:
        return self.copied_tokens > 0


def probe_score(k_deviation: float, v_deviation: float) -> float:
    values = (float(k_deviation), float(v_deviation))
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("probe deviations must be finite and non-negative")
    return max(values)


def decide_probe_candidates(
    *,
    candidates: Sequence[ProbeCandidate],
    scores: Mapping[str, float],
    head_tokens: int,
    threshold: float,
    cost_model: CostModel,
    probe_compare_us: float = 0.0,
    max_islands: int = MAX_COPY_ISLANDS,
) -> tuple[ProbeDecision, ...]:
    """Apply a frozen ProbeHead threshold in target-token order.

    The planner is intentionally online-compatible.  Removing a module can
    split an existing V11 island, so a later accepted module starts a new
    island unless it is immediately adjacent to the previously accepted one.
    """
    if head_tokens <= 0 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid probe configuration")
    if probe_compare_us < 0 or max_islands <= 0:
        raise ValueError("invalid probe cost or island bound")

    ordered = sorted(candidates, key=lambda value: value.target_start)
    occupied_end = -1
    previous_accepted_end: int | None = None
    islands = 0
    decisions: list[ProbeDecision] = []
    for candidate in ordered:
        if candidate.target_start < occupied_end:
            raise ValueError("probe candidates overlap")
        occupied_end = candidate.target_start + candidate.length
        if candidate.module_id not in scores:
            raise ValueError(f"missing probe score for {candidate.module_id}")
        score = float(scores[candidate.module_id])
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"invalid probe score for {candidate.module_id}")

        head = min(head_tokens, candidate.length)
        body = candidate.length - head
        adjacent = (
            previous_accepted_end is not None
            and previous_accepted_end == candidate.target_start
        )
        new_island = not adjacent
        reason = "probe_copy"
        island_index: int | None = islands - 1 if adjacent else islands
        if body <= 0:
            reason = "probe_consumes_complete_module"
        elif score > threshold:
            reason = "probe_score_above_threshold"
        elif new_island and islands >= max_islands:
            reason = "probe_island_limit"
        elif (
            cost_model.net_saving_us(body, islands=int(new_island))
            - probe_compare_us
            <= 0
        ):
            reason = "probe_cost_negative"

        if reason == "probe_copy":
            if new_island:
                islands += 1
                island_index = islands - 1
            previous_accepted_end = candidate.target_start + candidate.length
            copied = body
            decision_head = head
        else:
            previous_accepted_end = None
            copied = 0
            island_index = None
            decision_head = candidate.length
        decisions.append(
            ProbeDecision(
                candidate=candidate,
                head_tokens=decision_head,
                probe_score=score,
                copied_tokens=copied,
                island_index=island_index,
                reason=reason,
            )
        )
    return tuple(decisions)


def copied_fraction(decisions: Sequence[ProbeDecision]) -> float:
    prompts = {decision.candidate.prompt_tokens for decision in decisions}
    if not decisions or len(prompts) != 1:
        return 0.0
    return sum(decision.copied_tokens for decision in decisions) / next(
        iter(prompts)
    )


def shuffled_exact_budget(
    *,
    candidates: Sequence[ProbeCandidate],
    copied_token_budget: int,
    head_tokens: int,
    scores: Mapping[str, float] | None = None,
    seed: int = SHUFFLE_SEED,
) -> tuple[ProbeDecision, ...]:
    """Build a deterministic module-level control with exact copied tokens.

    The final selected module may receive a larger head so the copied body
    exactly matches the policy budget.
    """
    if copied_token_budget < 0 or head_tokens <= 0:
        raise ValueError("invalid matched-control budget")
    ordered = sorted(candidates, key=lambda value: value.target_start)
    capacity = sum(max(0, value.length - head_tokens) for value in ordered)
    if copied_token_budget > capacity:
        raise ValueError("matched-control budget exceeds candidate capacity")
    ranked = sorted(
        ordered,
        key=lambda value: hashlib.sha256(
            f"{seed}|{value.session_id}|{value.turn_id}|{value.module_id}".encode()
        ).hexdigest(),
    )
    copied_by_id: dict[str, int] = {}
    remaining = copied_token_budget
    for candidate in ranked:
        amount = min(remaining, max(0, candidate.length - head_tokens))
        copied_by_id[candidate.module_id] = amount
        remaining -= amount
        if remaining == 0:
            break
    if remaining:
        raise AssertionError("exact matched budget was not placed")

    output = []
    island = -1
    previous_end: int | None = None
    for candidate in ordered:
        copied = copied_by_id.get(candidate.module_id, 0)
        if copied:
            if previous_end != candidate.target_start:
                island += 1
            previous_end = candidate.target_start + candidate.length
            reason = "shuffled_matched_copy"
            island_index: int | None = island
            decision_head = candidate.length - copied
        else:
            previous_end = None
            reason = "shuffled_matched_dense"
            island_index = None
            decision_head = candidate.length
        output.append(
            ProbeDecision(
                candidate=candidate,
                head_tokens=decision_head,
                probe_score=float((scores or {}).get(candidate.module_id, 0.0)),
                copied_tokens=copied,
                island_index=island_index,
                reason=reason,
            )
        )
    if sum(value.copied_tokens for value in output) != copied_token_budget:
        raise AssertionError("matched control changed the copied-token budget")
    return tuple(output)
