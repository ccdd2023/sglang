from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..types import KVReusePlan, RecoveryMode
from .common import ReusableSegment
from .epic_fixed_k import build_epic_fixed_k_plan


def build_raw_rope_plan(
    *,
    target_token_ids: Sequence[int],
    segments: Sequence[ReusableSegment],
) -> KVReusePlan:
    plan = build_epic_fixed_k_plan(
        target_token_ids=target_token_ids,
        segments=segments,
        repair_tokens=0,
    )
    return replace(plan, recovery_mode=RecoveryMode.RAW_ROPE)
