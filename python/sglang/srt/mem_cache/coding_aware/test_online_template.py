from __future__ import annotations

import inspect

from sglang.srt.mem_cache.coding_aware.online_admit import (
    BindAction,
    BindResult,
    SourceObservation,
)
from sglang.srt.mem_cache.coding_aware.online_template import OnlineFileTemplate
from sglang.srt.mem_cache.kvcomm.types import token_ids_hash as _hash


def _obs(content="mod", later_roles=3, **overrides) -> SourceObservation:
    tokens = tuple(ord(ch) for ch in content) + (1, 2, 3)
    values = dict(
        source_id=f"read:{content}",
        source_start=4,
        token_ids=tokens,
        content_hash=_hash(tokens),
        source_prefix_hash="pfx",
        single_file_repository_code=True,
        version_valid=True,
        later_roles_in_protocol=later_roles,
        policy_label="coding",
    )
    values.update(overrides)
    return SourceObservation(**values)


def _bind(obs: SourceObservation, action: BindAction, reason: str) -> BindResult:
    return BindResult(
        action=action,
        reason=reason,
        source_id=obs.source_id,
        content_hash=obs.content_hash,
        length=len(obs.token_ids),
    )


def test_cold_start_follows_protocol_not_a_plan():
    template = OnlineFileTemplate()
    assert template.admit(_obs(later_roles=3)) is None
    assert template.admit(_obs(later_roles=0)) == "no_protocol_reread"
    source = inspect.getsource(OnlineFileTemplate.admit)
    assert "target_start" not in source


def test_repeated_copies_unlock_later_roles_zero():
    template = OnlineFileTemplate(min_obs=2, admit_floor=0.6)
    obs = _obs(later_roles=0)
    hit = _bind(obs, BindAction.COPY, "online_bind")
    template.observe(hit)
    assert template.admit(obs) == "no_protocol_reread"
    template.observe(hit)
    assert template.admit(obs) is None
    assert template.posterior(obs.content_hash).copied == 2


def test_repeated_waste_stops_protocol_lease():
    template = OnlineFileTemplate(min_obs=2, skip_ceiling=0.30)
    obs = _obs(later_roles=3)
    miss = _bind(obs, BindAction.DENSE, "not_in_target")
    template.observe(miss)
    assert template.admit(obs) is None
    template.observe(miss)
    assert template.admit(obs) == "learned_low_reuse"
    assert template.posterior(obs.content_hash).wasted == 2


def test_modules_learn_independently():
    template = OnlineFileTemplate(min_obs=2, admit_floor=0.6, skip_ceiling=0.30)
    hot = _obs(content="hot", later_roles=0)
    cold = _obs(content="cold", later_roles=3)
    for _ in range(3):
        template.observe(_bind(hot, BindAction.COPY, "online_bind"))
        template.observe(_bind(cold, BindAction.DENSE, "not_in_target"))
    assert template.admit(hot) is None
    assert template.admit(cold) == "learned_low_reuse"


def test_prefetch_priority_rises_with_hits():
    template = OnlineFileTemplate()
    obs = _obs()
    before = template.prefetch_priority(obs.content_hash, later_roles=3)
    template.observe(_bind(obs, BindAction.COPY, "online_bind"))
    template.observe(_bind(obs, BindAction.COPY, "online_bind"))
    after = template.prefetch_priority(obs.content_hash, later_roles=3)
    assert after > before
    assert before >= 3


def test_mechanical_gates_still_fail_closed():
    template = OnlineFileTemplate()
    obs = _obs(version_valid=False, later_roles=3)
    assert template.admit(obs) == "version_invalid"
