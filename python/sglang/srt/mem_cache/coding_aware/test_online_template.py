from __future__ import annotations

import inspect

from sglang.srt.mem_cache.coding_aware.online_admit import (
    BindAction,
    BindResult,
    SourceObservation,
)
from sglang.srt.mem_cache.coding_aware.online_template import (
    OnlineFileTemplate,
    featurize,
    task_class_id,
)
from sglang.srt.mem_cache.kvcomm.types import token_ids_hash as _hash


def _obs(content="mod", later_roles=3, policy="coding", **overrides) -> SourceObservation:
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
        policy_label=policy,
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


def test_feature_is_task_class_not_file_or_issue():
    django = _obs(content="django-models")
    sphinx = _obs(content="sphinx-build")
    assert featurize(django) == featurize(sphinx) == "coding_agent"
    assert task_class_id("coding_natural_code_cost") == "coding_agent"
    assert task_class_id("general_shifted_lcs") == "general"
    assert "content_hash" not in inspect.getsource(featurize)


def test_cold_start_follows_protocol():
    template = OnlineFileTemplate()
    assert template.admit(_obs(later_roles=3)) is None
    assert template.admit(_obs(later_roles=0)) == "no_protocol_reread"


def test_copies_transfer_across_files_in_the_same_class():
    template = OnlineFileTemplate(min_obs=2, admit_floor=0.6)
    django = _obs(content="django", later_roles=3)
    sphinx = _obs(content="sphinx", later_roles=0)
    hit = _bind(django, BindAction.COPY, "online_bind")
    template.observe(hit, django)
    assert template.admit(sphinx) == "no_protocol_reread"
    template.observe(hit, django)
    assert template.admit(sphinx) is None
    assert featurize(django) == featurize(sphinx)


def test_offline_prior_survives_a_few_online_misses():
    template = OnlineFileTemplate.from_json(
        {
            "bins": {
                "coding_agent": {"alpha": 180, "beta": 20, "offline_n": 200},
            }
        }
    )
    obs = _obs(later_roles=3)
    miss = _bind(obs, BindAction.DENSE, "not_in_target")
    template.observe(miss, obs)
    template.observe(miss, obs)
    assert template.admit(obs) is None
    assert template.bin_for(obs).mean > 0.8


def test_json_roundtrip_keeps_class_bins():
    template = OnlineFileTemplate.from_json(
        {"bins": {"coding_agent": {"alpha": 10, "beta": 2, "offline_n": 11}}}
    )
    again = OnlineFileTemplate.from_json(template.to_json())
    assert again.bin_for(_obs()).alpha == 10
    assert again.bin_for(_obs()).offline_n == 11


def test_damped_finetune_does_not_overwrite_offline_class_prior():
    template = OnlineFileTemplate.from_json(
        {
            "bins": {
                "coding_agent": {"alpha": 116, "beta": 93, "offline_n": 207},
            }
        }
    )
    obs = _obs(later_roles=3)
    roles0 = _obs(content="other", later_roles=0)
    start = template.bin_for(obs).mean
    hit = _bind(obs, BindAction.COPY, "online_bind")
    for _ in range(8):
        template.observe(hit, obs)
    assert abs(template.bin_for(obs).mean - start) < 0.02
    assert template.admit(obs) is None
    assert template.admit(roles0) == "no_protocol_reread"


def _coding_agent_fixture():
    from pathlib import Path

    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "benchmark/multi_workflow/templates/coding_agent.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("coding_agent.json fixture")


def test_compiled_coding_agent_fixture_is_a_class_prior():
    template = OnlineFileTemplate.from_path(_coding_agent_fixture())
    obs = _obs(later_roles=3)
    assert template.bin_for(obs).offline_n == 207
    assert 0.50 < template.bin_for(obs).mean < 0.62
    assert template.admit(obs) is None
    assert template.admit(_obs(later_roles=0)) == "no_protocol_reread"


def test_prefetch_priority_uses_class_mean():
    template = OnlineFileTemplate()
    obs = _obs()
    before = template.prefetch_priority(obs)
    template.observe(_bind(obs, BindAction.COPY, "online_bind"), obs)
    template.observe(_bind(obs, BindAction.COPY, "online_bind"), obs)
    assert template.prefetch_priority(obs) >= before
