from benchmark.multi_workflow.probe_v14_logit_impact_kv import (
    _shared_mix,
    repair_fraction,
)

import torch


def test_component_mix_selects_only_requested_component():
    source_k = torch.zeros(1, 1, 4, 2)
    source_v = torch.zeros(1, 1, 4, 2)
    target_k = torch.ones(1, 1, 4, 2)
    target_v = torch.ones(1, 1, 4, 2)
    k, v = _shared_mix(
        variant="target_k_source_v",
        layer=0,
        source_k=source_k,
        source_v=source_v,
        target_k=target_k,
        target_v=target_v,
    )
    assert torch.equal(k, target_k)
    assert torch.equal(v, source_v)


def test_repair_fraction_accounts_for_components_layers_and_tokens():
    assert repair_fraction("target_k_source_v", 64) == 0.5
    assert repair_fraction("repair_early12", 64) == 1 / 3
    assert repair_fraction("repair_head16_tail16", 64) == 0.5
