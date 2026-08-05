import math

import torch

from benchmark.multi_workflow.motivate_v48_attention_kv_risk import (
    _cosine_deviation_by_head,
    _model_theta,
    _pearson,
    _ranks,
    _relative_l2_by_head,
    _spearman,
)


def test_per_head_drift_is_zero_for_identical_tensors():
    value = torch.randn(2, 7, 4)
    assert torch.allclose(
        _cosine_deviation_by_head(value, value), torch.zeros(2), atol=1e-6
    )
    assert torch.equal(_relative_l2_by_head(value, value), torch.zeros(2))


def test_rank_and_correlation_helpers_handle_ties():
    assert _ranks([1.0, 1.0, 3.0]) == [0.5, 0.5, 2.0]
    assert math.isclose(_pearson([1, 2, 3], [2, 4, 6]), 1.0)
    assert math.isclose(_spearman([3, 1, 2], [30, 10, 20]), 1.0)


def test_model_theta_supports_transformers_v5_rope_parameters():
    class Config:
        rope_parameters = {"rope_theta": 12345.0}

    assert _model_theta(Config()) == 12345.0
