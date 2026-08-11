from __future__ import annotations

import torch

from benchmark.multi_workflow.motivate_attention_kv_perturbation_bound import (
    _head_query_bound_metrics,
    _softmax_island_l1_bound,
    _softmax_l1_bound,
)


def test_zero_perturbation_has_zero_error() -> None:
    generator = torch.Generator().manual_seed(7)
    scores = torch.randn(3, 4, 9, generator=generator)
    values = torch.randn(3, 9, 5, generator=generator)
    metrics = _head_query_bound_metrics(
        dense_scores=scores,
        stale_scores=scores.clone(),
        dense_values=values,
        stale_values=values.clone(),
        island_start=2,
        island_end=7,
    )
    assert torch.equal(
        metrics["actual_kv_output_delta"],
        torch.zeros_like(metrics["actual_kv_output_delta"]),
    )
    assert torch.equal(
        metrics["analytic_bound"],
        torch.zeros_like(metrics["analytic_bound"]),
    )


def test_finite_attention_bound_covers_random_island_perturbations() -> None:
    generator = torch.Generator().manual_seed(20260806)
    for _ in range(20):
        dense_scores = torch.randn(4, 6, 17, generator=generator)
        dense_values = torch.randn(4, 17, 8, generator=generator)
        stale_scores = dense_scores.clone()
        stale_values = dense_values.clone()
        stale_scores[..., 4:12] += 0.4 * torch.randn(
            4, 6, 8, generator=generator
        )
        stale_values[:, 4:12] += 0.3 * torch.randn(
            4, 8, 8, generator=generator
        )
        metrics = _head_query_bound_metrics(
            dense_scores=dense_scores,
            stale_scores=stale_scores,
            dense_values=dense_values,
            stale_values=stale_values,
            island_start=4,
            island_end=12,
        )
        assert torch.all(
            metrics["actual_kv_output_delta"]
            <= metrics["exact_finite_bound"] + 1e-5
        )
        assert torch.all(
            metrics["actual_kv_output_delta"]
            <= metrics["analytic_bound"] + 1e-5
        )
        assert torch.all(
            metrics["actual_kv_output_delta"]
            <= metrics["mass_aware_analytic_bound"] + 1e-5
        )


def test_softmax_l1_bound_is_zero_and_below_two() -> None:
    epsilon = torch.tensor([0.0, 0.1, 1.0, 10.0])
    bound = _softmax_l1_bound(epsilon)
    assert bound[0] == 0
    assert torch.all(bound >= 0)
    assert torch.all(bound <= 2)
    assert torch.all(bound[1:] > bound[:-1])


def test_mass_aware_softmax_bound_covers_island_only_changes() -> None:
    generator = torch.Generator().manual_seed(41)
    for _ in range(40):
        dense_scores = torch.randn(3, 5, 19, generator=generator)
        stale_scores = dense_scores.clone()
        stale_scores[..., 7:12] += torch.randn(3, 5, 5, generator=generator)
        dense = torch.softmax(dense_scores, dim=-1)
        stale = torch.softmax(stale_scores, dim=-1)
        epsilon = (stale_scores[..., 7:12] - dense_scores[..., 7:12]).abs().amax(-1)
        mass = dense[..., 7:12].sum(-1)
        assert torch.all(
            (stale - dense).abs().sum(-1)
            <= _softmax_island_l1_bound(epsilon, mass) + 1e-6
        )
