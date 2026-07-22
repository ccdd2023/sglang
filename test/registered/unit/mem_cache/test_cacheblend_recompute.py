from __future__ import annotations

import unittest

import torch

from sglang.srt.mem_cache.cacheblend.recompute import (
    LayerRecomputeCoordinator,
    LayerRecomputeResult,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class RecordingLayerRecomputeBackend:
    """Fake real per-layer recompute hook: writes a deterministic marker
    into a fake KV buffer for every slot in a single batched call and
    records every call it received, so tests can assert the coordinator
    never issues a per-token loop."""

    def __init__(self, k_buffer: torch.Tensor) -> None:
        self.k_buffer = k_buffer
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def recompute_layer(self, *, layer_id, slot_indices, token_positions):
        self.calls.append((layer_id, tuple(int(s) for s in slot_indices.tolist())))
        marker = 9000.0 + layer_id * 10.0
        self.k_buffer[layer_id][slot_indices] = (
            marker + token_positions.float().unsqueeze(-1).unsqueeze(-1)
        )
        return LayerRecomputeResult(
            layer_id=layer_id,
            recomputed_slot_indices=tuple(int(s) for s in slot_indices.tolist()),
        )


class PartialLayerRecomputeBackend:
    """Fake backend that (incorrectly) only covers part of the requested
    slots -- used to prove the coordinator rejects incomplete coverage."""

    def recompute_layer(self, *, layer_id, slot_indices, token_positions):
        del token_positions
        covered = slot_indices[:-1]
        return LayerRecomputeResult(
            layer_id=layer_id,
            recomputed_slot_indices=tuple(int(s) for s in covered.tolist()),
        )


class TestLayerRecomputeCoordinator(unittest.TestCase):
    def test_recompute_issues_exactly_one_batched_call_per_layer(self):
        layer_num = 4
        k_buffer = torch.zeros(layer_num, 64, 2, 4)
        backend = RecordingLayerRecomputeBackend(k_buffer)
        coordinator = LayerRecomputeCoordinator(
            backend,
            first_recompute_layer=1,
            layer_num=layer_num,
        )
        selected_slots = [10, 20, 30, 40, 50]
        selected_positions = [100, 200, 300, 400, 500]
        results = coordinator.recompute_selected(
            slot_indices=selected_slots,
            token_positions=selected_positions,
        )

        # Exactly one call per recomputed layer (1, 2, 3) -- never a
        # per-token loop like the falsified "minipre" approach.
        self.assertEqual(len(backend.calls), 3)
        for layer_id, called_slots in backend.calls:
            self.assertEqual(set(called_slots), set(selected_slots))
        self.assertEqual([r.layer_id for r in results], [1, 2, 3])

        # Layer 0 (before first_recompute_layer) must be untouched.
        torch.testing.assert_close(k_buffer[0], torch.zeros(64, 2, 4))
        # Layers 1..3 carry the fake backend's real marker at the selected
        # slots only.
        for layer_id in (1, 2, 3):
            for slot, position in zip(selected_slots, selected_positions):
                expected = 9000.0 + layer_id * 10.0 + position
                self.assertTrue(
                    torch.allclose(
                        k_buffer[layer_id][slot],
                        torch.full((2, 4), expected),
                    )
                )

    def test_empty_selection_issues_no_calls(self):
        k_buffer = torch.zeros(3, 8, 2, 4)
        backend = RecordingLayerRecomputeBackend(k_buffer)
        coordinator = LayerRecomputeCoordinator(
            backend, first_recompute_layer=1, layer_num=3
        )
        results = coordinator.recompute_selected(slot_indices=[], token_positions=[])
        self.assertEqual(results, ())
        self.assertEqual(backend.calls, [])

    def test_partial_backend_coverage_is_rejected(self):
        coordinator = LayerRecomputeCoordinator(
            PartialLayerRecomputeBackend(),
            first_recompute_layer=0,
            layer_num=1,
        )
        with self.assertRaisesRegex(RuntimeError, "did not cover exactly"):
            coordinator.recompute_selected(
                slot_indices=[1, 2, 3], token_positions=[10, 20, 30]
            )

    def test_duplicate_slots_rejected(self):
        k_buffer = torch.zeros(2, 8, 2, 4)
        coordinator = LayerRecomputeCoordinator(
            RecordingLayerRecomputeBackend(k_buffer),
            first_recompute_layer=0,
            layer_num=2,
        )
        with self.assertRaises(ValueError):
            coordinator.recompute_selected(
                slot_indices=[1, 1], token_positions=[10, 11]
            )

    def test_misaligned_inputs_rejected(self):
        k_buffer = torch.zeros(2, 8, 2, 4)
        coordinator = LayerRecomputeCoordinator(
            RecordingLayerRecomputeBackend(k_buffer),
            first_recompute_layer=0,
            layer_num=2,
        )
        with self.assertRaises(ValueError):
            coordinator.recompute_selected(
                slot_indices=[1, 2], token_positions=[10]
            )

    def test_invalid_layer_bounds_rejected(self):
        with self.assertRaises(ValueError):
            LayerRecomputeCoordinator(
                RecordingLayerRecomputeBackend(torch.zeros(2, 4, 2, 4)),
                first_recompute_layer=2,
                layer_num=2,
            )


if __name__ == "__main__":
    unittest.main()
