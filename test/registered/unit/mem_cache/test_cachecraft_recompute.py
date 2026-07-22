from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.srt.mem_cache.approx_kv.cachecraft_recompute import (
    CacheCraftRecomputeBackend,
    CacheCraftUnsupportedError,
    ChunkRecomputeHook,
    RecomputeInvocation,
)
from sglang.srt.mem_cache.approx_kv.types import KVLayerTransferResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class FakeKVCache:
    """Minimal fake KV cache: one (max_index, head_dim) tensor per K/V."""

    def __init__(self, num_slots: int = 32, head_dim: int = 4):
        self.k_buffer = torch.zeros(num_slots, head_dim)
        self.v_buffer = torch.zeros(num_slots, head_dim)


class FakeInnerBackend:
    """Stands in for the real `RadixKVTransferBackend.copy_and_rotate`."""

    def __init__(self):
        self.copy_calls: list[dict] = []

    def copy_and_rotate(self, **kwargs) -> KVLayerTransferResult:
        self.copy_calls.append(kwargs)
        n = kwargs.get("length", 0)
        return KVLayerTransferResult(
            copied_k_tokens=n, rotated_k_tokens=n, copied_v_tokens=n
        )


@dataclass
class RealMarkerRecomputeHook:
    """A genuine (if fake-model) recompute hook: it performs an actual
    computation over `token_ids` (not just bookkeeping) and writes real,
    distinguishable values into the caller-specified physical KV indices,
    proving the hook is a real per-token writer rather than metadata-only.
    """

    kvcache: FakeKVCache
    calls: list[dict] = field(default_factory=list)

    def recompute(
        self,
        *,
        kvcache: Any,
        target_indices: torch.Tensor,
        token_ids: tuple[int, ...],
        reason: str,
    ) -> KVLayerTransferResult:
        self.calls.append(
            {
                "target_indices": target_indices.clone(),
                "token_ids": token_ids,
                "reason": reason,
            }
        )
        for slot, position, token_id in zip(
            target_indices.tolist(), range(len(token_ids)), token_ids
        ):
            # A real per-token computation: derive a value from the actual
            # token id and its position, not a constant/no-op placeholder.
            marker = float(token_id) * 100.0 + float(position)
            kvcache.k_buffer[slot] = torch.full((kvcache.k_buffer.shape[1],), marker)
            kvcache.v_buffer[slot] = torch.full(
                (kvcache.v_buffer.shape[1],), marker + 0.5
            )
        return KVLayerTransferResult(
            copied_k_tokens=len(token_ids),
            rotated_k_tokens=len(token_ids),
            copied_v_tokens=len(token_ids),
        )


class IncompleteRecomputeHook:
    """Simulates an unsupported/broken hook that only covers part of the
    requested range -- must be rejected, not silently accepted."""

    def recompute(self, *, kvcache, target_indices, token_ids, reason):
        n = len(token_ids)
        return KVLayerTransferResult(
            copied_k_tokens=max(0, n - 1), rotated_k_tokens=n, copied_v_tokens=n
        )


class MisalignedRecomputeHook:
    """Simulates a hook whose keys are position-incorrect (rotated count
    does not match copied count) -- must be rejected."""

    def recompute(self, *, kvcache, target_indices, token_ids, reason):
        n = len(token_ids)
        return KVLayerTransferResult(
            copied_k_tokens=n, rotated_k_tokens=max(0, n - 1), copied_v_tokens=n
        )


def _make_backend(kvcache: FakeKVCache, hook: ChunkRecomputeHook | None):
    inner = FakeInnerBackend()
    backend = CacheCraftRecomputeBackend(
        inner=inner,
        kvcache=kvcache,
        target_indices=lambda start, length: torch.arange(start, start + length),
        token_ids=lambda start, length: tuple(
            range(1000 + start, 1000 + start + length)
        ),
        recompute_hook=hook,
    )
    return backend, inner


class TestCacheCraftRecomputeBackendInvokesRealHook(unittest.TestCase):
    def test_dense_prefill_invokes_hook_and_writes_real_distinguishable_values(self):
        kvcache = FakeKVCache()
        hook = RealMarkerRecomputeHook(kvcache=kvcache)
        backend, inner = _make_backend(kvcache, hook)

        # Sanity: buffers start at zero everywhere.
        self.assertTrue(
            torch.equal(kvcache.k_buffer, torch.zeros_like(kvcache.k_buffer))
        )

        backend.dense_prefill(
            target_start=5, length=3, reason="cachecraft_partial_repair"
        )

        # The hook was really invoked (not a no-op / metadata-only path):
        self.assertEqual(len(hook.calls), 1)
        call = hook.calls[0]
        self.assertEqual(call["token_ids"], (1005, 1006, 1007))
        self.assertEqual(call["reason"], "cachecraft_partial_repair")
        self.assertTrue(torch.equal(call["target_indices"], torch.tensor([5, 6, 7])))

        # The genuine per-token computation actually wrote distinguishable,
        # token/position-derived values into exactly the selected slots...
        for position, token_id in enumerate((1005, 1006, 1007)):
            slot = 5 + position
            expected_k = float(token_id) * 100.0 + float(position)
            self.assertTrue(
                torch.allclose(
                    kvcache.k_buffer[slot],
                    torch.full_like(kvcache.k_buffer[slot], expected_k),
                )
            )
            self.assertTrue(
                torch.allclose(
                    kvcache.v_buffer[slot],
                    torch.full_like(kvcache.v_buffer[slot], expected_k + 0.5),
                )
            )

        # ...and left every other slot untouched (still zero).
        untouched_mask = torch.ones(kvcache.k_buffer.shape[0], dtype=torch.bool)
        untouched_mask[5:8] = False
        self.assertTrue(
            torch.equal(
                kvcache.k_buffer[untouched_mask],
                torch.zeros_like(kvcache.k_buffer[untouched_mask]),
            )
        )

        # Real invocation bookkeeping is recorded for telemetry/tests.
        self.assertEqual(len(backend.invocations), 1)
        self.assertIsInstance(backend.invocations[0], RecomputeInvocation)
        self.assertEqual(backend.recomputed_tokens, 3)
        self.assertEqual(backend.unsupported_reasons, [])

    def test_copy_and_rotate_delegates_to_real_inner_backend(self):
        kvcache = FakeKVCache()
        backend, inner = _make_backend(
            kvcache, RealMarkerRecomputeHook(kvcache=kvcache)
        )
        backend.copy_and_rotate(target_start=0, length=4)
        self.assertEqual(len(inner.copy_calls), 1)
        self.assertEqual(inner.copy_calls[0]["length"], 4)

    def test_no_hook_records_unsupported_reason_and_writes_nothing(self):
        kvcache = FakeKVCache()
        backend, _ = _make_backend(kvcache, hook=None)
        backend.dense_prefill(
            target_start=2, length=2, reason="no_recompute_hook_available"
        )
        self.assertEqual(backend.unsupported_reasons, ["no_recompute_hook_available"])
        self.assertEqual(backend.invocations, [])
        self.assertTrue(
            torch.equal(kvcache.k_buffer, torch.zeros_like(kvcache.k_buffer))
        )

    def test_incomplete_hook_result_raises_unsupported_error(self):
        kvcache = FakeKVCache()
        backend, _ = _make_backend(kvcache, IncompleteRecomputeHook())
        with self.assertRaises(CacheCraftUnsupportedError):
            backend.dense_prefill(target_start=0, length=3, reason="partial")
        # A rejected hook result must not be recorded as a real invocation.
        self.assertEqual(backend.invocations, [])

    def test_misaligned_rope_hook_result_raises_unsupported_error(self):
        kvcache = FakeKVCache()
        backend, _ = _make_backend(kvcache, MisalignedRecomputeHook())
        with self.assertRaises(CacheCraftUnsupportedError):
            backend.dense_prefill(target_start=0, length=3, reason="partial")
        self.assertEqual(backend.invocations, [])

    def test_token_ids_length_mismatch_raises_value_error(self):
        kvcache = FakeKVCache()
        inner = FakeInnerBackend()
        backend = CacheCraftRecomputeBackend(
            inner=inner,
            kvcache=kvcache,
            target_indices=lambda start, length: torch.arange(start, start + length),
            token_ids=lambda start, length: tuple(range(length - 1)),  # wrong length
            recompute_hook=RealMarkerRecomputeHook(kvcache=kvcache),
        )
        with self.assertRaises(ValueError):
            backend.dense_prefill(target_start=0, length=3, reason="partial")


if __name__ == "__main__":
    unittest.main()
