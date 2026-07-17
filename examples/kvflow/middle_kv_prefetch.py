"""Runnable CPU-only example for the middle-KV handoff interface.

Run from the repository root:

    PYTHONPATH=python python examples/kvflow/middle_kv_prefetch.py

The fake allocator models the methods already exposed by SGLang's
token_to_kv_pool_allocator. No CUDA device or model server is required.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.radix_backend import DeviceKVRef
from sglang.srt.mem_cache.kvcomm.types import (
    DenseRange,
    KVReusePlan,
    TransferSpan,
)
from sglang.srt.mem_cache.kvcomm_prefetch import MiddleKVPrefetchAPI


class DemoAllocator:
    def __init__(self) -> None:
        self._next_slot = 100
        self.host_exports: list[torch.Tensor] = []
        self.device_payloads: dict[int, torch.Tensor] = {}

    def alloc(self, need_size: int) -> torch.Tensor:
        indices = torch.arange(self._next_slot, self._next_slot + need_size)
        self._next_slot += need_size
        return indices

    def free(self, indices: torch.Tensor) -> None:
        for index in indices.tolist():
            self.device_payloads.pop(index, None)

    def get_kvcache(self):
        raise NotImplementedError("this export/prefetch example does not consume KV")

    def get_cpu_copy(self, indices: torch.Tensor) -> torch.Tensor:
        # A real allocator returns the all-layer K/V payload for these slots.
        payload = indices.detach().cpu().clone()
        self.host_exports.append(payload)
        return payload

    def load_cpu_copy(self, payload: torch.Tensor, indices: torch.Tensor) -> None:
        for destination, value in zip(indices.tolist(), payload.tolist()):
            self.device_payloads[destination] = torch.tensor(value)


class DemoTransferBackend:
    """Records the policy-neutral transfer request made by KVCommManager."""

    def __init__(self) -> None:
        self.dense_ranges: list[tuple[int, int, str]] = []
        self.copies: list[tuple[list[int], int, int, int, int]] = []

    def dense_prefill(
        self, *, target_start: int, length: int, reason: str
    ) -> None:
        self.dense_ranges.append((target_start, length, reason))

    def copy_and_rotate(
        self,
        *,
        source_ref: object,
        source_offset: int,
        target_start: int,
        length: int,
        rope_delta: int,
    ) -> tuple[int, int, int]:
        assert isinstance(source_ref, DeviceKVRef)
        self.copies.append(
            (
                source_ref.indices.tolist(),
                source_offset,
                target_start,
                length,
                rope_delta,
            )
        )
        # A production RadixKVTransferBackend copies all K/V and rotates all K.
        return length, length, length


def main() -> None:
    allocator = DemoAllocator()
    manager = KVCommManager(
        KVCommFeatureConfig(core_enabled=True, prefetch_enabled=True)
    )
    middle_kv = MiddleKVPrefetchAPI(
        manager=manager,
        allocator=allocator,
        model_id="Qwen2.5-Coder-7B-Instruct",
        cache_dtype="bf16",
    )

    # Producer side: export a computed middle-of-request segment to host.
    exported = middle_kv.export_middle_kv(
        token_ids=(101, 102, 103),
        kv_indices=torch.tensor([40, 41, 42]),
        source_start=256,
        content_hash="repo.py:parse_config:v1",
    )
    print("exported:", exported.key.content_hash, exported.residency.value)

    # Consumer/scheduler side: request device residency before the request runs.
    ticket = middle_kv.prefetch(exported.key, deadline_s=0.050, priority=10)
    with ticket:
        print("prefetched device slots:", ticket.device_indices().tolist())
        resident = ticket.wait()
        target_tokens = (900, 101, 102, 103, 901)
        plan = KVReusePlan(
            target_token_ids=target_tokens,
            copied_spans=(
                TransferSpan(
                    source=resident,
                    source_offset=0,
                    target_start=1,
                    length=3,
                    rope_delta=1 - resident.source_start,
                    chunk_start=1,
                    chunk_length=3,
                ),
            ),
            dense_ranges=(
                DenseRange(target_start=0, length=1, reason="new_prefix"),
                DenseRange(target_start=4, length=1, reason="new_suffix"),
            ),
            require_full_coverage=True,
        )
        backend = DemoTransferBackend()
        stats = manager.execute(plan, backend)
        print(
            "consumed:",
            {
                "copied_tokens": stats.copied_k_tokens,
                "recomputed_tokens": stats.recomputed_tokens,
                "mechanically_valid": stats.mechanically_valid,
            },
        )

    # The scheduler owns the prefetched copy and explicitly drops it when stale.
    middle_kv.drop(exported.key)
    print("released device payload:", allocator.device_payloads == {})


if __name__ == "__main__":
    main()
