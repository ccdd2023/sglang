from __future__ import annotations

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.kvcomm.radix_backend import (
    AllocatorResidencyLoader,
    DeviceKVRef,
    HostKVRef,
    RadixKVTransferBackend,
    RoPEConfig,
)
from sglang.srt.mem_cache.kvcomm.types import (
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)


class Cache:
    def __init__(self, layers=2, size=16, heads=2, dim=4):
        self.layer_num = layers
        generator = torch.Generator().manual_seed(7)
        self.keys = [
            torch.randn(size, heads, dim, generator=generator)
            for _ in range(layers)
        ]
        self.values = [
            torch.randn(size, heads, dim, generator=generator)
            for _ in range(layers)
        ]

    def move_kv_cache(self, target, source):
        for layer in range(self.layer_num):
            self.keys[layer][target] = self.keys[layer][source].clone()
            self.values[layer][target] = self.values[layer][source].clone()

    def get_key_buffer(self, layer):
        return self.keys[layer]


class Allocator:
    def __init__(self, cache):
        self.cache = cache
        self.next_slot = 8
        self.freed = []
        self.loaded = []

    def alloc(self, size):
        result = torch.arange(self.next_slot, self.next_slot + size)
        self.next_slot += size
        return result

    def free(self, indices):
        self.freed.extend(indices.tolist())

    def get_kvcache(self):
        return self.cache

    def load_cpu_copy(self, payload, indices):
        self.loaded.append((payload, tuple(indices.tolist())))


def _handle(tokens, backend_ref, residency):
    key = KVSegmentKey(
        content_hash="segment",
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="test",
        cache_dtype="fp32",
        kind=SegmentKind.MIDDLE,
    )
    return KVSegmentHandle(
        key=key,
        generation=1,
        residency=residency,
        source_start=0,
        token_ids=tuple(tokens),
        backend_ref=backend_ref,
    )


def _reference_rotation(keys, indices, delta, config):
    inverse_frequency = 1.0 / (
        config.base
        ** (
            torch.arange(0, config.rotary_dim, 2, dtype=torch.float32)
            / config.rotary_dim
        )
    )
    frequencies = torch.einsum(
        "i,j->ij",
        torch.full((len(indices),), float(delta)),
        inverse_frequency,
    )
    selected = keys[indices][..., : config.rotary_dim]
    return apply_rotary_emb(
        selected,
        frequencies.cos(),
        frequencies.sin(),
        config.is_neox_style,
    )


@torch.no_grad()
def test_device_copy_uses_requested_source_offset_and_rotates_all_k():
    cache = Cache()
    allocator = Allocator(cache)
    source_indices = torch.arange(0, 6)
    target_indices = torch.arange(8, 12)
    original_keys = [layer.clone() for layer in cache.keys]
    original_values = [layer.clone() for layer in cache.values]
    config = RoPEConfig(rotary_dim=4, base=10_000, is_neox_style=True)
    backend = RadixKVTransferBackend(
        allocator=allocator,
        target_indices=lambda start, length: target_indices,
        dense_prefill=lambda *_: None,
        rope=config,
    )
    counts = backend.copy_and_rotate(
        source_ref=DeviceKVRef(source_indices),
        source_offset=2,
        target_start=0,
        length=4,
        rope_delta=5,
    )
    assert counts == (4, 4, 4)
    for layer in range(cache.layer_num):
        expected_k = _reference_rotation(
            original_keys[layer], source_indices[2:6], 5, config
        )
        assert torch.allclose(
            cache.keys[layer][target_indices], expected_k, atol=2e-6
        )
        assert torch.equal(
            cache.values[layer][target_indices],
            original_values[layer][source_indices[2:6]],
        )


@torch.no_grad()
def test_positive_negative_and_zero_rope_delta_match_reference():
    for delta in (-11, 0, 13):
        cache = Cache(layers=1)
        allocator = Allocator(cache)
        source = torch.arange(0, 3)
        target = torch.arange(8, 11)
        original = cache.keys[0].clone()
        config = RoPEConfig(rotary_dim=4, base=10_000, is_neox_style=False)
        backend = RadixKVTransferBackend(
            allocator=allocator,
            target_indices=lambda start, length, target=target: target,
            dense_prefill=lambda *_: None,
            rope=config,
        )
        backend.copy_and_rotate(
            source_ref=DeviceKVRef(source),
            source_offset=0,
            target_start=0,
            length=3,
            rope_delta=delta,
        )
        expected = (
            original[source]
            if delta == 0
            else _reference_rotation(original, source, delta, config)
        )
        assert torch.allclose(cache.keys[0][target], expected, atol=2e-6)


def test_host_residency_loader_allocates_and_loads_payload():
    cache = Cache()
    allocator = Allocator(cache)
    tokens = (1, 2, 3)
    handle = _handle(tokens, HostKVRef(payload="cpu-kv"), ResidencyTier.HOST)
    result = AllocatorResidencyLoader(allocator).load(
        handle, ResidencyTier.DEVICE
    )
    loaded = result.backend_ref
    assert isinstance(loaded, DeviceKVRef)
    assert tuple(loaded.indices.tolist()) == (8, 9, 10)
    assert allocator.loaded == [("cpu-kv", (8, 9, 10))]
    assert result.release_backend is not None
    result.release_backend(loaded, ResidencyTier.DEVICE)
    assert allocator.freed == [8, 9, 10]
