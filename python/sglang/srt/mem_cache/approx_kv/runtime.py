from __future__ import annotations

from typing import Any

import torch

from .radix_backend import (
    DeviceKVRef,
    RadixKVTransferBackend,
    RoPEConfig,
)
from .types import (
    KVReusePlan,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)


def _allocator(tree_cache: Any) -> Any:
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator is None:
        raise RuntimeError("approximate KV requires a token allocator")
    return allocator


def _release_device_ref(allocator: Any):
    def release(backend_ref: object, residency: ResidencyTier) -> None:
        if residency != ResidencyTier.DEVICE or not isinstance(
            backend_ref,
            DeviceKVRef,
        ):
            raise TypeError("invalid approximate KV device reference")
        allocator.free(backend_ref.indices)

    return release


def register_request_segments(tree_cache: Any, req: Any) -> int:
    metadata = getattr(req, "approx_kv_metadata", None)
    if metadata is None or not metadata.register_source:
        return 0
    if req.req_pool_idx is None or req.kv is None:
        raise RuntimeError("request KV must exist before source registration")
    if not tree_cache.approx_kv.config.core_enabled:
        return 0

    allocator = _allocator(tree_cache)
    registered = 0
    for segment in metadata.segments:
        target_end = segment.target_end
        if target_end > req.effective_kv_committed_len():
            raise ValueError("source segment exceeds committed request KV")
        source_indices = tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            segment.target_start : target_end,
        ].clone()
        target_indices = allocator.alloc(segment.length)
        if target_indices is None or len(target_indices) != segment.length:
            if target_indices is not None:
                allocator.free(target_indices)
            raise MemoryError("unable to allocate canonical approximate KV slots")

        backend = RadixKVTransferBackend(
            allocator=allocator,
            target_indices=lambda start, length, indices=target_indices: indices[
                start : start + length
            ],
            dense_prefill=lambda start, length, reason: None,
            rope=RoPEConfig(
                rotary_dim=0,
                base=10000.0,
                is_neox_style=True,
            ),
        )
        backend.copy_and_rotate(
            source_ref=DeviceKVRef(source_indices),
            source_offset=0,
            target_start=0,
            length=segment.length,
            rope_delta=0,
        )
        tokens = tuple(
            int(token)
            for token in req.full_untruncated_fill_ids[
                segment.target_start : target_end
            ]
        )
        key = KVSegmentKey(
            content_hash=segment.source_content_hash,
            token_hash=token_ids_hash(tokens),
            token_count=len(tokens),
            model_id=metadata.model_id,
            cache_dtype=metadata.cache_dtype,
            kind=SegmentKind.CANONICAL_BASE,
        )
        tree_cache.approx_kv.register_segment(
            key=key,
            token_ids=tokens,
            source_start=segment.target_start,
            residency=ResidencyTier.DEVICE,
            backend_ref=DeviceKVRef(target_indices),
            release_backend=_release_device_ref(allocator),
        )
        registered += segment.length
    return registered


def restore_request_prefix(tree_cache: Any, req: Any) -> bool:
    metadata = getattr(req, "approx_kv_metadata", None)
    if (
        metadata is None
        or metadata.register_source
        or not tree_cache.approx_kv.config.core_enabled
        or not tree_cache.approx_kv.config.lossy_recovery_enabled
    ):
        return False
    if len(metadata.segments) != 1:
        raise NotImplementedError(
            "the sequential MVP accepts one whole-prefix source segment"
        )

    segment = metadata.segments[0]
    reusable_limit = len(req.full_untruncated_fill_ids) - 1
    exact_length = len(req.prefix_indices)
    if segment.target_start != 0 or segment.target_end < reusable_limit:
        raise ValueError("speed-only MVP source must cover the whole reusable prefix")
    restore_length = reusable_limit - exact_length
    if restore_length <= 0:
        return False

    source = tree_cache.approx_kv.store.find_by_content_hash(
        segment.source_content_hash
    )
    if source is None or len(source.token_ids) < reusable_limit:
        return False

    allocator = _allocator(tree_cache)
    restored_indices = allocator.alloc(restore_length)
    if restored_indices is None or len(restored_indices) != restore_length:
        if restored_indices is not None:
            allocator.free(restored_indices)
        return False

    fallback: list[str] = []
    backend = RadixKVTransferBackend(
        allocator=allocator,
        target_indices=lambda start, length: restored_indices[start : start + length],
        dense_prefill=lambda start, length, reason: fallback.append(reason),
        rope=RoPEConfig(
            rotary_dim=0,
            base=10000.0,
            is_neox_style=True,
        ),
    )
    target_tokens = tuple(
        int(token)
        for token in req.full_untruncated_fill_ids[exact_length:reusable_limit]
    )
    plan = KVReusePlan(
        target_token_ids=target_tokens,
        recovery_mode=metadata.recovery_mode,
        copied_spans=(
            TransferSpan(
                source=source,
                source_offset=exact_length,
                target_start=0,
                length=restore_length,
                rope_delta=0,
                chunk_start=0,
                chunk_length=restore_length,
            ),
        ),
        require_full_coverage=True,
        allow_token_mismatch=metadata.speed_only,
    )
    stats = tree_cache.approx_kv.execute(plan, backend)
    if fallback or stats.recomputed_tokens:
        allocator.free(restored_indices)
        return False

    req.prefix_indices = torch.cat(
        (
            req.prefix_indices,
            restored_indices.to(
                device=req.prefix_indices.device,
                dtype=req.prefix_indices.dtype,
            ),
        )
    )
    req.approx_kv_restored_len = restore_length
    return True
