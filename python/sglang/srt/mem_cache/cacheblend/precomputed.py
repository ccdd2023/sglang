from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from sglang.srt.mem_cache.approx_kv.radix_backend import DeviceKVRef
from sglang.srt.mem_cache.approx_kv.types import KVSegmentHandle

from .recompute import LayerRecomputeResult


@dataclass(frozen=True)
class FreshKVSpan:
    target_start: int
    length: int
    source: KVSegmentHandle
    source_offset: int = 0

    def __post_init__(self) -> None:
        if self.target_start < 0 or self.length <= 0 or self.source_offset < 0:
            raise ValueError("fresh KV span has invalid bounds")
        if not isinstance(self.source.backend_ref, DeviceKVRef):
            raise TypeError("fresh KV span requires a device-resident handle")

    @property
    def target_end(self) -> int:
        return self.target_start + self.length


class PrecomputedCacheBlendBackend:
    """Fresh target-context KV captured by a separate dense preparation request."""

    def __init__(self, *, kvcache, spans: Sequence[FreshKVSpan]) -> None:
        self._kvcache = kvcache
        self._spans = tuple(sorted(spans, key=lambda span: span.target_start))
        if not self._spans:
            raise ValueError("at least one fresh KV span is required")

    def _source_indices(self, token_positions: torch.Tensor) -> torch.Tensor:
        indices = []
        for raw_position in token_positions.tolist():
            position = int(raw_position)
            match = next(
                (
                    span
                    for span in self._spans
                    if span.target_start <= position < span.target_end
                ),
                None,
            )
            if match is None:
                raise KeyError(
                    f"fresh KV is unavailable for token position {position}"
                )
            offset = match.source_offset + position - match.target_start
            source_ref = match.source.backend_ref
            indices.append(source_ref.indices[offset])
        if not indices:
            return torch.empty(0, dtype=torch.long)
        return torch.stack(indices).long()

    def probe_layer(
        self,
        *,
        layer_id: int,
        slot_indices: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        del slot_indices
        key_buffer = self._kvcache.get_key_buffer(layer_id)
        source_indices = self._source_indices(token_positions).to(key_buffer.device)
        return key_buffer[source_indices].clone()

    def recompute_layer(
        self,
        *,
        layer_id: int,
        slot_indices: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> LayerRecomputeResult:
        key_buffer = self._kvcache.get_key_buffer(layer_id)
        value_buffer = self._kvcache.get_value_buffer(layer_id)
        target_indices = slot_indices.to(key_buffer.device).long()
        source_indices = self._source_indices(token_positions).to(
            key_buffer.device
        )
        key_buffer[target_indices] = key_buffer[source_indices]
        value_buffer[target_indices] = value_buffer[source_indices]
        return LayerRecomputeResult(
            layer_id=layer_id,
            recomputed_slot_indices=tuple(
                int(index) for index in slot_indices.tolist()
            ),
        )
