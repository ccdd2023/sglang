from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.utils.hf_transformers.common import get_rope_config

from .store import ResidencyLoadResult
from .types import (
    KVLayerTransferResult,
    KVSegmentHandle,
    ResidencyTier,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

# Architectures this module's cross-context relocation (see
# `_rotate_all_copied_keys` below) has actually been verified against:
# neox-style rotation over the *full* head_dim, with rope_theta/
# rope_scaling read via the standard `rope_parameters`-or-`rope_theta`/
# `rope_scaling` precedence (see qwen2.py/qwen3.py's own Attention
# modules). Many other architectures in this codebase pass
# `is_neox_style=False` to `get_rope` (chatglm.py, glm4.py, commandr.py,
# cohere2_moe.py, gpt_j.py, ernie45_moe_vl.py, mistral_large_3.py,
# sarvam_moe.py, deepseek_v4.py) or use a partial rotary factor / other
# conventions this module has not been validated against; applying the
# neox-style formula to one of those would silently relocate to the
# WRONG KV value, not merely skip an optimization, so
# `resolve_model_rope_config` conservatively refuses to resolve a real
# RoPE config for anything outside this allowlist.
_ROPE_RELOCATION_VERIFIED_ARCHITECTURES = frozenset(
    {
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
    }
)


@dataclass(frozen=True)
class DeviceKVRef:
    indices: torch.Tensor


@dataclass(frozen=True)
class CPUKVRef:
    payload: Any


@dataclass(frozen=True)
class RoPEConfig:
    rotary_dim: int
    base: float
    is_neox_style: bool

    def __post_init__(self) -> None:
        if self.rotary_dim < 0 or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be a non-negative even number")
        if self.base <= 0:
            raise ValueError("RoPE base must be positive")


def _is_genuinely_scaled(rope_params: dict[str, Any] | None) -> tuple[bool, Any]:
    """Return `(genuinely_scaled, scaling_type)` for a resolved
    rope_scaling/rope_parameters dict.

    Mirrors `sglang.srt.layers.rotary_embedding.factory.get_rope`'s own
    dispatch precedence exactly: transformers v5's `rope_parameters` may
    be non-None even for models with no real scaling in effect (see
    `get_rope_config`'s docstring), so a non-None dict alone does not
    mean "scaled". The dispatch key is `rope_scaling["rope_type"]`,
    falling back to `rope_scaling["type"]`; only the literal value
    `"default"` -- with neither an `mrope_section` (multimodal M-RoPE)
    nor a truthy `use_fope` key, both of which `get_rope` special-cases
    even under `scaling_type == "default"` -- means "plain, unscaled
    RoPE". Anything else (`"llama3"`, `"linear"`, `"dynamic"`, `"yarn"`,
    `"deepseek_yarn"`, `"longrope"`, `"proportional"`, a mrope/fope
    variant, or a missing/unrecognized key) genuinely alters the
    frequency basis and is NOT reproduced by this module's simple
    `base**(-2i/d)` absolute-delta rotation.
    """
    if not rope_params:
        return False, "default"
    scaling_type = rope_params.get("rope_type", rope_params.get("type"))
    if scaling_type != "default":
        return True, scaling_type
    if "mrope_section" in rope_params or rope_params.get("use_fope", False):
        return True, scaling_type
    return False, scaling_type


def resolve_model_rope_config(model_config: ModelConfig) -> RoPEConfig:
    """Resolve a real `RoPEConfig` for cross-context KV relocation from a
    loaded `ModelConfig`, for the Qwen2/Qwen3 dense text-model family
    this project's SM75 canary targets.

    This is the fix for a previously-dead binding: `ApproxKVManager.
    rope_config` was never populated in production (nothing ever called
    `bind_rope_config`), so `restore_request_prefix`/
    `restore_request_prefix_cachetune` always fell back to the dummy
    `RoPEConfig(rotary_dim=0, ...)` sentinel, which forces a dense
    fallback on *every* repair whose source/target token positions
    differ (`rope_delta != 0`) -- not a rare edge case, but the common
    case for any multi-segment / non-contiguous-origin restore (see
    `create_tree_cache`'s caller in `kv_cache_builder.py`, which binds
    this function's result immediately after tree-cache construction).

    Conservative, no-silent-wrong-rotation fallback (returns
    `RoPEConfig(rotary_dim=0, ...)`, the sentinel both restore paths
    already treat as "rope config unavailable -> dense fallback on any
    nonzero-delta repair") whenever:

      - `model_config`'s architecture is not one this module's neox-
        style, full-head-dim relocation formula has been verified
        against (see `_ROPE_RELOCATION_VERIFIED_ARCHITECTURES`), or
      - a non-empty `dual_chunk_attention_config` is present (Qwen2/2.5
        long-context checkpoints route through `DualChunkRotaryEmbedding`
        -- a chunk-aware, clamped-position scheme, not the plain
        absolute-delta neox rotation this module implements -- see
        `qwen2.py`'s own `dual_chunk_attention_config` plumbing into
        `get_rope`), or
      - `rope_scaling`/`rope_parameters` resolves to a genuinely-scaled
        scheme (see `_is_genuinely_scaled`) -- YaRN/linear/dynamic-NTK/
        llama3/mrope/fope/... variants alter the frequency basis in a
        way this module's simple delta rotation does not reproduce.

    Never silently applies a rotation this module cannot vouch for; the
    caller (`restore_request_prefix`/`restore_request_prefix_cachetune`)
    is already required to dense-fall-back cleanly on
    `rotary_dim == 0`, so returning the sentinel here is always safe,
    merely slower.
    """
    architectures = tuple(getattr(model_config.hf_config, "architectures", None) or ())
    if not any(
        name in _ROPE_RELOCATION_VERIFIED_ARCHITECTURES for name in architectures
    ):
        logger.warning(
            "approx_kv RoPE relocation: architectures=%s is not in the "
            "verified Qwen2/Qwen3 dense-text-model allowlist %s; "
            "conservatively disabling cross-context RoPE relocation "
            "(rotary_dim=0 -> dense fallback on any nonzero-delta "
            "repair) rather than risk an unverified rotation "
            "convention.",
            architectures,
            sorted(_ROPE_RELOCATION_VERIFIED_ARCHITECTURES),
        )
        return RoPEConfig(rotary_dim=0, base=1.0, is_neox_style=True)

    rope_theta, rope_params = get_rope_config(model_config.hf_text_config)

    dual_chunk_attention_config = getattr(
        model_config.hf_text_config, "dual_chunk_attention_config", None
    )
    if dual_chunk_attention_config:
        logger.warning(
            "approx_kv RoPE relocation: architectures=%s has a non-empty "
            "dual_chunk_attention_config=%r; dual-chunk long-context "
            "attention routes through DualChunkRotaryEmbedding's chunked/"
            "clamped position scheme, not this module's plain neox "
            "absolute-delta rotation; conservatively disabling "
            "cross-context RoPE relocation (rotary_dim=0 -> dense "
            "fallback) rather than risk computing a wrong rotation.",
            architectures,
            dual_chunk_attention_config,
        )
        return RoPEConfig(rotary_dim=0, base=float(rope_theta), is_neox_style=True)

    genuinely_scaled, scaling_type = _is_genuinely_scaled(rope_params)
    if genuinely_scaled:
        logger.warning(
            "approx_kv RoPE relocation: rope_scaling type=%r for "
            "architectures=%s genuinely alters the frequency basis; "
            "conservatively disabling cross-context RoPE relocation "
            "(rotary_dim=0 -> dense fallback) rather than risk "
            "computing a wrong rotation.",
            scaling_type,
            architectures,
        )
        return RoPEConfig(rotary_dim=0, base=float(rope_theta), is_neox_style=True)

    return RoPEConfig(
        rotary_dim=model_config.head_dim,
        base=float(rope_theta),
        is_neox_style=True,
    )


class KVPoolAllocator(Protocol):
    def alloc(self, need_size: int) -> torch.Tensor | None: ...

    def free(self, indices: torch.Tensor) -> None: ...

    def get_kvcache(self) -> Any: ...

    def get_cpu_copy(
        self,
        indices: torch.Tensor,
        mamba_indices: torch.Tensor | None = None,
    ) -> Any: ...

    def load_cpu_copy(
        self,
        payload: Any,
        indices: torch.Tensor,
        mamba_indices: torch.Tensor | None = None,
    ) -> None: ...


class AllocatorCPUResidencyBackend:
    def __init__(self, allocator: KVPoolAllocator) -> None:
        self._allocator = allocator

    def export_to_host(self, device_ref: DeviceKVRef) -> ResidencyLoadResult:
        payload = self._allocator.get_cpu_copy(device_ref.indices)
        return ResidencyLoadResult(
            backend_ref=CPUKVRef(payload),
            release_backend=self.release_host,
        )

    def load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> ResidencyLoadResult:
        if target_tier != ResidencyTier.DEVICE:
            raise NotImplementedError("allocator backend loads only to device")
        if not isinstance(handle.backend_ref, CPUKVRef):
            raise TypeError("host-resident handle must carry CPUKVRef")
        indices = self._allocator.alloc(len(handle.token_ids))
        if indices is None or len(indices) != len(handle.token_ids):
            if indices is not None:
                self._allocator.free(indices)
            raise MemoryError("unable to allocate device slots for approximate KV")
        try:
            self._allocator.load_cpu_copy(
                handle.backend_ref.payload,
                indices,
            )
        except Exception:
            self._allocator.free(indices)
            raise
        return ResidencyLoadResult(
            backend_ref=DeviceKVRef(indices),
            release_backend=self.release_device,
        )

    def release_device(
        self,
        backend_ref: object,
        residency: ResidencyTier,
    ) -> None:
        if residency != ResidencyTier.DEVICE or not isinstance(
            backend_ref,
            DeviceKVRef,
        ):
            raise TypeError("allocator releaser received a non-device KV ref")
        self._allocator.free(backend_ref.indices)

    @staticmethod
    def release_host(
        backend_ref: object,
        residency: ResidencyTier,
    ) -> None:
        if residency != ResidencyTier.HOST or not isinstance(
            backend_ref,
            CPUKVRef,
        ):
            raise TypeError("allocator releaser received a non-host KV ref")


class RadixKVTransferBackend:
    def __init__(
        self,
        *,
        allocator: KVPoolAllocator,
        target_indices: Callable[[int, int], torch.Tensor],
        dense_prefill: Callable[[int, int, str], None],
        rope: RoPEConfig,
    ) -> None:
        self._allocator = allocator
        self._target_indices = target_indices
        self._dense_prefill = dense_prefill
        self._rope = rope

    def dense_prefill(
        self,
        *,
        target_start: int,
        length: int,
        reason: str,
    ) -> None:
        self._dense_prefill(target_start, length, reason)

    def copy_and_rotate(
        self,
        *,
        source_ref: object,
        source_offset: int,
        target_start: int,
        length: int,
        rope_delta: int,
    ) -> KVLayerTransferResult:
        if not isinstance(source_ref, DeviceKVRef):
            raise TypeError("device transfer requires DeviceKVRef")
        source_indices = source_ref.indices[source_offset : source_offset + length]
        target_indices = self._target_indices(target_start, length)
        self._validate_indices(source_indices, target_indices, length)

        kvcache = self._allocator.get_kvcache()
        copy_start = time.perf_counter()
        kvcache.move_kv_cache(target_indices, source_indices)
        copy_ms = (time.perf_counter() - copy_start) * 1000

        rope_start = time.perf_counter()
        self._rotate_all_copied_keys(
            kvcache=kvcache,
            target_indices=target_indices,
            rope_delta=rope_delta,
        )
        rope_ms = (time.perf_counter() - rope_start) * 1000
        return KVLayerTransferResult(
            copied_k_tokens=length,
            rotated_k_tokens=length,
            copied_v_tokens=length,
            copy_ms=copy_ms,
            rope_ms=rope_ms,
        )

    @staticmethod
    def _validate_indices(
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        length: int,
    ) -> None:
        if len(source_indices) != length or len(target_indices) != length:
            raise ValueError("physical KV index slice length mismatch")

    def _rotate_all_copied_keys(
        self,
        *,
        kvcache: Any,
        target_indices: torch.Tensor,
        rope_delta: int,
    ) -> None:
        rotary_dim = self._rope.rotary_dim
        if rotary_dim == 0 or rope_delta == 0 or len(target_indices) == 0:
            return
        flat_indices = target_indices.reshape(-1).long()
        first_key_buffer = kvcache.get_key_buffer(0)
        device = first_key_buffer.device
        inverse_frequency = 1.0 / (
            self._rope.base
            ** (
                torch.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / rotary_dim
            )
        )
        delta = torch.full(
            (len(flat_indices),),
            int(rope_delta),
            dtype=torch.float32,
            device=device,
        )
        frequencies = torch.einsum("i,j->ij", delta, inverse_frequency)
        cosine = frequencies.cos()
        sine = frequencies.sin()

        for layer_id in range(kvcache.layer_num):
            key_buffer = kvcache.get_key_buffer(layer_id)
            selected = key_buffer[flat_indices]
            if selected.shape[-1] < rotary_dim:
                raise ValueError("KV head dimension is smaller than rotary_dim")
            rotary = selected[..., :rotary_dim]
            rotated = apply_rotary_emb(
                rotary,
                cosine,
                sine,
                self._rope.is_neox_style,
            )
            if rotary_dim == selected.shape[-1]:
                key_buffer[flat_indices] = rotated
            else:
                key_buffer[flat_indices, ..., :rotary_dim] = rotated
