from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import torch

from .plugins import RecoveryRequestContext
from .radix_backend import DeviceKVRef, RoPEConfig
from .store import ApproxKVSegmentStore
from .types import (
    DenseRange,
    KVReusePlan,
    KVSegmentHandle,
    KVSegmentKey,
    KVTransferStats,
    RecoveryMode,
    ResidencyTier,
    SchedulerMetadata,
    SegmentKind,
    token_ids_hash,
)


class KVCOMMInvariantError(RuntimeError):
    pass


class KVCOMMCapabilityError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class KVCOMMAction(str, Enum):
    BASE = "base"
    ANCHOR = "anchor"
    REUSE = "reuse"


class KVCOMMSegmentRole(str, Enum):
    PLACEHOLDER = "placeholder"
    NEIGHBOR = "neighbor"


@dataclass(frozen=True)
class KVCOMMRuntimeCapabilities:
    supported: bool
    reason: str | None
    model_type: str
    attention_arch: str
    cache_layout: str
    layer_count: int
    kv_head_count: int
    key_head_dim: int
    value_head_dim: int
    rope: RoPEConfig
    tp_size: int = 1
    pp_size: int = 1

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        *,
        tp_size: int,
        is_hybrid_swa: bool,
        is_hybrid_ssm: bool,
        is_multimodal: bool,
        is_speculative: bool,
        pp_size: int = 1,
    ) -> "KVCOMMRuntimeCapabilities":
        text_config = getattr(model_config, "hf_text_config", None)
        if text_config is None:
            text_config = getattr(model_config, "hf_config", None)
        model_type = str(getattr(text_config, "model_type", "")).lower()
        attention_arch = getattr(
            getattr(model_config, "attention_arch", None),
            "name",
            str(getattr(model_config, "attention_arch", "")),
        ).upper()
        layer_count = int(
            getattr(
                text_config,
                "num_hidden_layers",
                getattr(text_config, "num_layers", 0),
            )
            or 0
        )
        key_head_dim = int(getattr(model_config, "head_dim", 0) or 0)
        value_head_dim = int(getattr(model_config, "v_head_dim", key_head_dim) or 0)
        kv_head_count = int(getattr(model_config, "num_key_value_heads", 0) or 0)
        partial_rotary_factor = float(
            getattr(text_config, "partial_rotary_factor", 1.0) or 1.0
        )
        rotary_dim = int(key_head_dim * partial_rotary_factor)
        rotary_dim -= rotary_dim % 2
        rope_base = float(getattr(text_config, "rope_theta", 10000.0))
        is_neox_style = bool(getattr(text_config, "rope_is_neox_style", True))
        rope_scaling = getattr(text_config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(text_config, "rope_parameters", None)
        rope_type = "default"
        rope_factor = 1.0
        if isinstance(rope_scaling, Mapping):
            rope_type = str(
                rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"
            ).lower()
            rope_factor = float(rope_scaling.get("factor", 1.0) or 1.0)

        reason = None
        supported_models = {"llama", "qwen2", "qwen3"}
        if attention_arch != "MHA":
            reason = "unsupported_attention_arch"
        elif model_type not in supported_models:
            reason = "unsupported_model"
        elif tp_size != 1:
            reason = "unsupported_tensor_parallel"
        elif pp_size != 1:
            reason = "unsupported_pipeline_parallel"
        elif is_hybrid_swa or is_hybrid_ssm:
            reason = "unsupported_hybrid_cache"
        elif is_multimodal:
            reason = "unsupported_multimodal_model"
        elif is_speculative:
            reason = "unsupported_speculative_cache"
        elif rope_type != "default" or rope_factor != 1.0:
            reason = "unsupported_rope_scaling"
        elif (
            layer_count <= 0
            or kv_head_count <= 0
            or key_head_dim <= 0
            or value_head_dim <= 0
            or rotary_dim <= 0
            or rotary_dim > key_head_dim
        ):
            reason = "unsupported_model_dimensions"

        return cls(
            supported=reason is None,
            reason=reason,
            model_type=model_type,
            attention_arch=attention_arch,
            cache_layout="separate_kv_token_major",
            layer_count=layer_count,
            kv_head_count=kv_head_count,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            rope=RoPEConfig(
                rotary_dim=rotary_dim,
                base=rope_base,
                is_neox_style=is_neox_style,
            ),
            tp_size=tp_size,
            pp_size=pp_size,
        )

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.model_type}:{self.attention_arch}:{self.cache_layout}:"
            f"{self.layer_count}:{self.kv_head_count}:{self.key_head_dim}:"
            f"{self.value_head_dim}:{self.rope.rotary_dim}:"
            f"{self.rope.base}:{int(self.rope.is_neox_style)}:"
            f"tp{self.tp_size}:pp{self.pp_size}"
        )

    @property
    def rope_fingerprint(self) -> str:
        return (
            f"rope:{self.rope.rotary_dim}:{self.rope.base}:"
            f"{int(self.rope.is_neox_style)}"
        )

    def guard_kvcache(self, kvcache: Any) -> str | None:
        if not self.supported:
            return self.reason or "unsupported_runtime"
        if self.cache_layout != "separate_kv_token_major":
            return "unsupported_cache_layout"
        if type(kvcache).__name__.startswith("NoOp"):
            return "unsupported_cache_layout"
        physical_layout = str(getattr(kvcache, "kv_cache_layout", "nhd")).lower()
        if physical_layout not in ("nhd", "none"):
            return "unsupported_cache_layout"
        if bool(getattr(kvcache, "is_quantized_kv_cache", False)):
            return "unsupported_cache_dtype"
        cache_dtype = getattr(kvcache, "dtype", None)
        store_dtype = getattr(kvcache, "store_dtype", cache_dtype)
        if cache_dtype is not None and store_dtype != cache_dtype:
            return "unsupported_cache_dtype"
        if not hasattr(kvcache, "get_key_buffer") or not hasattr(
            kvcache,
            "get_value_buffer",
        ):
            return "unsupported_cache_layout"
        if int(getattr(kvcache, "layer_num", -1)) != self.layer_count:
            return "layer_count_mismatch"
        start_layer = int(getattr(kvcache, "start_layer", 0) or 0)
        end_layer_raw = getattr(kvcache, "end_layer", self.layer_count - 1)
        end_layer = (
            self.layer_count - 1 if end_layer_raw is None else int(end_layer_raw)
        )
        if start_layer != 0 or end_layer not in (
            self.layer_count - 1,
            self.layer_count,
        ):
            return "unsupported_pipeline_parallel"

        allowed_dtypes = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        cache_device = None
        for layer_id in range(self.layer_count):
            key = kvcache.get_key_buffer(layer_id)
            value = kvcache.get_value_buffer(layer_id)
            if (
                not isinstance(key, torch.Tensor)
                or not isinstance(value, torch.Tensor)
                or key is value
            ):
                return "unsupported_cache_layout"
            if key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
                return "unsupported_cache_layout"
            if key.ndim != 3 or value.ndim != 3:
                return "unsupported_cache_layout"
            if key.device != value.device:
                return "unsupported_cache_layout"
            if cache_device is None:
                cache_device = key.device
            elif key.device != cache_device:
                return "unsupported_cache_layout"
            if key.shape[0] != value.shape[0]:
                return "cache_capacity_mismatch"
            if key.shape[1] != self.kv_head_count:
                return "kv_head_count_mismatch"
            if value.shape[1] != self.kv_head_count:
                return "kv_head_count_mismatch"
            if key.shape[-1] != self.key_head_dim:
                return "key_head_dim_mismatch"
            if value.shape[-1] != self.value_head_dim:
                return "value_head_dim_mismatch"
            if key.dtype not in allowed_dtypes or value.dtype not in allowed_dtypes:
                return "unsupported_cache_dtype"
            if key.dtype != value.dtype:
                return "cache_dtype_mismatch"
        return None

    @staticmethod
    def guard_declared_dtype(kvcache: Any, declared: str) -> str | None:
        normalized = declared.strip().lower().replace("torch.", "")
        aliases = {
            "fp16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
            "float": "float32",
            "fp64": "float64",
            "double": "float64",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized == "auto":
            return "cache_dtype_unspecified"
        actual = {
            str(buffer.dtype).replace("torch.", "")
            for layer_id in range(int(kvcache.layer_num))
            for buffer in (
                kvcache.get_key_buffer(layer_id),
                kvcache.get_value_buffer(layer_id),
            )
        }
        if actual != {normalized}:
            return "cache_dtype_mismatch"
        return None


def _rotate_half(tensor: torch.Tensor, is_neox_style: bool) -> torch.Tensor:
    if is_neox_style:
        first = tensor[..., : tensor.shape[-1] // 2]
        second = tensor[..., tensor.shape[-1] // 2 :]
        return torch.cat((-second, first), dim=-1)
    paired = tensor.reshape(*tensor.shape[:-1], -1, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.flatten(-2)


def rotate_key_positions(
    keys: torch.Tensor,
    positions: Sequence[int] | torch.Tensor,
    rope: RoPEConfig,
) -> torch.Tensor:
    if keys.ndim < 2:
        raise ValueError("keys must have a token and head dimension")
    position_tensor = torch.as_tensor(
        positions,
        dtype=torch.float32,
        device=keys.device,
    ).reshape(-1)
    if position_tensor.numel() != keys.shape[0]:
        raise ValueError("position count must equal key token count")
    if rope.rotary_dim <= 0 or rope.rotary_dim > keys.shape[-1]:
        raise ValueError("invalid rotary dimension for key tensor")

    inverse_frequency = 1.0 / (
        rope.base
        ** (
            torch.arange(
                0,
                rope.rotary_dim,
                2,
                dtype=torch.float32,
                device=keys.device,
            )
            / rope.rotary_dim
        )
    )
    frequencies = torch.einsum(
        "i,j->ij",
        position_tensor,
        inverse_frequency,
    )
    if rope.is_neox_style:
        cosine = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1)
        sine = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1)
    else:
        cosine = frequencies.cos().repeat_interleave(2, dim=-1)
        sine = frequencies.sin().repeat_interleave(2, dim=-1)
    broadcast_shape = (keys.shape[0],) + (1,) * (keys.ndim - 2) + (rope.rotary_dim,)
    cosine = cosine.reshape(broadcast_shape)
    sine = sine.reshape(broadcast_shape)

    rotary = keys[..., : rope.rotary_dim].float()
    rotated = (
        rotary * cosine
        + _rotate_half(
            rotary,
            rope.is_neox_style,
        )
        * sine
    )
    result = keys.clone()
    result[..., : rope.rotary_dim] = rotated.to(keys.dtype)
    return result


def normalize_key_positions(
    keys: torch.Tensor,
    positions: Sequence[int] | torch.Tensor,
    rope: RoPEConfig,
) -> torch.Tensor:
    position_tensor = torch.as_tensor(positions, dtype=torch.int64)
    return rotate_key_positions(keys, -position_tensor, rope)


def relocate_key_positions(
    keys: torch.Tensor,
    source_positions: Sequence[int] | torch.Tensor,
    target_positions: Sequence[int] | torch.Tensor,
    rope: RoPEConfig,
) -> torch.Tensor:
    normalized = normalize_key_positions(keys, source_positions, rope)
    return rotate_key_positions(normalized, target_positions, rope)


def validate_interpolation_weights(
    weights: Sequence[float] | torch.Tensor,
    expected_count: int,
    *,
    atol: float = 1e-5,
) -> torch.Tensor:
    tensor = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
    if tensor.numel() != expected_count or expected_count <= 0:
        raise KVCOMMInvariantError("anchor weight count mismatch")
    if not torch.isfinite(tensor).all():
        raise KVCOMMInvariantError("anchor weights must be finite")
    if bool((tensor < 0).any()):
        raise KVCOMMInvariantError("anchor weights must be non-negative")
    if not torch.isclose(
        tensor.sum(),
        tensor.new_tensor(1.0),
        atol=atol,
        rtol=0,
    ):
        raise KVCOMMInvariantError("anchor weights must sum to one")
    return tensor


@dataclass(frozen=True)
class KVCOMMWeightResult:
    weights: tuple[float, ...]
    entropy: float


def compute_interpolation_weights(
    target_embedding: torch.Tensor,
    anchor_embeddings: Sequence[torch.Tensor],
    *,
    temperature: float,
) -> KVCOMMWeightResult:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if not anchor_embeddings:
        raise KVCOMMInvariantError("at least one anchor is required")
    target = target_embedding.detach().float().reshape(-1)
    anchors = [
        embedding.detach().float().reshape(-1) for embedding in anchor_embeddings
    ]
    if any(anchor.shape != target.shape for anchor in anchors):
        raise KVCOMMInvariantError("anchor embedding shape mismatch")
    stacked = torch.stack(
        [anchor.to(device=target.device) for anchor in anchors],
        dim=0,
    )
    distances = torch.linalg.vector_norm(stacked - target.unsqueeze(0), dim=1)
    if not torch.isfinite(distances).all():
        raise KVCOMMInvariantError("anchor distances must be finite")
    weights = torch.softmax(-distances / temperature, dim=0)
    validate_interpolation_weights(weights, len(anchor_embeddings))
    entropy = float(
        (
            -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum()
        ).item()
    )
    return KVCOMMWeightResult(
        weights=tuple(float(value) for value in weights.cpu()),
        entropy=entropy,
    )


@dataclass(frozen=True)
class KVCOMMProvenance:
    model_fingerprint: str
    tokenizer_fingerprint: str
    source_fingerprint: str
    cache_dtype: str
    rope_fingerprint: str
    layout_fingerprint: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.model_fingerprint,
                self.tokenizer_fingerprint,
                self.source_fingerprint,
                self.cache_dtype,
                self.rope_fingerprint,
                self.layout_fingerprint,
            )
        ):
            raise ValueError("KVCOMM provenance fields must be non-empty")

    def compatible_with(self, other: "KVCOMMProvenance") -> bool:
        return (
            self.model_fingerprint == other.model_fingerprint
            and self.tokenizer_fingerprint == other.tokenizer_fingerprint
            and self.cache_dtype == other.cache_dtype
            and self.rope_fingerprint == other.rope_fingerprint
            and self.layout_fingerprint == other.layout_fingerprint
        )


@dataclass(frozen=True)
class KVCOMMSegmentDescriptor:
    segment_index: int
    placeholder_id: str
    role: KVCOMMSegmentRole
    source_fingerprint: str

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if not self.placeholder_id or not self.source_fingerprint:
            raise ValueError("placeholder_id and source_fingerprint must be non-empty")


@dataclass(frozen=True)
class KVCOMMRequestSpec:
    action: KVCOMMAction
    agent_id: str
    tokenizer_fingerprint: str
    template_fingerprint: str
    context_fingerprint: str
    segments: tuple[KVCOMMSegmentDescriptor, ...]
    entropy_threshold: float = 0.3
    temperature: float = 1.0
    max_anchors: int = 20
    prune_window: int = 5
    min_anchors: int = 2

    @classmethod
    def from_metadata(cls, metadata: Any) -> "KVCOMMRequestSpec":
        params = metadata.plugin_params
        default_action = (
            KVCOMMAction.BASE.value
            if str(getattr(metadata.operation, "value", metadata.operation))
            == "register"
            else KVCOMMAction.REUSE.value
        )
        action = KVCOMMAction(str(params.get("action", default_action)))
        operation = str(getattr(metadata.operation, "value", metadata.operation))
        if action in (KVCOMMAction.BASE, KVCOMMAction.ANCHOR):
            if operation != "register":
                raise ValueError(
                    "KVCOMM base/anchor actions require register operation"
                )
        elif operation != "reuse":
            raise ValueError("KVCOMM reuse action requires reuse operation")

        raw_descriptors = params.get("segments")
        if not isinstance(raw_descriptors, Sequence) or isinstance(
            raw_descriptors,
            (str, bytes),
        ):
            raise ValueError("KVCOMM plugin_params.segments must be an array")
        descriptors = []
        for raw in raw_descriptors:
            if not isinstance(raw, Mapping):
                raise ValueError("KVCOMM segment descriptor must be an object")
            index = int(raw["segment_index"])
            if index < 0 or index >= len(metadata.segments):
                raise ValueError("KVCOMM segment_index is out of range")
            source_fingerprint = str(
                raw.get(
                    "source_fingerprint",
                    metadata.segments[index].content_hash,
                )
            )
            descriptors.append(
                KVCOMMSegmentDescriptor(
                    segment_index=index,
                    placeholder_id=str(raw["placeholder_id"]),
                    role=KVCOMMSegmentRole(str(raw["role"])),
                    source_fingerprint=source_fingerprint,
                )
            )
        if len(descriptors) != len(metadata.segments):
            raise ValueError("KVCOMM must describe every approximate KV segment")
        indices = [descriptor.segment_index for descriptor in descriptors]
        if len(set(indices)) != len(indices):
            raise ValueError("KVCOMM segment descriptors must be unique")

        spec = cls(
            action=action,
            agent_id=str(params.get("agent_id", "")),
            tokenizer_fingerprint=str(params.get("tokenizer_fingerprint", "")),
            template_fingerprint=str(params.get("template_fingerprint", "")),
            context_fingerprint=str(params.get("context_fingerprint", "")),
            segments=tuple(descriptors),
            entropy_threshold=float(params.get("entropy_threshold", 0.3)),
            temperature=float(params.get("temperature", 1.0)),
            max_anchors=int(params.get("max_anchors", 20)),
            prune_window=int(params.get("prune_window", 5)),
            min_anchors=int(params.get("min_anchors", 2)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not all(
            (
                self.agent_id,
                self.tokenizer_fingerprint,
                self.template_fingerprint,
                self.context_fingerprint,
            )
        ):
            raise ValueError("KVCOMM request provenance must be non-empty")
        if not 0 <= self.entropy_threshold <= 1:
            raise ValueError("entropy_threshold must be in [0, 1]")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        if self.max_anchors <= 0 or self.prune_window <= 0:
            raise ValueError("anchor capacity values must be positive")
        if self.min_anchors < 2:
            raise ValueError("KVCOMM requires at least two anchors")
        if self.min_anchors > self.max_anchors:
            raise ValueError("min_anchors cannot exceed max_anchors")


@dataclass(frozen=True)
class KVCOMMObservedSegment:
    descriptor: KVCOMMSegmentDescriptor
    key: KVSegmentKey
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    indices: torch.Tensor = field(compare=False, repr=False)
    handle: KVSegmentHandle | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.positions):
            raise ValueError("token and position counts must match")
        if len(self.token_ids) != self.key.token_count:
            raise ValueError("observed segment token count mismatch")
        if self.indices.numel() != self.key.token_count:
            raise ValueError("observed segment index count mismatch")


@dataclass(frozen=True)
class KVCOMMBaseRecord:
    descriptor: KVCOMMSegmentDescriptor
    key: KVSegmentKey
    handle: KVSegmentHandle
    positions: tuple[int, ...]
    provenance: KVCOMMProvenance
    embedding: torch.Tensor = field(compare=False, repr=False)


@dataclass(frozen=True)
class KVCOMMDeltaRecord:
    role: KVCOMMSegmentRole
    handle: KVSegmentHandle
    base_key: KVSegmentKey
    base_generation: int
    provenance: KVCOMMProvenance
    layer_count: int


@dataclass
class KVCOMMAnchor:
    anchor_id: str
    placeholder_id: str
    agent_id: str
    template_fingerprint: str
    context_fingerprint: str
    provenance: KVCOMMProvenance
    placeholder_base_handle: KVSegmentHandle
    neighbor_base_handle: KVSegmentHandle | None
    placeholder_delta: KVCOMMDeltaRecord
    neighbor_delta: KVCOMMDeltaRecord | None
    embedding: torch.Tensor = field(compare=False, repr=False)
    generation: int = 0
    created_order: int = 0
    access_count: int = 0

    @property
    def placeholder_length(self) -> int:
        return self.placeholder_delta.handle.key.token_count

    @property
    def neighbor_length(self) -> int:
        if self.neighbor_delta is None:
            return 0
        return self.neighbor_delta.handle.key.token_count


@dataclass(frozen=True)
class KVCOMMReconstructionSlice:
    role: KVCOMMSegmentRole
    base: KVCOMMBaseRecord
    deltas: tuple[KVCOMMDeltaRecord, ...]
    weights: tuple[float, ...]
    source_offset: int
    target_start: int
    target_position: int
    length: int


@dataclass(frozen=True)
class KVCOMMReconstructionPlan:
    exact_prefix_length: int
    restore_length: int
    slices: tuple[KVCOMMReconstructionSlice, ...]
    pool_generations: tuple[tuple[tuple[str, ...], int], ...]
    capability_fingerprint: str


def make_kvcomm_segment_key(
    *,
    tokens: Sequence[int],
    content_hash: str,
    model_fingerprint: str,
    cache_dtype: str,
    kind: SegmentKind,
) -> KVSegmentKey:
    token_tuple = tuple(int(token) for token in tokens)
    return KVSegmentKey(
        content_hash=content_hash,
        token_hash=token_ids_hash(token_tuple),
        token_count=len(token_tuple),
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        kind=kind,
    )


class KVCOMMRecoveryPlugin:
    name = "kvcomm"

    def __init__(self) -> None:
        self._capabilities: KVCOMMRuntimeCapabilities | None = None
        self._bases: dict[
            tuple[str, KVCOMMSegmentRole, KVSegmentKey],
            KVCOMMBaseRecord,
        ] = {}
        self._pools: dict[
            tuple[str, ...],
            OrderedDict[str, KVCOMMAnchor],
        ] = {}
        self._pool_generations: dict[tuple[str, ...], int] = {}
        self._sequence = 0
        self._lock = threading.RLock()

    def bind_runtime_capabilities(
        self,
        capabilities: KVCOMMRuntimeCapabilities | None,
    ) -> None:
        self._capabilities = capabilities

    def reset(self) -> None:
        with self._lock:
            self._bases.clear()
            self._pools.clear()
            self._pool_generations.clear()
            self._sequence = 0

    def scheduler_metadata(
        self,
        context: RecoveryRequestContext,
    ) -> tuple[SchedulerMetadata, ...]:
        del context
        return ()

    def _dense_plan(
        self,
        context: RecoveryRequestContext,
        reason: str,
    ) -> KVReusePlan:
        remaining = context.target_token_ids[
            context.exact_prefix_length : max(
                context.exact_prefix_length,
                len(context.target_token_ids) - 1,
            )
        ]
        dense_ranges = (DenseRange(0, len(remaining), reason),) if remaining else ()
        return KVReusePlan(
            target_token_ids=tuple(remaining),
            recovery_mode=RecoveryMode.DENSE,
            dense_ranges=dense_ranges,
            require_full_coverage=bool(remaining),
        )

    def _require_capabilities(self) -> KVCOMMRuntimeCapabilities:
        capabilities = self._capabilities
        if capabilities is None:
            raise KVCOMMCapabilityError("capability_unavailable")
        if not capabilities.supported:
            raise KVCOMMCapabilityError(capabilities.reason or "unsupported_runtime")
        return capabilities

    @staticmethod
    def _provenance(
        metadata: Any,
        spec: KVCOMMRequestSpec,
        descriptor: KVCOMMSegmentDescriptor,
        capabilities: KVCOMMRuntimeCapabilities,
    ) -> KVCOMMProvenance:
        return KVCOMMProvenance(
            model_fingerprint=metadata.model_fingerprint,
            tokenizer_fingerprint=spec.tokenizer_fingerprint,
            source_fingerprint=descriptor.source_fingerprint,
            cache_dtype=metadata.cache_dtype,
            rope_fingerprint=capabilities.rope_fingerprint,
            layout_fingerprint=capabilities.fingerprint,
        )

    @staticmethod
    def _base_identity(
        descriptor: KVCOMMSegmentDescriptor,
        key: KVSegmentKey,
    ) -> tuple[str, KVCOMMSegmentRole, KVSegmentKey]:
        return descriptor.placeholder_id, descriptor.role, key

    @staticmethod
    def _pool_key(
        spec: KVCOMMRequestSpec,
        placeholder_id: str,
        provenance: KVCOMMProvenance,
    ) -> tuple[str, ...]:
        return (
            placeholder_id,
            spec.agent_id,
            spec.template_fingerprint,
            provenance.model_fingerprint,
            provenance.tokenizer_fingerprint,
            provenance.cache_dtype,
            provenance.rope_fingerprint,
            provenance.layout_fingerprint,
        )

    @staticmethod
    def _embedding_from_base(
        record: KVCOMMBaseRecord,
        allocator: Any,
        capabilities: KVCOMMRuntimeCapabilities,
    ) -> torch.Tensor:
        if not isinstance(record.handle.backend_ref, DeviceKVRef):
            raise KVCOMMInvariantError("KVCOMM base must be device-resident")
        kvcache = allocator.get_kvcache()
        indices = record.handle.backend_ref.indices.long()
        layer_embeddings = []
        for layer_id in range(capabilities.layer_count):
            keys = normalize_key_positions(
                kvcache.get_key_buffer(layer_id)[indices].float(),
                record.positions,
                capabilities.rope,
            )
            values = kvcache.get_value_buffer(layer_id)[indices].float()
            layer_embeddings.append(
                torch.cat(
                    (
                        keys.mean(dim=1),
                        values.mean(dim=1),
                    ),
                    dim=-1,
                )
            )
        embedding = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return embedding.detach().clone()

    def register_base_segments(
        self,
        *,
        metadata: Any,
        spec: KVCOMMRequestSpec,
        observed: Sequence[KVCOMMObservedSegment],
        store: ApproxKVSegmentStore,
        allocator: Any,
    ) -> int:
        capabilities = self._require_capabilities()
        guard_reason = capabilities.guard_kvcache(allocator.get_kvcache())
        if guard_reason is not None:
            raise KVCOMMCapabilityError(guard_reason)
        dtype_reason = capabilities.guard_declared_dtype(
            allocator.get_kvcache(),
            metadata.cache_dtype,
        )
        if dtype_reason is not None:
            raise KVCOMMCapabilityError(dtype_reason)
        registered = 0
        records = []
        for segment in observed:
            handle = segment.handle
            if (
                handle is None
                or handle.residency != ResidencyTier.DEVICE
                or not store.is_current(handle)
                or not isinstance(handle.backend_ref, DeviceKVRef)
            ):
                raise KVCOMMInvariantError(
                    "base registration requires a current device handle"
                )
            provenance = self._provenance(
                metadata,
                spec,
                segment.descriptor,
                capabilities,
            )
            provisional = KVCOMMBaseRecord(
                descriptor=segment.descriptor,
                key=segment.key,
                handle=handle,
                positions=segment.positions,
                provenance=provenance,
                embedding=torch.empty(0),
            )
            record = KVCOMMBaseRecord(
                descriptor=segment.descriptor,
                key=segment.key,
                handle=handle,
                positions=segment.positions,
                provenance=provenance,
                embedding=self._embedding_from_base(
                    provisional,
                    allocator,
                    capabilities,
                ),
            )
            records.append(record)
            registered += handle.key.token_count

        with self._lock:
            for record in records:
                self._bases[self._base_identity(record.descriptor, record.key)] = record
        return registered

    def _lookup_base(
        self,
        descriptor: KVCOMMSegmentDescriptor,
        key: KVSegmentKey,
        provenance: KVCOMMProvenance,
        store: ApproxKVSegmentStore,
    ) -> tuple[KVCOMMBaseRecord | None, str | None]:
        with self._lock:
            record = self._bases.get(self._base_identity(descriptor, key))
        if record is None:
            return None, "base_missing"
        if not record.provenance.compatible_with(provenance):
            return None, "base_provenance_mismatch"
        if record.provenance.source_fingerprint != provenance.source_fingerprint:
            return None, "base_source_mismatch"
        if not store.is_current(record.handle):
            identity = self._base_identity(descriptor, key)
            with self._lock:
                if self._bases.get(identity) is record:
                    self._bases.pop(identity, None)
            return None, "stale_base_generation"
        if record.handle.residency != ResidencyTier.DEVICE:
            return None, "base_not_device_resident"
        return record, None

    @staticmethod
    def _group_descriptors(
        spec: KVCOMMRequestSpec,
    ) -> tuple[
        tuple[
            KVCOMMSegmentDescriptor,
            KVCOMMSegmentDescriptor | None,
        ],
        ...,
    ]:
        groups: OrderedDict[
            str,
            dict[KVCOMMSegmentRole, KVCOMMSegmentDescriptor],
        ] = OrderedDict()
        for descriptor in spec.segments:
            roles = groups.setdefault(descriptor.placeholder_id, {})
            if descriptor.role in roles:
                raise ValueError("KVCOMM placeholder group contains duplicate roles")
            roles[descriptor.role] = descriptor
        result = []
        for roles in groups.values():
            placeholder = roles.get(KVCOMMSegmentRole.PLACEHOLDER)
            if placeholder is None:
                raise ValueError("KVCOMM placeholder group lacks a placeholder segment")
            result.append(
                (
                    placeholder,
                    roles.get(KVCOMMSegmentRole.NEIGHBOR),
                )
            )
        return tuple(result)

    def _valid_anchor(
        self,
        anchor: KVCOMMAnchor,
        *,
        placeholder_length: int,
        neighbor: KVCOMMBaseRecord | None,
        store: ApproxKVSegmentStore,
    ) -> bool:
        if anchor.placeholder_length < placeholder_length:
            return False
        if not store.is_current(anchor.placeholder_base_handle):
            return False
        if not store.is_current(anchor.placeholder_delta.handle):
            return False
        if (
            anchor.placeholder_delta.base_key != anchor.placeholder_base_handle.key
            or anchor.placeholder_delta.base_generation
            != anchor.placeholder_base_handle.generation
        ):
            return False
        if anchor.placeholder_delta.handle.residency != ResidencyTier.DEVICE:
            return False
        if (
            not anchor.placeholder_delta.provenance.compatible_with(anchor.provenance)
            or anchor.placeholder_delta.provenance.source_fingerprint
            != anchor.provenance.source_fingerprint
        ):
            return False
        if neighbor is None:
            return anchor.neighbor_delta is None and anchor.neighbor_base_handle is None
        if (
            anchor.neighbor_delta is None
            or anchor.neighbor_base_handle is None
            or anchor.neighbor_length != neighbor.key.token_count
            or anchor.neighbor_base_handle.key != neighbor.handle.key
            or anchor.neighbor_delta.base_key != anchor.neighbor_base_handle.key
            or anchor.neighbor_delta.base_generation
            != anchor.neighbor_base_handle.generation
            or not store.is_current(anchor.neighbor_base_handle)
            or not store.is_current(anchor.neighbor_delta.handle)
            or anchor.neighbor_delta.handle.residency != ResidencyTier.DEVICE
            or not anchor.neighbor_delta.provenance.compatible_with(anchor.provenance)
            or anchor.neighbor_delta.provenance.source_fingerprint
            != neighbor.provenance.source_fingerprint
        ):
            return False
        return True

    def _select_anchors(
        self,
        *,
        spec: KVCOMMRequestSpec,
        placeholder: KVCOMMBaseRecord,
        neighbor: KVCOMMBaseRecord | None,
        store: ApproxKVSegmentStore,
    ) -> tuple[
        tuple[KVCOMMAnchor, ...] | None,
        tuple[float, ...] | None,
        tuple[str, ...],
        int,
        str | None,
    ]:
        pool_key = self._pool_key(
            spec,
            placeholder.descriptor.placeholder_id,
            placeholder.provenance,
        )
        with self._lock:
            pool = self._pools.get(pool_key)
            pool_generation = self._pool_generations.get(pool_key, 0)
            candidates = tuple(pool.values()) if pool is not None else ()
        if not candidates:
            return None, None, pool_key, pool_generation, "anchor_pool_empty"
        if placeholder.key.token_count > max(
            anchor.placeholder_length for anchor in candidates
        ):
            return None, None, pool_key, pool_generation, "anchor_length_uncovered"

        valid = tuple(
            anchor
            for anchor in candidates
            if self._valid_anchor(
                anchor,
                placeholder_length=placeholder.key.token_count,
                neighbor=neighbor,
                store=store,
            )
        )
        if len(valid) < spec.min_anchors:
            return (
                None,
                None,
                pool_key,
                pool_generation,
                "insufficient_compatible_anchors",
            )
        weight_result = compute_interpolation_weights(
            placeholder.embedding,
            [anchor.embedding[: placeholder.key.token_count] for anchor in valid],
            temperature=spec.temperature,
        )
        entropy_limit = spec.entropy_threshold * math.log(len(valid))
        if weight_result.entropy > entropy_limit:
            return (
                None,
                None,
                pool_key,
                pool_generation,
                "anchor_entropy_gate",
            )

        with self._lock:
            current_pool = self._pools.get(pool_key)
            if current_pool is None:
                return (
                    None,
                    None,
                    pool_key,
                    pool_generation,
                    "stale_anchor_pool",
                )
            for anchor in valid:
                current = current_pool.get(anchor.anchor_id)
                if current is None or current.generation != anchor.generation:
                    return (
                        None,
                        None,
                        pool_key,
                        pool_generation,
                        "stale_anchor_generation",
                    )
                current.access_count += 1
        return (
            valid,
            weight_result.weights,
            pool_key,
            pool_generation,
            None,
        )

    def build_plan(
        self,
        context: RecoveryRequestContext,
        store: ApproxKVSegmentStore,
    ) -> KVReusePlan:
        metadata = context.custom_metadata.get("approx_kv_metadata")
        if metadata is None:
            return self._dense_plan(context, "kvcomm_metadata_missing")
        try:
            spec = KVCOMMRequestSpec.from_metadata(metadata)
            if spec.action != KVCOMMAction.REUSE:
                return self._dense_plan(context, "kvcomm_not_reuse_action")
            capabilities = self._require_capabilities()
            groups = self._group_descriptors(spec)
        except (KVCOMMCapabilityError, ValueError) as exc:
            reason = getattr(exc, "reason", "kvcomm_metadata_invalid")
            return self._dense_plan(context, str(reason))

        exact_length = context.exact_prefix_length
        reusable_limit = len(context.target_token_ids) - 1
        if exact_length >= reusable_limit:
            return KVReusePlan(target_token_ids=())

        group_data = []
        for placeholder_descriptor, neighbor_descriptor in groups:
            placeholder_segment = metadata.segments[
                placeholder_descriptor.segment_index
            ]
            neighbor_segment = (
                metadata.segments[neighbor_descriptor.segment_index]
                if neighbor_descriptor is not None
                else None
            )
            group_data.append(
                (
                    placeholder_segment.target_start,
                    placeholder_descriptor,
                    placeholder_segment,
                    neighbor_descriptor,
                    neighbor_segment,
                )
            )
        group_data.sort(key=lambda item: item[0])

        next_target = exact_length
        slices = []
        pool_generations = []
        for (
            _,
            placeholder_descriptor,
            placeholder_segment,
            neighbor_descriptor,
            neighbor_segment,
        ) in group_data:
            group_start = placeholder_segment.target_start
            group_end = placeholder_segment.target_end
            if neighbor_segment is not None:
                if neighbor_segment.target_start != group_end:
                    return self._dense_plan(
                        context,
                        "neighbor_not_adjacent",
                    )
                group_end = neighbor_segment.target_end
            if group_end <= exact_length:
                continue
            if next_target >= reusable_limit:
                break
            if group_start > next_target:
                return self._dense_plan(context, "kvcomm_prefix_gap")
            if group_start < next_target and next_target != exact_length:
                return self._dense_plan(context, "kvcomm_group_overlap")

            placeholder_tokens = tuple(
                int(token)
                for token in context.target_token_ids[
                    placeholder_segment.target_start : placeholder_segment.target_end
                ]
            )
            placeholder_key = make_kvcomm_segment_key(
                tokens=placeholder_tokens,
                content_hash=placeholder_segment.content_hash,
                model_fingerprint=metadata.model_fingerprint,
                cache_dtype=metadata.cache_dtype,
                kind=SegmentKind.KVCOMM_BASE,
            )
            placeholder_provenance = self._provenance(
                metadata,
                spec,
                placeholder_descriptor,
                capabilities,
            )
            placeholder_base, reason = self._lookup_base(
                placeholder_descriptor,
                placeholder_key,
                placeholder_provenance,
                store,
            )
            if placeholder_base is None:
                return self._dense_plan(context, reason or "base_missing")

            neighbor_base = None
            if neighbor_descriptor is not None and neighbor_segment is not None:
                neighbor_tokens = tuple(
                    int(token)
                    for token in context.target_token_ids[
                        neighbor_segment.target_start : neighbor_segment.target_end
                    ]
                )
                neighbor_key = make_kvcomm_segment_key(
                    tokens=neighbor_tokens,
                    content_hash=neighbor_segment.content_hash,
                    model_fingerprint=metadata.model_fingerprint,
                    cache_dtype=metadata.cache_dtype,
                    kind=SegmentKind.KVCOMM_BASE,
                )
                neighbor_provenance = self._provenance(
                    metadata,
                    spec,
                    neighbor_descriptor,
                    capabilities,
                )
                neighbor_base, reason = self._lookup_base(
                    neighbor_descriptor,
                    neighbor_key,
                    neighbor_provenance,
                    store,
                )
                if neighbor_base is None:
                    return self._dense_plan(
                        context,
                        reason or "neighbor_base_missing",
                    )

            (
                anchors,
                weights,
                pool_key,
                pool_generation,
                reason,
            ) = self._select_anchors(
                spec=spec,
                placeholder=placeholder_base,
                neighbor=neighbor_base,
                store=store,
            )
            if anchors is None or weights is None:
                return self._dense_plan(
                    context,
                    reason or "anchor_selection_failed",
                )
            pool_generations.append((pool_key, pool_generation))

            for role, descriptor, segment, base in (
                (
                    KVCOMMSegmentRole.PLACEHOLDER,
                    placeholder_descriptor,
                    placeholder_segment,
                    placeholder_base,
                ),
                (
                    KVCOMMSegmentRole.NEIGHBOR,
                    neighbor_descriptor,
                    neighbor_segment,
                    neighbor_base,
                ),
            ):
                if descriptor is None or segment is None or base is None:
                    continue
                overlap_start = max(segment.target_start, exact_length)
                overlap_end = min(segment.target_end, reusable_limit)
                if overlap_end <= overlap_start:
                    continue
                deltas = tuple(
                    (
                        anchor.placeholder_delta
                        if role == KVCOMMSegmentRole.PLACEHOLDER
                        else anchor.neighbor_delta
                    )
                    for anchor in anchors
                )
                if any(delta is None for delta in deltas):
                    return self._dense_plan(
                        context,
                        "neighbor_delta_missing",
                    )
                slices.append(
                    KVCOMMReconstructionSlice(
                        role=role,
                        base=base,
                        deltas=tuple(deltas),
                        weights=weights,
                        source_offset=overlap_start - segment.target_start,
                        target_start=overlap_start - exact_length,
                        target_position=overlap_start,
                        length=overlap_end - overlap_start,
                    )
                )
            next_target = min(group_end, reusable_limit)

        restore_length = next_target - exact_length
        if restore_length <= 0 or not slices:
            return self._dense_plan(context, "kvcomm_no_contiguous_span")
        occupied = set()
        for reconstruction_slice in slices:
            occupied.update(
                range(
                    reconstruction_slice.target_start,
                    reconstruction_slice.target_start + reconstruction_slice.length,
                )
            )
        if occupied != set(range(restore_length)):
            return self._dense_plan(context, "kvcomm_prefix_gap")

        data = KVCOMMReconstructionPlan(
            exact_prefix_length=exact_length,
            restore_length=restore_length,
            slices=tuple(slices),
            pool_generations=tuple(pool_generations),
            capability_fingerprint=capabilities.fingerprint,
        )
        return KVReusePlan(
            target_token_ids=tuple(
                context.target_token_ids[exact_length : exact_length + restore_length]
            ),
            recovery_mode=RecoveryMode.KVCOMM,
            require_full_coverage=True,
            plugin_data=data,
        )

    def validate_plan(
        self,
        plan: KVCOMMReconstructionPlan,
        store: ApproxKVSegmentStore,
    ) -> str | None:
        try:
            capabilities = self._require_capabilities()
        except KVCOMMCapabilityError as exc:
            return exc.reason
        if plan.capability_fingerprint != capabilities.fingerprint:
            return "capability_generation_mismatch"
        with self._lock:
            for pool_key, generation in plan.pool_generations:
                if self._pool_generations.get(pool_key, 0) != generation:
                    return "stale_anchor_generation"
        for reconstruction_slice in plan.slices:
            if not store.is_current(reconstruction_slice.base.handle):
                return "stale_base_generation"
            if (
                reconstruction_slice.base.handle.residency != ResidencyTier.DEVICE
                or not isinstance(
                    reconstruction_slice.base.handle.backend_ref,
                    DeviceKVRef,
                )
            ):
                return "base_not_device_resident"
            if (
                reconstruction_slice.base.handle.key != reconstruction_slice.base.key
                or len(reconstruction_slice.base.positions)
                != reconstruction_slice.base.key.token_count
                or reconstruction_slice.target_position
                != plan.exact_prefix_length + reconstruction_slice.target_start
                or reconstruction_slice.source_offset < 0
                or reconstruction_slice.length <= 0
                or reconstruction_slice.source_offset + reconstruction_slice.length
                > reconstruction_slice.base.key.token_count
            ):
                return "invalid_reconstruction_slice"
            try:
                validate_interpolation_weights(
                    reconstruction_slice.weights,
                    len(reconstruction_slice.deltas),
                )
            except KVCOMMInvariantError:
                return "invalid_anchor_weights"
            for delta in reconstruction_slice.deltas:
                if not store.is_current(delta.handle):
                    return "stale_delta_generation"
                if delta.handle.residency != ResidencyTier.DEVICE:
                    return "delta_not_device_resident"
                delta_base = store.lookup(delta.base_key)
                if (
                    delta_base is None
                    or delta_base.generation != delta.base_generation
                    or delta_base.residency != ResidencyTier.DEVICE
                    or not isinstance(delta_base.backend_ref, DeviceKVRef)
                ):
                    return "stale_anchor_base_generation"
                if (
                    delta.role != reconstruction_slice.role
                    or not delta.provenance.compatible_with(
                        reconstruction_slice.base.provenance
                    )
                    or (
                        reconstruction_slice.role == KVCOMMSegmentRole.NEIGHBOR
                        and delta.provenance.source_fingerprint
                        != reconstruction_slice.base.provenance.source_fingerprint
                    )
                    or delta.layer_count != capabilities.layer_count
                    or delta.handle.key.token_count
                    < reconstruction_slice.source_offset + reconstruction_slice.length
                ):
                    return "delta_provenance_mismatch"
        return None

    @staticmethod
    def _anchor_id(
        spec: KVCOMMRequestSpec,
        placeholder: KVCOMMObservedSegment,
    ) -> str:
        digest = hashlib.sha256()
        for value in (
            placeholder.descriptor.placeholder_id,
            spec.agent_id,
            spec.template_fingerprint,
            spec.context_fingerprint,
            placeholder.key.content_hash,
            placeholder.key.token_hash,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _materialize_delta(
        self,
        *,
        anchor_id: str,
        update_serial: int,
        observed: KVCOMMObservedSegment,
        base: KVCOMMBaseRecord,
        provenance: KVCOMMProvenance,
        store: ApproxKVSegmentStore,
        allocator: Any,
        capabilities: KVCOMMRuntimeCapabilities,
    ) -> KVCOMMDeltaRecord:
        if (
            not isinstance(base.handle.backend_ref, DeviceKVRef)
            or base.handle.residency != ResidencyTier.DEVICE
            or not store.is_current(base.handle)
        ):
            raise KVCOMMInvariantError("anchor update requires a current device base")
        length = observed.key.token_count
        delta_indices = allocator.alloc(length)
        if delta_indices is None or len(delta_indices) != length:
            if delta_indices is not None:
                allocator.free(delta_indices)
            raise MemoryError("unable to allocate KVCOMM delta slots")

        kvcache = allocator.get_kvcache()
        base_indices = base.handle.backend_ref.indices.long()
        actual_indices = observed.indices.long()
        try:
            for layer_id in range(capabilities.layer_count):
                key_buffer = kvcache.get_key_buffer(layer_id)
                value_buffer = kvcache.get_value_buffer(layer_id)
                if layer_id == 0:
                    # Layer-0 KV is context-independent once Key positions are normalized.
                    key_buffer[delta_indices] = 0
                    value_buffer[delta_indices] = 0
                    continue
                base_key = key_buffer[base_indices].clone()
                actual_key = key_buffer[actual_indices].clone()
                base_normalized = normalize_key_positions(
                    base_key,
                    base.positions,
                    capabilities.rope,
                )
                actual_normalized = normalize_key_positions(
                    actual_key,
                    observed.positions,
                    capabilities.rope,
                )
                key_buffer[delta_indices] = (actual_normalized - base_normalized).to(
                    key_buffer.dtype
                )
                value_buffer[delta_indices] = (
                    value_buffer[actual_indices] - value_buffer[base_indices]
                ).to(value_buffer.dtype)
        except Exception:
            allocator.free(delta_indices)
            raise

        digest = hashlib.sha256(
            (
                f"{anchor_id}:{observed.descriptor.role.value}:" f"{update_serial}"
            ).encode("utf-8")
        ).hexdigest()
        kind = (
            SegmentKind.KVCOMM_PLACEHOLDER_DELTA
            if observed.descriptor.role == KVCOMMSegmentRole.PLACEHOLDER
            else SegmentKind.KVCOMM_NEIGHBOR_DELTA
        )
        delta_key = make_kvcomm_segment_key(
            tokens=observed.token_ids,
            content_hash=digest,
            model_fingerprint=observed.key.model_fingerprint,
            cache_dtype=observed.key.cache_dtype,
            kind=kind,
        )

        released = False

        def release(ref: object, residency: ResidencyTier) -> None:
            nonlocal released
            if residency != ResidencyTier.DEVICE or not isinstance(
                ref,
                DeviceKVRef,
            ):
                raise TypeError("invalid KVCOMM delta device reference")
            allocator.free(ref.indices)
            released = True

        try:
            handle = store.register(
                key=delta_key,
                token_ids=observed.token_ids,
                source_start=0,
                residency=ResidencyTier.DEVICE,
                backend_ref=DeviceKVRef(delta_indices),
                release_backend=release,
            )
        except Exception:
            if not released:
                allocator.free(delta_indices)
            raise
        return KVCOMMDeltaRecord(
            role=observed.descriptor.role,
            handle=handle,
            base_key=base.key,
            base_generation=base.handle.generation,
            provenance=provenance,
            layer_count=capabilities.layer_count,
        )

    @staticmethod
    def _release_anchor_handles(
        anchor: KVCOMMAnchor,
        store: ApproxKVSegmentStore,
    ) -> None:
        store.release(anchor.placeholder_delta.handle)
        if anchor.neighbor_delta is not None:
            store.release(anchor.neighbor_delta.handle)

    def update_from_dense(
        self,
        *,
        metadata: Any,
        spec: KVCOMMRequestSpec,
        observed: Sequence[KVCOMMObservedSegment],
        store: ApproxKVSegmentStore,
        allocator: Any,
    ) -> int:
        capabilities = self._require_capabilities()
        guard_reason = capabilities.guard_kvcache(allocator.get_kvcache())
        if guard_reason is not None:
            raise KVCOMMCapabilityError(guard_reason)
        dtype_reason = capabilities.guard_declared_dtype(
            allocator.get_kvcache(),
            metadata.cache_dtype,
        )
        if dtype_reason is not None:
            raise KVCOMMCapabilityError(dtype_reason)
        observed_by_index = {
            segment.descriptor.segment_index: segment for segment in observed
        }
        updated_tokens = 0
        for placeholder_descriptor, neighbor_descriptor in self._group_descriptors(
            spec
        ):
            placeholder = observed_by_index[placeholder_descriptor.segment_index]
            neighbor = (
                observed_by_index[neighbor_descriptor.segment_index]
                if neighbor_descriptor is not None
                else None
            )
            placeholder_provenance = self._provenance(
                metadata,
                spec,
                placeholder_descriptor,
                capabilities,
            )
            placeholder_base, reason = self._lookup_base(
                placeholder_descriptor,
                placeholder.key,
                placeholder_provenance,
                store,
            )
            if placeholder_base is None:
                raise KVCOMMInvariantError(reason or "placeholder base missing")

            neighbor_base = None
            neighbor_provenance = None
            if neighbor is not None and neighbor_descriptor is not None:
                neighbor_provenance = self._provenance(
                    metadata,
                    spec,
                    neighbor_descriptor,
                    capabilities,
                )
                neighbor_base, reason = self._lookup_base(
                    neighbor_descriptor,
                    neighbor.key,
                    neighbor_provenance,
                    store,
                )
                if neighbor_base is None:
                    raise KVCOMMInvariantError(reason or "neighbor base missing")

            anchor_id = self._anchor_id(spec, placeholder)
            with self._lock:
                self._sequence += 1
                update_serial = self._sequence

            created_deltas = []
            try:
                placeholder_delta = self._materialize_delta(
                    anchor_id=anchor_id,
                    update_serial=update_serial,
                    observed=placeholder,
                    base=placeholder_base,
                    provenance=placeholder_provenance,
                    store=store,
                    allocator=allocator,
                    capabilities=capabilities,
                )
                created_deltas.append(placeholder_delta.handle)
                neighbor_delta = None
                if (
                    neighbor is not None
                    and neighbor_base is not None
                    and neighbor_provenance is not None
                ):
                    neighbor_delta = self._materialize_delta(
                        anchor_id=anchor_id,
                        update_serial=update_serial,
                        observed=neighbor,
                        base=neighbor_base,
                        provenance=neighbor_provenance,
                        store=store,
                        allocator=allocator,
                        capabilities=capabilities,
                    )
                    created_deltas.append(neighbor_delta.handle)
            except Exception:
                for handle in created_deltas:
                    store.release(handle)
                raise

            pool_key = self._pool_key(
                spec,
                placeholder_descriptor.placeholder_id,
                placeholder_provenance,
            )
            release_after = []
            with self._lock:
                pool = self._pools.setdefault(pool_key, OrderedDict())
                existing = pool.pop(anchor_id, None)
                if existing is not None:
                    release_after.append(existing)
                elif len(pool) >= spec.max_anchors:
                    candidate_items = list(pool.items())[: spec.prune_window]
                    victim_id, victim = min(
                        candidate_items,
                        key=lambda item: (
                            item[1].access_count,
                            item[1].created_order,
                        ),
                    )
                    del pool[victim_id]
                    release_after.append(victim)

                generation = self._pool_generations.get(pool_key, 0) + 1
                self._pool_generations[pool_key] = generation
                anchor = KVCOMMAnchor(
                    anchor_id=anchor_id,
                    placeholder_id=placeholder_descriptor.placeholder_id,
                    agent_id=spec.agent_id,
                    template_fingerprint=spec.template_fingerprint,
                    context_fingerprint=spec.context_fingerprint,
                    provenance=placeholder_provenance,
                    placeholder_base_handle=placeholder_base.handle,
                    neighbor_base_handle=(
                        neighbor_base.handle if neighbor_base is not None else None
                    ),
                    placeholder_delta=placeholder_delta,
                    neighbor_delta=neighbor_delta,
                    embedding=placeholder_base.embedding.detach().clone(),
                    generation=generation,
                    created_order=update_serial,
                )
                pool[anchor_id] = anchor

            for old_anchor in release_after:
                self._release_anchor_handles(old_anchor, store)
            updated_tokens += placeholder.key.token_count
            if neighbor is not None:
                updated_tokens += neighbor.key.token_count
        return updated_tokens

    def pool_snapshot(
        self,
    ) -> dict[tuple[str, ...], tuple[KVCOMMAnchor, ...]]:
        with self._lock:
            return {key: tuple(pool.values()) for key, pool in self._pools.items()}

    def pool_generation(self, pool_key: tuple[str, ...]) -> int:
        with self._lock:
            return self._pool_generations.get(pool_key, 0)


def execute_kvcomm_reconstruction(
    *,
    plan: KVCOMMReconstructionPlan,
    store: ApproxKVSegmentStore,
    allocator: Any,
    target_indices: torch.Tensor,
    capabilities: KVCOMMRuntimeCapabilities,
) -> KVTransferStats:
    if (
        target_indices.ndim != 1
        or len(target_indices) != plan.restore_length
        or torch.unique(target_indices).numel() != plan.restore_length
    ):
        raise KVCOMMInvariantError("target allocation length mismatch")
    if plan.capability_fingerprint != capabilities.fingerprint:
        raise KVCOMMCapabilityError("capability_generation_mismatch")
    kvcache = allocator.get_kvcache()
    guard_reason = capabilities.guard_kvcache(kvcache)
    if guard_reason is not None:
        raise KVCOMMCapabilityError(guard_reason)
    for declared_dtype in {
        reconstruction_slice.base.key.cache_dtype
        for reconstruction_slice in plan.slices
    }:
        dtype_reason = capabilities.guard_declared_dtype(kvcache, declared_dtype)
        if dtype_reason is not None:
            raise KVCOMMCapabilityError(dtype_reason)

    occupied = set()
    for reconstruction_slice in plan.slices:
        if (
            reconstruction_slice.source_offset < 0
            or reconstruction_slice.target_start < 0
            or reconstruction_slice.length <= 0
            or reconstruction_slice.target_start + reconstruction_slice.length
            > plan.restore_length
            or reconstruction_slice.target_position
            != plan.exact_prefix_length + reconstruction_slice.target_start
            or reconstruction_slice.source_offset + reconstruction_slice.length
            > reconstruction_slice.base.key.token_count
            or len(reconstruction_slice.base.positions)
            != reconstruction_slice.base.key.token_count
        ):
            raise KVCOMMInvariantError("invalid reconstruction slice bounds")
        positions = set(
            range(
                reconstruction_slice.target_start,
                reconstruction_slice.target_start + reconstruction_slice.length,
            )
        )
        if occupied & positions:
            raise KVCOMMInvariantError("overlapping KVCOMM reconstruction slices")
        occupied |= positions
        if not store.is_current(reconstruction_slice.base.handle):
            raise KVCOMMInvariantError("stale base generation")
        if (
            not isinstance(
                reconstruction_slice.base.handle.backend_ref,
                DeviceKVRef,
            )
            or reconstruction_slice.base.handle.residency != ResidencyTier.DEVICE
        ):
            raise KVCOMMInvariantError("base is not device-resident")
        if reconstruction_slice.base.handle.key != reconstruction_slice.base.key:
            raise KVCOMMInvariantError("base handle identity mismatch")
        validate_interpolation_weights(
            reconstruction_slice.weights,
            len(reconstruction_slice.deltas),
        )
        for delta in reconstruction_slice.deltas:
            delta_base = store.lookup(delta.base_key)
            if (
                not store.is_current(delta.handle)
                or not isinstance(delta.handle.backend_ref, DeviceKVRef)
                or delta_base is None
                or delta_base.generation != delta.base_generation
                or delta_base.residency != ResidencyTier.DEVICE
                or not isinstance(delta_base.backend_ref, DeviceKVRef)
                or delta.role != reconstruction_slice.role
                or not delta.provenance.compatible_with(
                    reconstruction_slice.base.provenance
                )
                or (
                    reconstruction_slice.role == KVCOMMSegmentRole.NEIGHBOR
                    and delta.provenance.source_fingerprint
                    != reconstruction_slice.base.provenance.source_fingerprint
                )
                or delta.layer_count != capabilities.layer_count
                or delta.handle.key.token_count
                < reconstruction_slice.source_offset + reconstruction_slice.length
            ):
                raise KVCOMMInvariantError("stale or incompatible delta generation")
    if occupied != set(range(plan.restore_length)):
        raise KVCOMMInvariantError(
            "KVCOMM reconstruction does not cover the full prefix"
        )

    copy_start = time.perf_counter()
    rope_seconds = 0.0
    for layer_id in range(capabilities.layer_count):
        key_buffer = kvcache.get_key_buffer(layer_id)
        value_buffer = kvcache.get_value_buffer(layer_id)
        for reconstruction_slice in plan.slices:
            offset = reconstruction_slice.source_offset
            end = offset + reconstruction_slice.length
            base_indices = reconstruction_slice.base.handle.backend_ref.indices[
                offset:end
            ].long()
            destination = target_indices[
                reconstruction_slice.target_start : reconstruction_slice.target_start
                + reconstruction_slice.length
            ].long()
            base_positions = reconstruction_slice.base.positions[offset:end]
            target_positions = range(
                reconstruction_slice.target_position,
                reconstruction_slice.target_position + reconstruction_slice.length,
            )

            rope_start = time.perf_counter()
            normalized_key = normalize_key_positions(
                key_buffer[base_indices].clone(),
                base_positions,
                capabilities.rope,
            )
            rope_seconds += time.perf_counter() - rope_start
            reconstructed_value = value_buffer[base_indices].clone()
            weights = validate_interpolation_weights(
                reconstruction_slice.weights,
                len(reconstruction_slice.deltas),
            ).to(device=key_buffer.device)
            for weight, delta in zip(
                weights,
                reconstruction_slice.deltas,
            ):
                delta_indices = delta.handle.backend_ref.indices[offset:end].long()
                normalized_key.add_(
                    key_buffer[delta_indices].to(normalized_key.dtype),
                    alpha=float(weight),
                )
                reconstructed_value.add_(
                    value_buffer[delta_indices].to(reconstructed_value.dtype),
                    alpha=float(weight),
                )

            rope_start = time.perf_counter()
            reconstructed_key = rotate_key_positions(
                normalized_key,
                target_positions,
                capabilities.rope,
            )
            rope_seconds += time.perf_counter() - rope_start
            key_buffer[destination] = reconstructed_key.to(key_buffer.dtype)
            value_buffer[destination] = reconstructed_value.to(value_buffer.dtype)

    elapsed_ms = (time.perf_counter() - copy_start) * 1000
    rope_ms = rope_seconds * 1000
    layer_tokens = plan.restore_length * capabilities.layer_count
    stats = KVTransferStats(
        recovery_mode=RecoveryMode.KVCOMM,
        target_tokens=plan.restore_length,
        copied_k_tokens=plan.restore_length,
        rotated_k_tokens=plan.restore_length,
        copied_v_tokens=plan.restore_length,
        layer_count=capabilities.layer_count,
        copied_k_layer_tokens=layer_tokens,
        rotated_k_layer_tokens=layer_tokens,
        copied_v_layer_tokens=layer_tokens,
        copy_ms=max(0.0, elapsed_ms - rope_ms),
        rope_ms=rope_ms,
    )
    if not stats.mechanically_valid:
        raise KVCOMMInvariantError("KVCOMM full-layer K/V accounting failed")
    return stats
