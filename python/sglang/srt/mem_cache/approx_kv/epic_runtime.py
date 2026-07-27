"""EPIC fixed leading-k attention-sink repair -- real server request hook.

This is the request-hook analogue of ``runtime.restore_request_prefix``
(the R0 raw-copy path), specialised for the EPIC recovery plugin
(``epic_plugin.EPICLeadingKPlugin``). It reuses the exact same
segment-resolution logic as R0 (``runtime.resolve_reuse_spans``) -- no
duplicated data plane -- and, for the degenerate ``k == 0`` case, delegates
to the exact same physical copy execution as R0
(``runtime.finalize_copy_reuse``).

For ``k > 0`` this module drives ``epic_recompute.LayerwiseEpicExecutor``
to genuinely recompute the leading-k target-context tokens layer-by-layer,
interleaved with a per-layer body copy (``radix_backend.
RadixKVTransferBackend.copy_and_rotate_layer``) -- so for every transformer
layer, the leading-k tokens are recomputed for that layer *before* that
layer's remaining body KV is reused, exactly matching the requirement that
"merely copying a body or planning k is not completion".

Capability guards (``epic_capability.inspect_layerwise_recompute_capability``)
and an explicit ``epic_forward_batch_factory`` seam gate the k>0 path.
``TorchNativeEpicForwardBatchFactory`` implements that seam for the
single-rank torch-native Qwen-style path used by the SM75 experiments. It
allocates a temporary request-table row, maps the exact prefix plus the
leading-k output slots, builds a one-request extend ``ForwardBatch``, and
uses the live model's input embedding and attention backend metadata.

Other attention backends, TP/PP/DP layouts, LoRA, multimodal requests, and
embedding overrides remain capability-gated to dense fallback. This narrow
production subset is intentional: it replaces the previous unbound seam
without claiming a backend-generic ForwardBatch constructor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Protocol

import torch

from .epic_capability import inspect_layerwise_recompute_capability
from .epic_plugin import EPICLeadingKPlugin, carve_leading_k
from .epic_recompute import (
    LayerwiseEpicExecutor,
    LayerwiseLeadingKRepairError,
    ModelRunnerLeadingKRecomputeBackend,
)
from .plugins import RecoveryRequestContext
from .radix_backend import RadixKVTransferBackend, RoPEConfig
from .request import ApproxKVRequestOperation
from .runtime import (
    ResolvedReuseSpans,
    _allocator,
    allocate_recovery_slots,
    finalize_copy_reuse,
    pin_reuse_sources,
    resolve_reuse_spans,
)
from .transfer import _validate_bounds
from .types import TransferSpan

logger = logging.getLogger(__name__)


def resolve_model_rope_config(model_config: Any) -> RoPEConfig | None:
    """Resolve the supported Qwen RoPE layout from the live model config."""
    hf_config = model_config.hf_config
    model_type = str(getattr(hf_config, "model_type", "")).lower()
    if model_type not in {"qwen2", "qwen3"}:
        return None
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling:
        rope_type = str(
            rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        ).lower()
        if rope_type not in {"", "default"}:
            return None
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        if hidden_size % num_heads:
            return None
        head_dim = hidden_size // num_heads
    rotary_dim = int(
        int(head_dim) * float(getattr(hf_config, "partial_rotary_factor", 1.0))
    )
    if rotary_dim <= 0 or rotary_dim % 2:
        return None
    return RoPEConfig(
        rotary_dim=rotary_dim,
        base=float(
            getattr(hf_config, "rope_theta", None)
            or (rope_scaling or {}).get("rope_theta", 10000.0)
        ),
        is_neox_style=True,
    )


@dataclass(frozen=True)
class EpicForwardBatchBundle:
    """Real forward-pass inputs for the leading-k recompute window.

    ``positions``/``hidden_states``/``residual`` are the genuine per-token
    model inputs for the leading-k tokens (e.g. ``hidden_states`` is the
    real output of the embedding lookup, not a placeholder);
    ``forward_batch`` is whatever the model's decoder layers expect as
    their ``forward_batch`` argument, already wired so that a real
    ``self.attn(..., save_kv_cache=True)`` call inside each layer's
    forward writes K/V for these tokens into ``leading_k_target_indices``.
    """

    positions: Any
    hidden_states: Any
    residual: Any
    forward_batch: Any
    release: Callable[[], None] | None = None


class EpicForwardBatchFactory(Protocol):
    """Builds real forward-pass inputs for a request's leading-k window.

    Production code binds an implementation onto
    ``ApproxKVManager.epic_forward_batch_factory`` that constructs a real,
    correctly-metadata'd ``ForwardBatch`` for exactly ``k`` tokens whose
    physical destination slots are ``leading_k_target_indices``.
    """

    def __call__(
        self,
        tree_cache: Any,
        req: Any,
        resolved: ResolvedReuseSpans,
        k: int,
        leading_k_target_indices: torch.Tensor,
    ) -> EpicForwardBatchBundle: ...


class TorchNativeEpicForwardBatchFactory:
    """Build a temporary single-request extend batch for torch-native EPIC."""

    def __init__(self, model_runner: Any) -> None:
        self._model_runner = model_runner

    def _validate_runtime(self, req: Any) -> None:
        model_runner = self._model_runner
        server_args = model_runner.server_args
        if str(server_args.attention_backend) != "torch_native":
            raise LayerwiseLeadingKRepairError(
                "epic_forward_batch_requires_torch_native"
            )
        from sglang.srt.layers.attention.torch_native_backend import (
            TorchNativeAttnBackend,
        )

        if not isinstance(model_runner.attn_backend, TorchNativeAttnBackend):
            raise LayerwiseLeadingKRepairError("epic_forward_batch_backend_mismatch")
        if bool(
            getattr(
                model_runner.attn_backend,
                "use_sliding_window_kv_pool",
                False,
            )
        ):
            raise LayerwiseLeadingKRepairError(
                "epic_forward_batch_does_not_support_sliding_window"
            )
        for field in ("tp_size", "pp_size", "dp_size"):
            if int(getattr(server_args, field, 1)) != 1:
                raise LayerwiseLeadingKRepairError(
                    f"epic_forward_batch_requires_{field}_1"
                )
        if bool(getattr(server_args, "enable_lora", False)):
            raise LayerwiseLeadingKRepairError(
                "epic_forward_batch_does_not_support_lora"
            )
        if bool(getattr(model_runner.model_config, "model_is_mrope", False)):
            raise LayerwiseLeadingKRepairError(
                "epic_forward_batch_does_not_support_mrope"
            )
        if any(
            getattr(req, field, None) is not None
            for field in (
                "input_embeds",
                "replace_embeds",
                "positional_embed_overrides",
                "multimodal_inputs",
            )
        ):
            raise LayerwiseLeadingKRepairError(
                "epic_forward_batch_does_not_support_embedding_overrides"
            )

    def __call__(
        self,
        tree_cache: Any,
        req: Any,
        resolved: ResolvedReuseSpans,
        k: int,
        leading_k_target_indices: torch.Tensor,
    ) -> EpicForwardBatchBundle:
        if k <= 0:
            raise ValueError("k must be positive")
        self._validate_runtime(req)
        if len(req.prefix_indices) != resolved.exact_length:
            raise LayerwiseLeadingKRepairError("epic_exact_prefix_length_mismatch")
        if resolved.exact_length + k > resolved.restore_end:
            raise LayerwiseLeadingKRepairError("epic_leading_k_exceeds_restore_window")

        model_runner = self._model_runner
        req_to_token_pool = tree_cache.req_to_token_pool
        temporary_req = SimpleNamespace(req_pool_idx=None)
        allocated = req_to_token_pool.alloc([temporary_req])
        if not allocated or temporary_req.req_pool_idx is None:
            raise LayerwiseLeadingKRepairError(
                "epic_temporary_request_slot_unavailable"
            )

        sequence_length = resolved.exact_length + k
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            row = temporary_req.req_pool_idx
            try:
                if row is not None:
                    zeros = torch.zeros(
                        sequence_length,
                        dtype=req_to_token_pool.req_to_token.dtype,
                        device=req_to_token_pool.req_to_token.device,
                    )
                    req_to_token_pool.write(
                        (row, slice(0, sequence_length)),
                        zeros,
                    )
            finally:
                try:
                    if temporary_req.req_pool_idx is not None:
                        req_to_token_pool.free(temporary_req)
                finally:
                    released = temporary_req.req_pool_idx is None

        try:
            device = torch.device(model_runner.device)
            input_ids = torch.tensor(
                req.full_untruncated_fill_ids[
                    resolved.exact_length : resolved.exact_length + k
                ],
                dtype=torch.int64,
                device=device,
            )
            positions = torch.arange(
                resolved.exact_length,
                resolved.exact_length + k,
                dtype=torch.int64,
                device=device,
            )
            out_cache_loc = leading_k_target_indices.to(
                device=device,
                dtype=torch.int64,
            )
            prefix_indices = req.prefix_indices.to(
                device=req_to_token_pool.req_to_token.device,
                dtype=req_to_token_pool.req_to_token.dtype,
            )
            mapped_leading = out_cache_loc.to(
                device=req_to_token_pool.req_to_token.device,
                dtype=req_to_token_pool.req_to_token.dtype,
            )
            req_to_token_pool.write(
                (temporary_req.req_pool_idx, slice(0, sequence_length)),
                torch.cat((prefix_indices, mapped_leading)),
            )

            from sglang.srt.model_executor.forward_batch_info import (
                ForwardBatch,
                ForwardMode,
            )

            req_pool_indices = torch.tensor(
                [temporary_req.req_pool_idx],
                dtype=torch.int64,
                device=device,
            )
            seq_lens = torch.tensor(
                [sequence_length],
                dtype=torch.int32,
                device=device,
            )
            extend_prefix_lens = torch.tensor(
                [resolved.exact_length],
                dtype=torch.int32,
                device=device,
            )
            extend_seq_lens = torch.tensor(
                [k],
                dtype=torch.int32,
                device=device,
            )
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                seq_lens_sum=sequence_length,
                seq_lens_cpu=torch.tensor(
                    [sequence_length],
                    dtype=torch.int32,
                ),
                positions=positions,
                extend_num_tokens=k,
                extend_seq_lens=extend_seq_lens,
                extend_prefix_lens=extend_prefix_lens,
                extend_start_loc=torch.zeros(
                    1,
                    dtype=torch.int32,
                    device=device,
                ),
                extend_prefix_lens_cpu=[resolved.exact_length],
                extend_seq_lens_cpu=[k],
                rids=[str(req.rid)],
            )
            model_runner.attn_backend.init_forward_metadata(forward_batch)
            forward_batch.mark_forward_metadata_ready()

            model = model_runner.model
            inner_model = getattr(model, "model", model)
            get_input_embedding = getattr(inner_model, "get_input_embedding", None)
            if not callable(get_input_embedding):
                raise LayerwiseLeadingKRepairError(
                    "epic_model_exposes_no_input_embedding"
                )
            with torch.inference_mode():
                hidden_states = get_input_embedding(input_ids)
            return EpicForwardBatchBundle(
                positions=positions,
                hidden_states=hidden_states,
                residual=None,
                forward_batch=forward_batch,
                release=release,
            )
        except Exception:
            release()
            raise


class _PerLayerBodyCopyBackend:
    """Adapts ``RadixKVTransferBackend.copy_and_rotate_layer`` into the
    ``epic_recompute.BodyLayerCopyBackend`` protocol, copying only the
    body spans (the resolved region strictly after the leading-k window)
    for exactly one layer at a time.
    """

    def __init__(
        self,
        *,
        backend: RadixKVTransferBackend,
        body_spans: tuple[TransferSpan, ...],
    ) -> None:
        self._backend = backend
        self._body_spans = body_spans

    def copy_layer(self, *, layer_id: int) -> None:
        for span in self._body_spans:
            result = self._backend.copy_and_rotate_layer(
                layer_id=layer_id,
                source_ref=span.source.backend_ref,
                source_offset=span.source_offset,
                target_start=span.target_start,
                length=span.length,
                rope_delta=span.rope_delta,
            )
            if result.copied_k_tokens != span.length:
                raise LayerwiseLeadingKRepairError(
                    "per-layer body copy did not cover the requested span"
                )


def restore_request_prefix_epic(tree_cache: Any, req: Any) -> bool:
    """EPIC's request hook: exact-cache-first, dense fallback on any gap.

    Mirrors ``runtime.restore_request_prefix``'s guard structure exactly,
    dispatching to the EPIC recovery plugin only once the request is
    known to need approximate reuse of a genuinely resolved, contiguous,
    device-resident body.
    """
    metadata = getattr(req, "approx_kv_metadata", None)
    manager = getattr(tree_cache, "approx_kv", None)
    if (
        metadata is None
        or metadata.operation != ApproxKVRequestOperation.REUSE
        or manager is None
        or not manager.config.core_enabled
        or not manager.config.epic_enabled
    ):
        return False
    if req.needs_host_load_back():
        manager.record_request("reuse", "exact_host_preferred")
        return False

    resolved = resolve_reuse_spans(tree_cache, req, metadata, manager)
    if resolved is None:
        return False

    with pin_reuse_sources(manager, resolved) as pinned:
        if not pinned:
            return False
        try:
            plugin = manager.plugins.get("epic")
        except KeyError:
            manager.record_fallback("epic_plugin_missing", resolved.restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False
        if not isinstance(plugin, EPICLeadingKPlugin):
            manager.record_fallback("epic_plugin_wrong_type", resolved.restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False

        k = plugin.leading_k_window(resolved.restore_length)
        if k <= 0:
            # No leading-k repair requested (or the region is too small to
            # carve one out): this degenerates to exactly the R0 raw-copy
            # mechanism, reusing it directly rather than re-implementing it.
            return finalize_copy_reuse(tree_cache, req, manager, resolved)
        if not plugin.attention_sink:
            manager.record_fallback(
                "epic_attention_sink_disabled",
                resolved.restore_length,
            )
            manager.record_request("reuse", "dense_fallback")
            return False

        if metadata.plugin == "epic_precomputed":
            return _restore_with_precomputed_leading_k(
                tree_cache,
                req,
                manager,
                resolved,
                plugin,
                k,
            )

        return _restore_with_leading_k_repair(
            tree_cache,
            req,
            manager,
            resolved,
            plugin,
            k,
        )


def _restore_with_precomputed_leading_k(
    tree_cache: Any,
    req: Any,
    manager: Any,
    resolved: ResolvedReuseSpans,
    plugin: EPICLeadingKPlugin,
    k: int,
) -> bool:
    cursor = 0
    for span in resolved.spans:
        if cursor >= k:
            break
        if span.target_start != cursor:
            break
        covered = min(span.length, k - cursor)
        source_position = span.source.source_start + span.source_offset
        target_position = resolved.exact_length + span.target_start
        if (
            not span.source.key.content_hash.startswith("epic-repair:")
            or span.rope_delta != 0
            or source_position != target_position
        ):
            break
        cursor += covered
    if cursor != k:
        manager.record_fallback(
            "epic_precomputed_repair_invalid",
            resolved.restore_length,
        )
        manager.record_request("reuse", "dense_fallback")
        return False

    restored = finalize_copy_reuse(tree_cache, req, manager, resolved)
    if restored:
        num_layers = 0
        if manager.model_runner is not None:
            capability = inspect_layerwise_recompute_capability(manager.model_runner)
            if capability.supported:
                num_layers = capability.num_layers
        manager.record_epic_layer_recompute(
            layers_recomputed=num_layers,
            leading_k_tokens=k,
            genuinely_layerwise=True,
        )
    return restored


def _restore_with_leading_k_repair(
    tree_cache: Any,
    req: Any,
    manager: Any,
    resolved: ResolvedReuseSpans,
    plugin: EPICLeadingKPlugin,
    k: int,
) -> bool:
    model_runner = manager.model_runner
    if model_runner is None:
        manager.record_fallback("epic_model_runner_unbound", resolved.restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    capability = inspect_layerwise_recompute_capability(model_runner)
    if not capability.supported:
        manager.record_fallback(
            f"epic_capability_unsupported:{capability.reason}",
            resolved.restore_length,
        )
        manager.record_request("reuse", "dense_fallback")
        return False

    factory: EpicForwardBatchFactory | None = getattr(
        manager, "epic_forward_batch_factory", None
    )
    if factory is None:
        manager.record_fallback(
            "epic_forward_batch_unavailable", resolved.restore_length
        )
        manager.record_request("reuse", "dense_fallback")
        return False

    target_tokens = tuple(
        int(token)
        for token in req.full_untruncated_fill_ids[
            resolved.exact_length : resolved.restore_end
        ]
    )
    context = RecoveryRequestContext(
        request_id=str(req.rid),
        target_token_ids=target_tokens,
        exact_prefix_length=0,
        custom_metadata={"resolved_spans": resolved.spans},
    )
    plan = plugin.build_plan(context, manager.store)
    try:
        _validate_bounds(plan)
    except ValueError as exc:
        manager.record_fallback(f"epic_plan_invalid:{exc}", resolved.restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    allocator = _allocator(tree_cache)
    restored_indices = allocate_recovery_slots(
        tree_cache,
        resolved.restore_length,
    )
    if restored_indices is None or len(restored_indices) != resolved.restore_length:
        if restored_indices is not None:
            allocator.free(restored_indices)
        manager.record_fallback("device_allocation_failed", resolved.restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    try:
        recompute_backend = ModelRunnerLeadingKRecomputeBackend(model_runner)
    except LayerwiseLeadingKRepairError as exc:
        allocator.free(restored_indices)
        manager.record_fallback(
            f"epic_capability_unsupported:{exc}", resolved.restore_length
        )
        manager.record_request("reuse", "dense_fallback")
        return False

    try:
        bundle = factory(tree_cache, req, resolved, k, restored_indices[:k])
    except (
        AssertionError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        LayerwiseLeadingKRepairError,
        MemoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        allocator.free(restored_indices)
        logger.exception(
            "EPIC forward-batch construction failed for request %s",
            getattr(req, "rid", "<unknown>"),
        )
        manager.record_fallback(
            "epic_forward_batch_construction_failed", resolved.restore_length
        )
        manager.record_request("reuse", "dense_fallback")
        return False

    body_spans = carve_leading_k(resolved.spans, k)
    fallback_reasons: list[str] = []
    backend = RadixKVTransferBackend(
        allocator=allocator,
        target_indices=lambda start, length: restored_indices[start : start + length],
        dense_prefill=lambda start, length, reason: fallback_reasons.append(reason),
        rope=resolved.rope_config,
    )
    body_copy_backend = (
        _PerLayerBodyCopyBackend(backend=backend, body_spans=body_spans)
        if body_spans
        else None
    )
    executor = LayerwiseEpicExecutor(
        recompute_backend=recompute_backend,
        body_copy_backend=body_copy_backend,
    )
    body_tokens = resolved.restore_length - k
    try:
        with torch.inference_mode():
            exec_stats, _, _ = executor.run(
                positions=bundle.positions,
                hidden_states=bundle.hidden_states,
                residual=bundle.residual,
                forward_batch=bundle.forward_batch,
                leading_k_tokens=k,
                body_tokens=body_tokens,
            )
    except Exception:
        _synchronize_failed_epic_work(model_runner)
        _release_forward_bundle(bundle, req)
        allocator.free(restored_indices)
        logger.exception(
            "EPIC layerwise recompute failed for request %s; "
            "falling back to dense prefill",
            getattr(req, "rid", "<unknown>"),
        )
        manager.record_fallback("epic_recompute_failed", resolved.restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    if not _release_forward_bundle(bundle, req):
        _synchronize_failed_epic_work(model_runner)
        allocator.free(restored_indices)
        manager.record_fallback(
            "epic_forward_batch_release_failed",
            resolved.restore_length,
        )
        manager.record_request("reuse", "dense_fallback")
        return False
    if fallback_reasons:
        _synchronize_failed_epic_work(model_runner)
        allocator.free(restored_indices)
        manager.record_request("reuse", "dense_fallback")
        return False
    if not exec_stats.genuinely_layerwise:
        # Mechanical proof failed: never silently commit a repair that
        # was not actually interleaved layer-by-layer.
        _synchronize_failed_epic_work(model_runner)
        allocator.free(restored_indices)
        manager.record_fallback("epic_not_genuinely_layerwise", resolved.restore_length)
        manager.record_request("reuse", "dense_fallback")
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
    req.approx_kv_restored_len = resolved.restore_length
    # Provisional until prepare_for_extend copies them into req_to_token.
    req.approx_kv_provisional_indices = restored_indices
    req.approx_kv_epic_stats = exec_stats
    manager.record_epic_layer_recompute(
        layers_recomputed=exec_stats.layers_invoked,
        leading_k_tokens=exec_stats.leading_k_tokens,
        genuinely_layerwise=exec_stats.genuinely_layerwise,
    )
    manager.record_request("reuse", "success")
    return True


def _synchronize_failed_epic_work(model_runner: Any) -> None:
    device = torch.device(getattr(model_runner, "device", "cpu"))
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.current_stream(device).synchronize()
        except Exception:
            logger.exception(
                "EPIC failed-work CUDA synchronization raised; "
                "continuing resource cleanup"
            )


def _release_forward_bundle(bundle: EpicForwardBatchBundle, req: Any) -> bool:
    if bundle.release is None:
        return True
    try:
        bundle.release()
    except Exception:
        logger.exception(
            "EPIC temporary forward-batch resources failed to release for %s",
            getattr(req, "rid", "<unknown>"),
        )
        return False
    return True
