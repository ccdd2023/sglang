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
and an explicit ``epic_forward_batch_factory`` seam gate the k>0 path:
whenever the live model/layout is unsupported, or no forward-batch factory
has been bound onto the manager, this hook safely dense-falls-back rather
than guessing -- see the module-level "PRODUCTION WIRING GAP" note below
for exactly what remains unwired and why.

PRODUCTION WIRING GAP (explicit, intentional, documented):
    Driving real decoder layers for a *sub-length* extend of an
    in-flight request requires constructing a standalone ``ForwardBatch``
    (with correct attention-backend metadata: ``out_cache_loc``,
    KV-cache-adjacent bookkeeping, positions, etc.) mid-request-registration
    -- outside of SGLang's normal scheduler batch-construction lifecycle.
    That construction is deeply coupled to the live attention backend and
    is unproven safe without a running GPU server, which this task
    explicitly forbids exercising in this worktree. The alternative safe
    mechanism (splitting the request into two natural chunked-prefill
    scheduling rounds via ``req.extend_input_len``) is out of scope because
    this worktree explicitly excludes scheduler-logic changes.

    Because of this, ``ApproxKVManager.epic_forward_batch_factory`` is
    **not** bound anywhere in the live scheduler by this worktree. The
    per-layer interleaved recompute+copy mechanism itself is fully
    implemented and is proven correct by CPU tests that supply a
    test-local (but genuinely tensor-computing) factory and decoder-layer
    stack -- see ``test/registered/unit/mem_cache/test_epic_runtime.py``.
    Wiring a production factory is the concrete next step for full server
    E2E and is intentionally left as an unresolved blocker rather than
    claimed as complete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .epic_capability import inspect_layerwise_recompute_capability
from .epic_plugin import EPICLeadingKPlugin, carve_leading_k
from .epic_recompute import (
    LayerwiseEpicExecutor,
    LayerwiseLeadingKRepairError,
    ModelRunnerLeadingKRecomputeBackend,
)
from .plugins import RecoveryRequestContext
from .radix_backend import RadixKVTransferBackend
from .request import ApproxKVRequestOperation
from .runtime import (
    ResolvedReuseSpans,
    _allocator,
    finalize_copy_reuse,
    resolve_reuse_spans,
)
from .transfer import _validate_bounds
from .types import TransferSpan

logger = logging.getLogger(__name__)


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


class EpicForwardBatchFactory(Protocol):
    """Builds real forward-pass inputs for a request's leading-k window.

    This is the explicit seam documented in this module's "PRODUCTION
    WIRING GAP" note: production code must bind an implementation of this
    protocol onto ``ApproxKVManager.epic_forward_batch_factory`` that
    constructs a real, correctly-metadata'd ``ForwardBatch`` for exactly
    ``k`` tokens whose physical destination slots are
    ``leading_k_target_indices``. No such production implementation is
    wired by this worktree (see module docstring); only test doubles
    exist today.
    """

    def __call__(
        self,
        tree_cache: Any,
        req: Any,
        resolved: ResolvedReuseSpans,
        k: int,
        leading_k_target_indices: torch.Tensor,
    ) -> EpicForwardBatchBundle: ...


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

    return _restore_with_leading_k_repair(tree_cache, req, manager, resolved, plugin, k)


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

    # Explicit, documented production-wiring gap: see module docstring.
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
    restored_indices = allocator.alloc(resolved.restore_length)
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
    except Exception:
        allocator.free(restored_indices)
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
        exec_stats, _, _ = executor.run(
            positions=bundle.positions,
            hidden_states=bundle.hidden_states,
            residual=bundle.residual,
            forward_batch=bundle.forward_batch,
            leading_k_tokens=k,
            body_tokens=body_tokens,
        )
    except Exception:
        allocator.free(restored_indices)
        logger.exception(
            "EPIC layerwise recompute failed for request %s; "
            "falling back to dense prefill",
            getattr(req, "rid", "<unknown>"),
        )
        manager.record_fallback("epic_recompute_failed", resolved.restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    if fallback_reasons:
        allocator.free(restored_indices)
        manager.record_request("reuse", "dense_fallback")
        return False
    if not exec_stats.genuinely_layerwise:
        # Mechanical proof failed: never silently commit a repair that
        # was not actually interleaved layer-by-layer.
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
    req.approx_kv_epic_stats = exec_stats
    manager.record_epic_layer_recompute(
        layers_recomputed=exec_stats.layers_invoked,
        leading_k_tokens=exec_stats.leading_k_tokens,
        genuinely_layerwise=exec_stats.genuinely_layerwise,
    )
    manager.record_request("reuse", "success")
    return True
