from __future__ import annotations

"""CacheBlend recovery plugin identity, config and capability surface.

Registered into the common-core ``RecoveryPluginRegistry``
(``ApproxKVManager.register_plugin``) under the name ``"cacheblend"``.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

from sglang.srt.mem_cache.approx_kv.types import (
    DenseRange,
    KVReusePlan,
    RecoveryMode,
    SchedulerMetadata,
)

from .hkvd import GradualFilterStage
from .recompute import CacheBlendLayerRecomputeBackend, CacheBlendProbeBackend

CACHEBLEND_PLUGIN_NAME = "cacheblend"

# The four experiment ratios this Phase 4 R2 sweep must cover.
CACHEBLEND_RATIOS: tuple[float, ...] = (0.01, 0.05, 0.15, 0.30)


def _is_valid_ratio(ratio: float) -> bool:
    return any(math.isclose(ratio, valid, rel_tol=1e-9) for valid in CACHEBLEND_RATIOS)


@dataclass(frozen=True)
class CacheBlendConfig:
    """Sweep-configurable CacheBlend parameters.

    ``ratio`` is the final HKVD selective-recompute ratio (one of
    ``CACHEBLEND_RATIOS``: 1/5/15/30%). ``probe_stages`` defines the
    gradual (coarse-to-fine) filtering funnel; each stage names a real
    probe layer and the fraction of the *current* candidate pool it keeps.
    ``first_recompute_layer`` is the first layer that gets a real,
    per-selected-token forward (layers before it are only ever probed for
    HKVD scoring, never mutated).
    """

    ratio: float
    probe_stages: tuple[GradualFilterStage, ...]
    first_recompute_layer: int

    def __post_init__(self) -> None:
        if not _is_valid_ratio(self.ratio):
            raise ValueError(
                f"CacheBlend ratio must be one of {CACHEBLEND_RATIOS}, "
                f"got {self.ratio}"
            )
        if not self.probe_stages:
            raise ValueError("CacheBlend requires at least one probe stage")
        if self.first_recompute_layer < 0:
            raise ValueError("first_recompute_layer must be non-negative")
        max_probe_layer = max(stage.probe_layer_id for stage in self.probe_stages)
        if max_probe_layer >= self.first_recompute_layer:
            raise ValueError(
                "first_recompute_layer must be strictly deeper than every "
                "probe stage's layer"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "CacheBlendConfig":
        """Build a sweep configuration from environment variables.

        ``SGLANG_CACHEBLEND_RATIO`` (default ``0.05``): one of
        ``CACHEBLEND_RATIOS``.
        ``SGLANG_CACHEBLEND_PROBE_LAYERS`` (default ``"0"``): comma
        separated probe layer ids for the gradual filtering funnel, in
        shallow-to-deep order.
        ``SGLANG_CACHEBLEND_FIRST_RECOMPUTE_LAYER`` (default ``1``): first
        layer that gets a real per-selected-token recompute.
        """
        env = os.environ if env is None else env
        ratio = float(env.get("SGLANG_CACHEBLEND_RATIO", "0.05"))
        probe_layers = [
            int(layer.strip())
            for layer in env.get("SGLANG_CACHEBLEND_PROBE_LAYERS", "0").split(",")
            if layer.strip()
        ]
        first_recompute_layer = int(
            env.get("SGLANG_CACHEBLEND_FIRST_RECOMPUTE_LAYER", "1")
        )
        num_stages = len(probe_layers)
        # A single global keep_ratio per stage that funnels the candidate
        # pool down toward `ratio` by the time the final stage runs; the
        # true final selection is always re-scored (see hkvd.select_hkvd_tokens).
        stage_keep_ratio = max(ratio, ratio ** (1.0 / max(num_stages, 1)))
        stages = tuple(
            GradualFilterStage(probe_layer_id=layer_id, keep_ratio=stage_keep_ratio)
            for layer_id in probe_layers
        )
        return cls(
            ratio=ratio,
            probe_stages=stages,
            first_recompute_layer=first_recompute_layer,
        )


class CacheBlendRecoveryPlugin:
    """Registered under ``RecoveryPluginRegistry`` as ``"cacheblend"``.

    Implements the common-core ``RecoveryPlugin`` protocol (``name``,
    ``build_plan``, ``scheduler_metadata``) for identity/registration and
    Phase 5 scheduler-metadata bookkeeping. The *actual* selective
    per-layer execution is driven by
    ``cacheblend.runtime.restore_request_prefix_cacheblend`` rather than by
    ``build_plan`` + ``execute_reuse_plan``, because CacheBlend's mixed
    reuse+selective-recompute semantics do not fit the generic contiguous
    copy-or-dense ``KVReusePlan`` model: that model can only mark one
    *contiguous token range* as fully reused or fully dense, not "these
    scattered HKVD-selected positions get a real per-layer forward while
    interleaved, position-for-position, with reused positions in the same
    span". ``build_plan`` therefore returns a conservative, honest
    "requires online HKVD execution" dense plan for any generic caller
    that only uses the offline planning contract, and never fabricates a
    selection it has not actually run.
    """

    def __init__(
        self,
        *,
        config: CacheBlendConfig,
        probe_backend: CacheBlendProbeBackend | None = None,
        recompute_backend: CacheBlendLayerRecomputeBackend | None = None,
    ) -> None:
        self.config = config
        self.probe_backend = probe_backend
        self.recompute_backend = recompute_backend

    @property
    def name(self) -> str:
        return CACHEBLEND_PLUGIN_NAME

    @property
    def capable(self) -> bool:
        """True only if both real per-layer hooks are bound.

        This is the capability guard: without a real probe backend HKVD
        scores cannot be measured (only fabricated), and without a real
        recompute backend selected tokens cannot actually be repaired.
        Either gap means the request path must dense-fallback rather than
        claim a CacheBlend restore happened.
        """
        return self.probe_backend is not None and self.recompute_backend is not None

    def build_plan(self, context, store) -> KVReusePlan:
        del store
        target_len = len(context.target_token_ids) - context.exact_prefix_length
        if target_len <= 0:
            raise ValueError("no reusable tokens beyond the exact prefix")
        return KVReusePlan(
            target_token_ids=tuple(
                context.target_token_ids[context.exact_prefix_length :]
            ),
            recovery_mode=RecoveryMode.DENSE,
            dense_ranges=(
                DenseRange(
                    target_start=0,
                    length=target_len,
                    reason="cacheblend_requires_online_hkvd_execution",
                ),
            ),
        )

    def scheduler_metadata(self, context) -> tuple[SchedulerMetadata, ...]:
        del context
        return ()


def _read_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def maybe_register_cacheblend_plugin(
    manager: Any,
    *,
    env: Mapping[str, str] | None = None,
    probe_backend: CacheBlendProbeBackend | None = None,
    recompute_backend: CacheBlendLayerRecomputeBackend | None = None,
) -> CacheBlendRecoveryPlugin | None:
    """Register the CacheBlend plugin into ``manager`` when explicitly
    enabled via ``SGLANG_APPROX_KV_CACHEBLEND=1``.

    ``probe_backend``/``recompute_backend`` default to ``None``: this
    fork does not yet have a ModelRunner-level hook that can run a real,
    single-batched per-layer forward for an arbitrary token subset inside
    an otherwise-cached prefix (see ``recompute.py`` and
    ``cacheblend/runtime.py`` module docstrings). Registering the plugin
    with no bound backends is still meaningful and honest: it wires the
    real request-path dispatch (``schedule_batch.py`` routes
    ``plugin="cacheblend"`` requests here) and exercises the capability
    guard, which correctly dense-falls-back every such request until a
    real backend is bound -- it must never silently claim a HKVD-driven
    restore that did not happen.
    """
    env = os.environ if env is None else env
    if not _read_bool(env, "SGLANG_APPROX_KV_CACHEBLEND", False):
        return None
    config = CacheBlendConfig.from_env(env)
    plugin = CacheBlendRecoveryPlugin(
        config=config,
        probe_backend=probe_backend,
        recompute_backend=recompute_backend,
    )
    manager.register_plugin(plugin)
    return plugin
