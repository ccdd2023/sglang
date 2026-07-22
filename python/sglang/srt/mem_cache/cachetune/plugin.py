from __future__ import annotations

"""CacheTune recovery plugin identity, config and capability surface.

Registered into the common-core ``RecoveryPluginRegistry``
(``ApproxKVManager.register_plugin``) under the name ``"cachetune"``.

Unlike `research/cacheblend`'s plugin (a fixed sweep ratio chosen once at
server startup from ``CACHEBLEND_RATIOS``), CacheTune's repair ratio is
chosen **dynamically per request** by a `CacheTuneController` from a
roofline model of real, measured hardware costs -- see
`hardware_profile.py` and `controller.py`. This module owns that
controller instance (one per server process) plus the deployment-wide
hardware measurement it is seeded with.
"""

import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from sglang.srt.mem_cache.approx_kv.types import (
    DenseRange,
    KVReusePlan,
    RecoveryMode,
    SchedulerMetadata,
)

from .controller import CacheTuneController
from .hardware_profile import CacheTuneMode, HardwareMeasurement
from .recompute import CacheTuneLayerRecomputeBackend, CacheTuneProbeBackend
from .token_selection import GradualFilterStage

CACHETUNE_PLUGIN_NAME = "cachetune"

_MODE_BY_ENV_VALUE: dict[str, CacheTuneMode] = {
    "paper_mechanism": CacheTuneMode.PAPER_MECHANISM,
    "speed_only": CacheTuneMode.SPEED_ONLY,
}

_MEASUREMENT_ENV_KEYS: tuple[str, str, str] = (
    "SGLANG_CACHETUNE_T_C_MS",
    "SGLANG_CACHETUNE_T_I_MS",
    "SGLANG_CACHETUNE_T_O_MS",
)


def _parse_mode(raw_value: str) -> CacheTuneMode:
    normalized = raw_value.strip().lower()
    try:
        return _MODE_BY_ENV_VALUE[normalized]
    except KeyError as exc:
        raise ValueError(
            "SGLANG_CACHETUNE_MODE must be one of "
            f"{sorted(_MODE_BY_ENV_VALUE)}, got {raw_value!r}"
        ) from exc


def _detect_hardware_tier() -> str:
    """Auto-detect a hardware-tier label with no fabricated fallback.

    Safe to call without a GPU present: ``torch.cuda.is_available()``
    never raises. Falls back to the plain label ``"cpu"`` only when no
    CUDA device is visible at all -- this is an honest description of
    the actual device, not a placeholder value.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _parse_deployment_measurement(
    env: Mapping[str, str],
) -> HardwareMeasurement | None:
    """Parse the required-together deployment-wide hardware measurement.

    Returns ``None`` when none of the three env vars are set at all (no
    deployment measurement has been configured yet -- the runtime must
    then honestly dense-fallback rather than fabricate one). Raises if
    only some of the three are set, since a partial measurement is very
    likely an operator misconfiguration, not an intentional "no
    measurement" state.
    """
    present = {
        key: env[key] for key in _MEASUREMENT_ENV_KEYS if env.get(key, "").strip()
    }
    if not present:
        return None
    if len(present) != len(_MEASUREMENT_ENV_KEYS):
        missing = [key for key in _MEASUREMENT_ENV_KEYS if key not in present]
        raise ValueError(
            "SGLANG_CACHETUNE_T_C_MS / T_I_MS / T_O_MS must be supplied "
            f"together as one deployment profiling pass; missing: {missing}"
        )
    try:
        t_c_ms = float(present["SGLANG_CACHETUNE_T_C_MS"])
        t_i_ms = float(present["SGLANG_CACHETUNE_T_I_MS"])
        t_o_ms = float(present["SGLANG_CACHETUNE_T_O_MS"])
    except ValueError as exc:
        raise ValueError(
            "SGLANG_CACHETUNE_T_C_MS / T_I_MS / T_O_MS must be valid "
            "floating point millisecond values"
        ) from exc
    return HardwareMeasurement(t_c_ms=t_c_ms, t_i_ms=t_i_ms, t_o_ms=t_o_ms)


@dataclass(frozen=True)
class CacheTuneConfig:
    """Deployment-wide CacheTune configuration.

    ``mode`` selects between the paper's quality-preserving 15% ratio
    floor (``PAPER_MECHANISM``) and this project's TTFT-only 0% floor
    (``SPEED_ONLY``); there is no default -- see ``from_env``.
    ``hardware_tier`` identifies the compute device for
    ``HardwareProfileKey`` scoping. ``probe_stages``/
    ``first_recompute_layer`` configure the repair-token selection funnel
    (see ``token_selection.py``/``recompute.py``).
    ``deployment_measurement`` is the single real, operator-supplied
    ``t_c``/``t_i``/``t_o`` measurement from one deployment profiling
    pass (paper: "one deployment profiling/calibration" cadence); it may
    be ``None`` if the server was started without it, in which case the
    real request path must dense-fallback honestly rather than fabricate
    a measurement (see ``runtime.py``).
    """

    mode: CacheTuneMode
    hardware_tier: str
    probe_stages: tuple[GradualFilterStage, ...]
    first_recompute_layer: int
    deployment_measurement: HardwareMeasurement | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CacheTuneMode):
            raise TypeError("mode must be a CacheTuneMode")
        if not self.hardware_tier.strip():
            raise ValueError("hardware_tier must be non-empty")
        if not self.probe_stages:
            raise ValueError("CacheTune requires at least one probe stage")
        if self.first_recompute_layer < 0:
            raise ValueError("first_recompute_layer must be non-negative")
        max_probe_layer = max(stage.probe_layer_id for stage in self.probe_stages)
        if max_probe_layer >= self.first_recompute_layer:
            raise ValueError(
                "first_recompute_layer must be strictly deeper than every "
                "probe stage's layer"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CacheTuneConfig:
        """Build the deployment configuration from environment variables.

        ``SGLANG_CACHETUNE_MODE`` (required, no default): ``paper_mechanism``
        or ``speed_only``.
        ``SGLANG_CACHETUNE_HARDWARE_TIER`` (optional): overrides
        auto-detection (``torch.cuda.get_device_name(0)`` else ``"cpu"``).
        ``SGLANG_CACHETUNE_PROBE_LAYERS`` (default ``"0"``): comma
        separated probe layer ids for the repair-token selection funnel,
        in shallow-to-deep order.
        ``SGLANG_CACHETUNE_STAGE_KEEP_RATIO`` (default ``0.5``): fraction
        of the current candidate pool each funnel stage keeps.
        ``SGLANG_CACHETUNE_FIRST_RECOMPUTE_LAYER`` (default ``1``): first
        layer that gets a real per-selected-token repair.
        ``SGLANG_CACHETUNE_T_C_MS`` / ``_T_I_MS`` / ``_T_O_MS``
        (optional, required together): the deployment-wide hardware
        measurement; omit all three to start with no measurement
        configured (requests will honestly dense-fallback until one is
        recorded).
        """
        env = os.environ if env is None else env
        mode_value = env.get("SGLANG_CACHETUNE_MODE")
        if mode_value is None:
            raise ValueError(
                "SGLANG_CACHETUNE_MODE is required when CacheTune is "
                "enabled (one of 'paper_mechanism', 'speed_only'); there "
                "is no default mode"
            )
        mode = _parse_mode(mode_value)
        hardware_tier = (
            env.get("SGLANG_CACHETUNE_HARDWARE_TIER") or _detect_hardware_tier()
        )
        probe_layers = [
            int(layer.strip())
            for layer in env.get("SGLANG_CACHETUNE_PROBE_LAYERS", "0").split(",")
            if layer.strip()
        ]
        stage_keep_ratio = float(env.get("SGLANG_CACHETUNE_STAGE_KEEP_RATIO", "0.5"))
        stages = tuple(
            GradualFilterStage(probe_layer_id=layer_id, keep_ratio=stage_keep_ratio)
            for layer_id in probe_layers
        )
        first_recompute_layer = int(
            env.get("SGLANG_CACHETUNE_FIRST_RECOMPUTE_LAYER", "1")
        )
        deployment_measurement = _parse_deployment_measurement(env)
        return cls(
            mode=mode,
            hardware_tier=hardware_tier,
            probe_stages=stages,
            first_recompute_layer=first_recompute_layer,
            deployment_measurement=deployment_measurement,
        )


class CacheTuneRecoveryPlugin:
    """Registered under ``RecoveryPluginRegistry`` as ``"cachetune"``.

    Implements the common-core ``RecoveryPlugin`` protocol (``name``,
    ``build_plan``, ``scheduler_metadata``) for identity/registration.
    The *actual* selective per-layer repair is driven by
    ``cachetune.runtime.restore_request_prefix_cachetune`` rather than by
    ``build_plan`` + ``execute_reuse_plan`` -- like CacheBlend, CacheTune's
    mixed reuse+selective-repair semantics do not fit the generic
    contiguous copy-or-dense ``KVReusePlan`` model. ``build_plan``
    therefore returns a conservative, honest "requires online repair
    execution" dense plan for any generic caller that only uses the
    offline planning contract, and never fabricates a selection it has
    not actually run.
    """

    def __init__(
        self,
        *,
        config: CacheTuneConfig,
        controller: CacheTuneController,
        probe_backend: CacheTuneProbeBackend | None = None,
        recompute_backend: CacheTuneLayerRecomputeBackend | None = None,
    ) -> None:
        self.config = config
        self.controller = controller
        self.probe_backend = probe_backend
        self.recompute_backend = recompute_backend

    @property
    def name(self) -> str:
        return CACHETUNE_PLUGIN_NAME

    @property
    def capable(self) -> bool:
        """True only if both real per-layer hooks are bound.

        This is the capability guard: without a real probe backend
        repair-token deviation cannot be measured (only fabricated), and
        without a real recompute backend selected tokens cannot actually
        be repaired. Either gap means the request path must
        dense-fallback rather than claim a CacheTune repair happened.
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
                    reason="cachetune_requires_online_repair_execution",
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


def maybe_register_cachetune_plugin(
    manager: Any,
    *,
    env: Mapping[str, str] | None = None,
    probe_backend: CacheTuneProbeBackend | None = None,
    recompute_backend: CacheTuneLayerRecomputeBackend | None = None,
) -> CacheTuneRecoveryPlugin | None:
    """Register the CacheTune plugin into ``manager`` when explicitly
    enabled via ``SGLANG_APPROX_KV_CACHETUNE=1``.

    ``probe_backend``/``recompute_backend`` default to ``None``: this
    fork does not yet have a ModelRunner-level hook that can run a real,
    single-batched per-layer forward for an arbitrary token subset inside
    an otherwise-cached prefix (see ``recompute.py`` and
    ``cachetune/runtime.py`` module docstrings). Registering the plugin
    with no bound backends is still meaningful and honest: it wires the
    real request-path dispatch (``schedule_batch.py`` routes
    ``plugin="cachetune"`` requests here), constructs the real
    ``CacheTuneController`` from real (or absent) hardware measurements,
    and exercises the capability guard, which correctly dense-falls-back
    every such request until a real backend is bound -- it must never
    silently claim a repaired restore that did not happen.
    """
    env = os.environ if env is None else env
    if not _read_bool(env, "SGLANG_APPROX_KV_CACHETUNE", False):
        return None
    config = CacheTuneConfig.from_env(env)
    controller = CacheTuneController(config.mode)
    plugin = CacheTuneRecoveryPlugin(
        config=config,
        controller=controller,
        probe_backend=probe_backend,
        recompute_backend=recompute_backend,
    )
    manager.register_plugin(plugin)
    return plugin
