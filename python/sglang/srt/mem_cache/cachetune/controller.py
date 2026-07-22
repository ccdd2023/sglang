from __future__ import annotations

"""The CacheTune hardware-aware repair-ratio controller.

`CacheTuneController` is the stateful object the runtime consults once
per request: it holds real, measured `HardwareMeasurement`s keyed by
`HardwareProfileKey`, derives the roofline `r0`, optionally refines it
with a small measured-TTFT calibration set (golden-section search,
warm-started at `r0`, operating only over *executable* quantized
ratios), and returns a fully-quantized, bounds-respecting decision.

This module intentionally never fabricates a measurement: every method
that needs a `HardwareMeasurement` for a given profile raises
`CacheTuneProfileError` if none was recorded, rather than silently
reusing another profile's numbers or inventing a default.
"""

import math
from dataclasses import dataclass
from typing import Callable, Literal

from .golden_section import golden_section_search_minimize, warm_start_bracket
from .hardware_profile import (
    CacheTuneMode,
    HardwareMeasurement,
    HardwareProfileKey,
    QuantizedRatio,
    RatioBounds,
    predict_ttft_ms,
    quantize_ratio,
    roofline_ratio,
)

RatioSource = Literal["roofline", "calibrated"]


class CacheTuneProfileError(RuntimeError):
    """Raised when a `HardwareProfileKey` has no recorded measurement."""


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of one `CacheTuneController.calibrate(...)` call.

    `ratio` is the best-observed *executable* ratio among every point
    the search actually measured (`probes`); `probes` is the ordered,
    de-duplicated set of `(executable_ratio, measured_ttft_ms)` pairs
    that were really evaluated -- i.e. the "small calibration set" the
    paper describes -- and `warm_start_ratio` records the roofline `r0`
    the search was centered on.
    """

    key: HardwareProfileKey
    context_length: int
    ratio: float
    quantized: QuantizedRatio
    probes: tuple[tuple[float, float], ...]
    warm_start_ratio: float

    def __post_init__(self) -> None:
        if not self.probes:
            raise ValueError("a calibration result must record at least one probe")


@dataclass(frozen=True)
class CacheTuneDecision:
    """The controller's full, telemetry-ready decision for one request."""

    key: HardwareProfileKey
    context_length: int
    num_layers: int
    mode: CacheTuneMode
    bounds: RatioBounds
    roofline_ratio: float
    source: RatioSource
    quantized: QuantizedRatio
    predicted_ttft_ms: float

    @property
    def repair_tokens(self) -> int:
        return self.quantized.repair_tokens

    @property
    def executable_ratio(self) -> float:
        return self.quantized.executable_ratio


class CacheTuneController:
    """Hardware-aware repair-ratio controller (Phase 4 R5 CacheTune subset).

    Implements only the controller half of CacheTune (arXiv
    2605.24022v1): the roofline-informed, hardware-profile-scoped choice
    of the recomputation-vs-transfer ratio `r`. It does not implement the
    paper's frequency-domain token selection, sparse transfer,
    multi-stream overlap or deferred RoPE -- see
    `cachetune/__init__.py` for the full scope statement.
    """

    def __init__(self, mode: CacheTuneMode, *, r_max: float = 1.0) -> None:
        if not isinstance(mode, CacheTuneMode):
            raise TypeError("mode must be a CacheTuneMode")
        self._mode = mode
        self._bounds = RatioBounds.for_mode(mode, r_max=r_max)
        self._measurements: dict[HardwareProfileKey, HardwareMeasurement] = {}
        self._calibrations: dict[HardwareProfileKey, CalibrationResult] = {}

    @property
    def mode(self) -> CacheTuneMode:
        return self._mode

    @property
    def bounds(self) -> RatioBounds:
        return self._bounds

    def has_measurement(self, key: HardwareProfileKey) -> bool:
        return key in self._measurements

    def measurement(self, key: HardwareProfileKey) -> HardwareMeasurement:
        return self._require_measurement(key)

    def record_measurement(
        self,
        key: HardwareProfileKey,
        measurement: HardwareMeasurement,
    ) -> None:
        if not isinstance(key, HardwareProfileKey):
            raise TypeError("key must be a HardwareProfileKey")
        if not isinstance(measurement, HardwareMeasurement):
            raise TypeError("measurement must be a HardwareMeasurement")
        self._measurements[key] = measurement
        # A new measurement invalidates any calibration computed under a
        # previous measurement for this exact profile -- never let a
        # stale calibration silently outlive the data it was based on.
        self._calibrations.pop(key, None)

    def _require_measurement(self, key: HardwareProfileKey) -> HardwareMeasurement:
        try:
            return self._measurements[key]
        except KeyError as exc:
            raise CacheTuneProfileError(
                f"no hardware measurement recorded for profile {key!r}"
            ) from exc

    def roofline(self, key: HardwareProfileKey) -> float:
        return roofline_ratio(self._require_measurement(key))

    def has_calibration(self, key: HardwareProfileKey) -> bool:
        return key in self._calibrations

    def calibration(self, key: HardwareProfileKey) -> CalibrationResult | None:
        return self._calibrations.get(key)

    def calibrate(
        self,
        key: HardwareProfileKey,
        *,
        context_length: int,
        evaluate: Callable[[float], float],
        warm_start_span: float = 0.6,
        tol: float = 0.02,
        max_iterations: int = 12,
    ) -> CalibrationResult:
        """Refine `r*` for `key` using a small measured-TTFT calibration set.

        `evaluate(ratio)` must return a real measured mean TTFT
        (milliseconds) for the given *executable* ratio -- in production
        this wraps a handful of real requests issued at that ratio and
        averaged, matching the paper's "small calibration set". Every
        ratio the golden-section search considers is first snapped to
        the nearest executable token count for `context_length` (see
        `quantize_ratio`), and results are memoized per exact executable
        ratio within this call so the same ratio is never measured
        twice -- keeping the calibration set small and the result
        deterministic for a deterministic `evaluate`.

        The search brackets a warm-start window around the roofline
        `r0` inside `[bounds.r_min, bounds.r_max]` (see
        `warm_start_bracket`), then reports the minimum *actually probed*
        point (ties broken toward the smaller ratio) rather than trusting
        only the search's final bracket midpoint, so measurement noise
        cannot make the reported ratio worse than everything that was
        really measured.
        """
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        measurement = self._require_measurement(key)
        r0 = roofline_ratio(measurement)
        lo, hi = warm_start_bracket(
            r0,
            self._bounds.r_min,
            self._bounds.r_max,
            span=warm_start_span,
        )

        probe_order: list[float] = []
        probe_values: dict[float, float] = {}

        def snapped_evaluate(ratio: float) -> float:
            quantized = quantize_ratio(
                ratio,
                context_length=context_length,
                bounds=self._bounds,
            )
            executable = quantized.executable_ratio
            if executable not in probe_values:
                measured = evaluate(executable)
                if not math.isfinite(measured):
                    raise ValueError("evaluate(ratio) must return a finite measurement")
                probe_values[executable] = measured
                probe_order.append(executable)
            return probe_values[executable]

        golden_section_search_minimize(
            snapped_evaluate,
            lo,
            hi,
            tol=tol,
            max_iterations=max_iterations,
        )

        # Deterministic argmin over every point actually probed: minimum
        # measured TTFT first, smaller ratio as an explicit tie-break.
        best_ratio = min(
            probe_order,
            key=lambda ratio: (probe_values[ratio], ratio),
        )
        result = CalibrationResult(
            key=key,
            context_length=context_length,
            ratio=best_ratio,
            quantized=quantize_ratio(
                best_ratio,
                context_length=context_length,
                bounds=self._bounds,
            ),
            probes=tuple((ratio, probe_values[ratio]) for ratio in probe_order),
            warm_start_ratio=r0,
        )
        self._calibrations[key] = result
        return result

    def select_ratio(
        self,
        key: HardwareProfileKey,
        context_length: int,
        num_layers: int,
    ) -> CacheTuneDecision:
        """Return the controller's decision for one real request.

        Uses the cached calibration for `key` if one has been computed
        (`source="calibrated"`); otherwise falls back to the roofline
        analytic optimum (`source="roofline"`). Either way the result is
        deterministically quantized to an executable integer repair
        token count for the *exact* `context_length` of this request
        (the profile key's `chunk_length_bucket` only selects which
        measurement/calibration to use -- the final quantization always
        uses the real, precise token count).
        """
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        measurement = self._require_measurement(key)
        calibration = self._calibrations.get(key)
        if calibration is not None:
            source: RatioSource = "calibrated"
            ratio = calibration.ratio
        else:
            source = "roofline"
            ratio = roofline_ratio(measurement)

        quantized = quantize_ratio(
            ratio,
            context_length=context_length,
            bounds=self._bounds,
        )
        predicted_ttft_ms = predict_ttft_ms(
            measurement,
            num_layers=num_layers,
            context_length=context_length,
            ratio=quantized.executable_ratio,
        )
        return CacheTuneDecision(
            key=key,
            context_length=context_length,
            num_layers=num_layers,
            mode=self._mode,
            bounds=self._bounds,
            roofline_ratio=roofline_ratio(measurement),
            source=source,
            quantized=quantized,
            predicted_ttft_ms=predicted_ttft_ms,
        )
