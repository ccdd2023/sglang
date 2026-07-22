from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..types import RecoveryMode, ResidencyTier


@dataclass(frozen=True)
class RecoveryMeasurement:
    mode: RecoveryMode
    token_count: int
    source_tier: ResidencyTier
    dense_ms: float
    h2d_ms: float = 0.0
    recovery_ms: float = 0.0
    last_token_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if (
            min(
                self.dense_ms,
                self.h2d_ms,
                self.recovery_ms,
                self.last_token_ms,
            )
            < 0
        ):
            raise ValueError("latency measurements must be non-negative")

    @property
    def predicted_ttft_ms(self) -> float:
        if self.mode == RecoveryMode.DENSE:
            return self.dense_ms + self.last_token_ms
        return self.h2d_ms + self.recovery_ms + self.last_token_ms

    @property
    def saved_ms(self) -> float:
        return self.dense_ms + self.last_token_ms - self.predicted_ttft_ms


@dataclass(frozen=True)
class RecoverySelection:
    mode: RecoveryMode
    predicted_ttft_ms: float
    saved_ms: float
    profile_token_count: int
    source_tier: ResidencyTier


class HardwareAwareRecoverySelector:
    def __init__(
        self,
        measurements: Iterable[RecoveryMeasurement] = (),
    ) -> None:
        self._measurements = list(measurements)

    def add(self, measurement: RecoveryMeasurement) -> None:
        self._measurements.append(measurement)

    def select(
        self,
        *,
        token_count: int,
        source_tier: ResidencyTier,
        allowed_modes: set[RecoveryMode] | None = None,
    ) -> RecoverySelection:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        allowed_modes = allowed_modes or set(RecoveryMode)
        candidates = [
            measurement
            for measurement in self._measurements
            if measurement.source_tier == source_tier
            and measurement.mode in allowed_modes
        ]
        if not candidates:
            raise LookupError(
                "no recovery profile matches the requested tier and modes"
            )

        nearest_by_mode: dict[RecoveryMode, RecoveryMeasurement] = {}
        for measurement in candidates:
            previous = nearest_by_mode.get(measurement.mode)
            if previous is None or self._distance(
                token_count,
                measurement.token_count,
            ) < self._distance(token_count, previous.token_count):
                nearest_by_mode[measurement.mode] = measurement

        selected = min(
            nearest_by_mode.values(),
            key=lambda measurement: (
                measurement.predicted_ttft_ms,
                measurement.mode.value,
            ),
        )
        return RecoverySelection(
            mode=selected.mode,
            predicted_ttft_ms=selected.predicted_ttft_ms,
            saved_ms=selected.saved_ms,
            profile_token_count=selected.token_count,
            source_tier=selected.source_tier,
        )

    @staticmethod
    def _distance(requested: int, measured: int) -> float:
        return abs(math.log2(requested) - math.log2(measured))
