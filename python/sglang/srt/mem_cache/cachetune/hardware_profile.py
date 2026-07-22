from __future__ import annotations

"""Hardware-profile types and roofline math for the CacheTune controller.

This module implements the *measurement and math* half of the CacheTune
hardware-aware repair controller (arXiv 2605.24022v1, "CacheTune"): a
typed hardware/workload profile key, a validated measurement of the two
overlappable critical paths the paper identifies (per-layer recomputation
and per-layer external-cache transfer), the closed-form roofline optimum
`r0`, the two explicit ratio-floor modes this project distinguishes
(`paper-mechanism` keeps the paper's 15% quality floor; `speed-only`
allows a 0% floor since this project does not optimize output quality),
and deterministic quantization from a continuous ratio down to an
*executable* integer repair-token count for a concrete context length.

Scope note: this is a "CacheTune hardware-controller inspired subset".
The full paper additionally specifies frequency-domain token selection,
sparse transfer, multi-stream overlap and deferred RoPE; none of those
are implemented here (see `cachetune/__init__.py` for the full scope
statement). Only the controller (this module + `golden_section.py` +
`controller.py`) is a faithful implementation of the paper's roofline
model and calibration procedure.
"""

import math
from dataclasses import dataclass
from enum import Enum


class CacheTuneMode(str, Enum):
    """Which ratio-floor convention governs `RatioBounds.for_mode`.

    `PAPER_MECHANISM` reproduces the paper's quality-preserving `r_min =
    15%` floor (Section on roofline-guided ratio selection). `SPEED_ONLY`
    is this project's explicit non-paper mode that allows `r_min = 0%`
    because TTFT is the only metric tracked here; it must never be
    presented as the paper's original setting.
    """

    PAPER_MECHANISM = "paper_mechanism"
    SPEED_ONLY = "speed_only"


PAPER_MECHANISM_R_MIN: float = 0.15


@dataclass(frozen=True)
class HardwareProfileKey:
    """Identity of one calibrated hardware/workload profile.

    `hardware_tier` names the compute device or device class (e.g. a GPU
    name string or an explicit tier label). `model_fingerprint` ties the
    profile to a specific model/dtype deployment (the same fingerprint
    concept already used by `approx_kv.types.KVSegmentKey`).
    `chunk_length_bucket` is a coarse, positive-integer bucket of the
    context length the measurement applies to (see
    `chunk_length_bucket()` below for the canonical bucketing function);
    keeping it a plain positive int here -- rather than enforcing the
    power-of-two bucketing policy in the key itself -- lets tests
    construct isolated keys directly without coupling this dataclass to
    one specific bucketing scheme.
    """

    hardware_tier: str
    model_fingerprint: str
    chunk_length_bucket: int

    def __post_init__(self) -> None:
        if not self.hardware_tier.strip():
            raise ValueError("hardware_tier must be non-empty")
        if not self.model_fingerprint.strip():
            raise ValueError("model_fingerprint must be non-empty")
        if self.chunk_length_bucket <= 0:
            raise ValueError("chunk_length_bucket must be positive")


def chunk_length_bucket(context_length: int) -> int:
    """Canonical chunk-length bucketing policy: next power of two.

    Coarsely groups nearby context lengths (e.g. 500 and 620 tokens both
    land in the 1024 bucket) so a single calibrated measurement can cover
    a realistic spread of request sizes without requiring a fresh
    calibration for every distinct token count.
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    return 1 << (context_length - 1).bit_length()


@dataclass(frozen=True)
class HardwareMeasurement:
    """Real, measured per-layer costs for the two CacheTune critical paths.

    All three fields are **per single transformer layer** costs in
    milliseconds, matching the paper's per-layer `T_layer(r) = max(r * N
    * t_c, (1 - r) * N * t_i) + t_o` model:

    * `t_c_ms`: per-layer, per-token recomputation cost.
    * `t_i_ms`: per-layer, per-token effective external-cache transfer
      cost.
    * `t_o_ms`: per-layer, fixed pipeline overhead (independent of `N`
      and `r`).

    `sample_count` records how many real timing samples this measurement
    was derived from, purely for telemetry/confidence reporting; it does
    not change the math.
    """

    t_c_ms: float
    t_i_ms: float
    t_o_ms: float
    sample_count: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.t_c_ms) or self.t_c_ms <= 0:
            raise ValueError("t_c_ms must be a positive, finite number")
        if not math.isfinite(self.t_i_ms) or self.t_i_ms <= 0:
            raise ValueError("t_i_ms must be a positive, finite number")
        if not math.isfinite(self.t_o_ms) or self.t_o_ms < 0:
            raise ValueError("t_o_ms must be a non-negative, finite number")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")


def roofline_ratio(measurement: HardwareMeasurement) -> float:
    """Closed-form roofline optimum `r0 = t_i / (t_c + t_i)`.

    This is the ratio that equalizes the two overlapped critical paths
    (`r * N * t_c == (1 - r) * N * t_i`), which is where `max(...)` -- and
    therefore `T_layer(r)` -- is minimized, independent of `N` and `t_o`.
    Always strictly within `(0, 1)` given the positivity constraints on
    `t_c_ms`/`t_i_ms` enforced by `HardwareMeasurement`.
    """
    return measurement.t_i_ms / (measurement.t_c_ms + measurement.t_i_ms)


def predict_layer_time_ms(
    measurement: HardwareMeasurement,
    *,
    context_length: int,
    ratio: float,
) -> float:
    """`T_layer(r) = max(r*N*t_c, (1-r)*N*t_i) + t_o` for one layer."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("ratio must be within [0, 1]")
    recompute_ms = ratio * context_length * measurement.t_c_ms
    transfer_ms = (1.0 - ratio) * context_length * measurement.t_i_ms
    return max(recompute_ms, transfer_ms) + measurement.t_o_ms


def predict_ttft_ms(
    measurement: HardwareMeasurement,
    *,
    num_layers: int,
    context_length: int,
    ratio: float,
) -> float:
    """`T_TTFT(r) ~= L * (max(r*N*t_c, (1-r)*N*t_i) + t_o)` steady state."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    return num_layers * predict_layer_time_ms(
        measurement,
        context_length=context_length,
        ratio=ratio,
    )


@dataclass(frozen=True)
class RatioBounds:
    """Inclusive `[r_min, r_max]` bounds enforced on the selected ratio."""

    r_min: float
    r_max: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.r_min <= 1.0):
            raise ValueError("r_min must be within [0, 1]")
        if not (0.0 <= self.r_max <= 1.0):
            raise ValueError("r_max must be within [0, 1]")
        if self.r_min > self.r_max:
            raise ValueError("r_min must not exceed r_max")

    def clamp(self, ratio: float) -> float:
        if not math.isfinite(ratio):
            raise ValueError("ratio must be a finite number")
        return min(max(ratio, self.r_min), self.r_max)

    @classmethod
    def for_mode(cls, mode: CacheTuneMode, *, r_max: float = 1.0) -> RatioBounds:
        """Explicit, non-defaulted construction of the mode's bounds.

        `PAPER_MECHANISM` reproduces the paper's `r_min = 15%` quality
        floor; `SPEED_ONLY` is this project's explicit non-paper mode
        that allows `r_min = 0%`. Both must be requested explicitly by
        the caller -- there is no ambient default mode.
        """
        if mode is CacheTuneMode.PAPER_MECHANISM:
            return cls(r_min=PAPER_MECHANISM_R_MIN, r_max=r_max)
        if mode is CacheTuneMode.SPEED_ONLY:
            return cls(r_min=0.0, r_max=r_max)
        raise ValueError(f"unknown CacheTuneMode: {mode!r}")


@dataclass(frozen=True)
class QuantizedRatio:
    """A continuous ratio decision resolved to an executable token count.

    Repair work can only ever cover an integer number of tokens, so
    `repair_tokens` (and the `executable_ratio` re-derived from it) is
    the single source of truth callers must use; `requested_ratio` and
    `bounded_ratio` are retained purely for telemetry/debugging.
    """

    requested_ratio: float
    bounded_ratio: float
    context_length: int
    repair_tokens: int
    executable_ratio: float


def round_half_up(value: float) -> int:
    """Deterministic round-half-up (never Python's round-half-to-even).

    `round(0.5)` in Python is banker's rounding (rounds to even), which
    is a surprising, easy-to-miss tie-break for a token-count computation
    that feeds directly into "how many tokens get dense-recomputed".
    This project uses a single, explicit, always-round-up-at-.5 tie-break
    instead, and tests it directly at an exact `.5` boundary.

    Public (not module-private) because `token_selection.py`'s funnel
    filtering also needs the exact same deterministic tie-break when
    shrinking its candidate pool, and reusing one definition guarantees
    every token-count rounding decision in this package agrees.
    """
    return math.floor(value + 0.5)


def quantize_ratio(
    ratio: float,
    *,
    context_length: int,
    bounds: RatioBounds,
) -> QuantizedRatio:
    """Deterministically map a continuous ratio to an executable count.

    1. Clamp `ratio` into `[bounds.r_min, bounds.r_max]`.
    2. Compute the admissible integer token-count range implied by the
       bounds for this exact `context_length` (`ceil(r_min * N)` .. `floor(r_max
       * N)`, with a small epsilon to absorb floating-point representation
       error at exact boundaries such as `0.15 * 100 == 15`).
    3. Round the bounded ratio's token count with `_round_half_up` and
       clamp it into the admissible range.

    Raises `ValueError` if the bounds admit no integer token count for
    this `context_length` (should not happen for the bounds this module
    constructs, but a caller-supplied `RatioBounds` could in principle be
    degenerate for a very small `context_length`).
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if not math.isfinite(ratio):
        raise ValueError("ratio must be a finite number")

    bounded = bounds.clamp(ratio)
    eps = 1e-9
    min_tokens = max(0, math.ceil(bounds.r_min * context_length - eps))
    max_tokens = min(context_length, math.floor(bounds.r_max * context_length + eps))
    if min_tokens > max_tokens:
        raise ValueError(
            "ratio bounds admit no executable token count for this " "context length"
        )

    raw_tokens = round_half_up(bounded * context_length)
    tokens = min(max(raw_tokens, min_tokens), max_tokens)
    return QuantizedRatio(
        requested_ratio=ratio,
        bounded_ratio=bounded,
        context_length=context_length,
        repair_tokens=tokens,
        executable_ratio=tokens / context_length,
    )


@dataclass(frozen=True)
class DenseTimingSample:
    """One real, measured dense (fully recomputed) TTFT sample."""

    context_length: int
    ttft_ms: float

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if not math.isfinite(self.ttft_ms) or self.ttft_ms <= 0:
            raise ValueError("ttft_ms must be a positive, finite number")


@dataclass(frozen=True)
class TransferTimingSample:
    """One real, measured raw-copy-plus-RoPE transfer timing sample.

    `copy_ms`/`rope_ms` are the aggregate (all-layers) costs already
    reported by `approx_kv.types.KVTransferStats` for a transfer covering
    `tokens` tokens -- see `RadixKVTransferBackend.copy_and_rotate`.
    """

    tokens: int
    copy_ms: float
    rope_ms: float

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("tokens must be positive")
        if not math.isfinite(self.copy_ms) or self.copy_ms < 0:
            raise ValueError("copy_ms must be a non-negative, finite number")
        if not math.isfinite(self.rope_ms) or self.rope_ms < 0:
            raise ValueError("rope_ms must be a non-negative, finite number")


def estimate_measurement_from_samples(
    *,
    dense_small: DenseTimingSample,
    dense_large: DenseTimingSample,
    transfer: TransferTimingSample,
    num_layers: int,
) -> HardwareMeasurement:
    """Derive a real `HardwareMeasurement` from genuine timing samples.

    `t_c_ms` is recovered from the *marginal* (finite-difference) slope
    between two real dense-TTFT samples at different context lengths,
    which cancels out fixed, non-layer overhead (tokenization, sampling,
    HTTP, etc.) that a single absolute TTFT sample would otherwise
    conflate with per-token recompute cost. `t_o_ms` is then the
    remaining per-layer fixed-overhead component implied by
    `dense_small`. `t_i_ms` is recovered directly from one real transfer
    sample's instrumented `copy_ms + rope_ms`, which already isolates
    the transfer path (no recompute is involved).

    This is a genuine measurement-derivation function operating on
    caller-supplied real numbers; it never fabricates or guesses a
    measurement.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if dense_large.context_length <= dense_small.context_length:
        raise ValueError(
            "dense_large must use a strictly greater context length than " "dense_small"
        )

    delta_tokens = dense_large.context_length - dense_small.context_length
    delta_ms = dense_large.ttft_ms - dense_small.ttft_ms
    if delta_ms <= 0:
        raise ValueError(
            "dense TTFT did not increase with context length; cannot "
            "derive a positive recompute cost from these samples"
        )
    t_c_ms = delta_ms / (delta_tokens * num_layers)

    overhead_total_ms = dense_small.ttft_ms - (
        t_c_ms * num_layers * dense_small.context_length
    )
    if overhead_total_ms < 0:
        raise ValueError(
            "derived per-layer overhead is negative; timing samples are "
            "inconsistent with the roofline model"
        )
    t_o_ms = overhead_total_ms / num_layers

    transfer_total_ms = transfer.copy_ms + transfer.rope_ms
    if transfer_total_ms <= 0:
        raise ValueError("transfer timing must be positive to derive a transfer cost")
    t_i_ms = transfer_total_ms / (transfer.tokens * num_layers)

    return HardwareMeasurement(
        t_c_ms=t_c_ms,
        t_i_ms=t_i_ms,
        t_o_ms=t_o_ms,
        sample_count=3,
    )
