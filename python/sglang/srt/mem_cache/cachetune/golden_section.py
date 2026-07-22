from __future__ import annotations

"""Generic golden-section search minimizer used by the CacheTune controller.

Pure numeric utilities with no dependency on hardware-profile types, so
they can be validated independently against known unimodal functions
before being trusted to refine a real, noisy measured-TTFT curve.
"""

import math
from typing import Callable

_INV_PHI: float = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
_INV_PHI2: float = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382, i.e. 1 - _INV_PHI


def golden_section_search_minimize(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-4,
    max_iterations: int = 100,
) -> float:
    """Minimize a (assumed unimodal) real function `f` over `[lo, hi]`.

    Standard golden-section search: at each step two interior points `c
    < d` split the bracket at the golden ratio, `f` is evaluated at
    whichever of the two is new, and the worse half of the bracket is
    discarded. Converges to within `tol` of the true minimizer for any
    strictly unimodal `f` (in particular, the roofline `T_layer(r)` /
    `T_TTFT(r)` formulas, which are a positively-weighted `max` of two
    monotonic-in-opposite-directions linear functions of `r` plus a
    constant, and are therefore convex, hence unimodal).

    Tie-break: on an exact tie (`f(c) == f(d)`), this implementation
    discards the *upper* half of the bracket (keeps `[lo, d]`), which
    deterministically biases the search toward the **smaller** ratio
    when `f` is perfectly flat across a sub-interval. This matters for
    CacheTune because a smaller ratio means less recompute work; ties
    should never non-deterministically prefer more work.
    """
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    a, b = float(lo), float(hi)
    if b - a <= tol:
        return (a + b) / 2.0

    span = b - a
    c = a + _INV_PHI2 * span
    d = a + _INV_PHI * span
    fc, fd = f(c), f(d)

    for _ in range(max_iterations):
        if b - a <= tol:
            break
        if fc <= fd:
            # Tie-break: `<=` (not `<`) keeps the lower half on an exact
            # tie, biasing convergence toward the smaller ratio.
            b = d
            d = c
            fd = fc
            span = b - a
            c = a + _INV_PHI2 * span
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            span = b - a
            d = a + _INV_PHI * span
            fd = f(d)

    return (a + b) / 2.0


def warm_start_bracket(
    r0: float,
    r_min: float,
    r_max: float,
    *,
    span: float = 0.6,
) -> tuple[float, float]:
    """Narrow `[r_min, r_max]` to a window centered on the roofline `r0`.

    This is what makes the calibration search "roofline warm-started"
    rather than a blind search over the full bounds: the analytic `r0`
    is used to bias the initial bracket, reducing how many real
    (expensive) measured-TTFT samples the golden-section search needs to
    converge. `span` is the fraction of the *full* `[r_min, r_max]` range
    the warm-start window covers (default 60%); if `r0` sits outside
    `[r_min, r_max]` or the centered window degenerates, the full bounds
    are used unmodified rather than silently producing an empty bracket.
    """
    if not (0.0 < span <= 1.0):
        raise ValueError("span must be within (0, 1]")
    if r_min > r_max:
        raise ValueError("r_min must not exceed r_max")
    if not math.isfinite(r0):
        raise ValueError("r0 must be a finite number")

    half_width = (r_max - r_min) * span / 2.0
    center = min(max(r0, r_min), r_max)
    lo = max(r_min, center - half_width)
    hi = min(r_max, center + half_width)
    if lo >= hi:
        return r_min, r_max
    return lo, hi
