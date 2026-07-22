from __future__ import annotations

"""CacheTune hardware-controller inspired subset (Phase 4 R5).

Implements a roofline-based, hardware-profile-scoped recompute-vs-transfer
repair-ratio controller inspired by CacheTune (arXiv 2605.24022v1), wired
into a real SGLang server request path via a ported (from
`research/cacheblend`) precomputed fresh-KV repair adapter.

**Scope statement -- read before assuming feature coverage.** This
package implements only:

* The paper's per-layer roofline cost model
  (``T_layer(r) = max(r*N*t_c, (1-r)*N*t_i) + t_o``) and closed-form
  optimum ``r0 = t_i / (t_c + t_i)`` (`hardware_profile.py`).
* Two explicit ratio-floor modes: ``paper_mechanism`` (the paper's
  quality-preserving ``r_min = 15%``) and ``speed_only`` (this project's
  non-paper ``r_min = 0%``, since only TTFT is tracked here -- never
  presented as the paper's original setting).
* Roofline-warm-started golden-section search over a small measured-TTFT
  calibration set (`golden_section.py`, `controller.py`), matching the
  paper's "one deployment profiling pass" calibration cadence.
* Deterministic quantization of the selected ratio to an executable
  integer repair-token count for the exact request context length.
* Repair execution reusing CacheBlend's real, controlled precomputed
  fresh-KV adapter (`precomputed.py`, `recompute.py`, ported from
  `research/cacheblend`), driven by a count-driven (not ratio-driven)
  token-selection funnel (`token_selection.py`) that self-validates the
  selected count matches the controller's decision exactly.

This package does **not** implement the rest of the CacheTune paper:
frequency-domain token selection, sparse transfer, multi-stream overlap,
or deferred RoPE are all out of scope. It also does not achieve genuine
wall-clock overlap between the "recompute" and "transfer" critical paths
during actual repair execution (SGLang's ``ModelRunner`` has no hook for
a real inline per-layer forward on an arbitrary token subset while a
concurrent transfer is in flight) -- the roofline model is used
faithfully to *choose* the ratio, but *executing* that ratio falls back
to the same real, separate dense-preparation-request adapter CacheBlend
uses. Any report or telemetry describing this package must call it a
"CacheTune hardware-controller inspired subset", never a complete,
faithful implementation of the paper.
"""

from .controller import (
    CacheTuneController,
    CacheTuneDecision,
    CacheTuneProfileError,
    CalibrationResult,
)
from .golden_section import golden_section_search_minimize, warm_start_bracket
from .hardware_profile import (
    CacheTuneMode,
    DenseTimingSample,
    HardwareMeasurement,
    HardwareProfileKey,
    QuantizedRatio,
    RatioBounds,
    TransferTimingSample,
    chunk_length_bucket,
    estimate_measurement_from_samples,
    predict_layer_time_ms,
    predict_ttft_ms,
    quantize_ratio,
    roofline_ratio,
    round_half_up,
)
from .plugin import (
    CACHETUNE_PLUGIN_NAME,
    CacheTuneConfig,
    CacheTuneRecoveryPlugin,
    maybe_register_cachetune_plugin,
)
from .precomputed import FreshKVSpan, PrecomputedCacheTuneBackend
from .recompute import (
    CacheTuneCapabilityError,
    CacheTuneLayerRecomputeBackend,
    CacheTuneProbeBackend,
    LayerRecomputeCoordinator,
    LayerRecomputeResult,
)
from .runtime import restore_request_prefix_cachetune
from .token_selection import (
    GradualFilterStage,
    TokenSelection,
    compute_token_deviation,
    select_repair_tokens,
)

__all__ = [
    "CACHETUNE_PLUGIN_NAME",
    "CacheTuneCapabilityError",
    "CacheTuneConfig",
    "CacheTuneController",
    "CacheTuneDecision",
    "CacheTuneLayerRecomputeBackend",
    "CacheTuneMode",
    "CacheTuneProbeBackend",
    "CacheTuneProfileError",
    "CacheTuneRecoveryPlugin",
    "CalibrationResult",
    "DenseTimingSample",
    "FreshKVSpan",
    "GradualFilterStage",
    "HardwareMeasurement",
    "HardwareProfileKey",
    "LayerRecomputeCoordinator",
    "LayerRecomputeResult",
    "PrecomputedCacheTuneBackend",
    "QuantizedRatio",
    "RatioBounds",
    "TokenSelection",
    "TransferTimingSample",
    "chunk_length_bucket",
    "compute_token_deviation",
    "estimate_measurement_from_samples",
    "golden_section_search_minimize",
    "maybe_register_cachetune_plugin",
    "predict_layer_time_ms",
    "predict_ttft_ms",
    "quantize_ratio",
    "restore_request_prefix_cachetune",
    "roofline_ratio",
    "round_half_up",
    "select_repair_tokens",
    "warm_start_bracket",
]
