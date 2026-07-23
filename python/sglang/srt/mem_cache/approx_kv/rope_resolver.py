"""Shared "default RoPE" resolver for approximate-KV recovery plugins.

Ported verbatim (same logic, same supported-model set) from the R1
EPIC/LegoLink fork (``approx_kv/epic_runtime.py``) and the R0 raw+RoPE
fork (``approx_kv/raw_rope.py``), where this exact function already lives
independently under two different plugin-specific modules. Living here in
common-core instead means every paper-specific plugin (CacheBlend, raw
RoPE, EPIC) that needs a real, production ``RoPEConfig`` to correct for a
non-zero rope_delta shares one resolver instead of three drifting copies.

This only resolves the "default" (unscaled) RoPE layout used by Qwen2 and
Qwen3: any ``rope_scaling`` config with a type other than the (missing or
``"default"``) baseline -- e.g. ``"linear"``, ``"yarn"``, ``"dynamic"`` --
returns ``None`` rather than guessing at a scaled rotation. Callers must
treat ``None`` exactly like "no RoPE config resolved": the recovery path
falls back to dense whenever it actually needs a non-zero rope_delta
correction (see ``runtime.py``'s ``rope_config_unavailable`` fallback), it
never silently applies an unscaled rotation to a scaled model.
"""

from __future__ import annotations

from typing import Any

from .radix_backend import RoPEConfig


def resolve_model_rope_config(model_config: Any) -> RoPEConfig | None:
    """Resolve the supported Qwen2/Qwen3 default-RoPE layout from the live
    model config, or ``None`` if this model/config isn't a case this
    resolver supports (non-Qwen model family, scaled RoPE, or an odd/zero
    rotary dimension)."""
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
    partial_factor = float(getattr(hf_config, "partial_rotary_factor", 1.0))
    rotary_dim = int(int(head_dim) * partial_factor)
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
