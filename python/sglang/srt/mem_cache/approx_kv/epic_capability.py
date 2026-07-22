"""Capability introspection for EPIC-style layer-wise leading-k repair.

The EPIC recovery path (see ``epic_recompute.py``) drives a model's real
transformer decoder layers one at a time so the leading-k target-context
tokens receive a genuine forward pass at every layer before the remaining
body KV is reused. That is only safe against a model runner whose layers
expose the same ``forward(positions, hidden_states, forward_batch, ...)``
contract used across upstream SGLang decoder layers (see
``sglang.srt.models.qwen3.Qwen3DecoderLayer.forward`` and siblings).

This module never guesses: any model/layout that does not match the
expected contract is reported as unsupported with an explicit reason so
callers can dense-fallback instead of risking a silently-wrong repair.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Parameter names every supported decoder layer's ``forward`` must accept.
# This mirrors the contract shared by upstream SGLang decoder layers, e.g.
# Qwen3DecoderLayer.forward(self, positions, hidden_states, forward_batch,
# residual, ...).
REQUIRED_LAYER_FORWARD_PARAMS: tuple[str, ...] = (
    "positions",
    "hidden_states",
    "forward_batch",
)


@dataclass(frozen=True)
class LayerwiseCapability:
    supported: bool
    reason: str
    num_layers: int = 0

    def __bool__(self) -> bool:
        return self.supported


def decoder_layers(model_runner: Any) -> Sequence[Any] | None:
    """Return the model's decoder layer list, or ``None`` if unreachable."""
    model = getattr(model_runner, "model", None)
    if model is None:
        return None
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return None
    try:
        materialized = list(layers)
    except TypeError:
        return None
    return materialized


def _layer_forward_matches_contract(layer: Any) -> bool:
    forward = getattr(layer, "forward", None)
    if forward is None or not callable(forward):
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    params = signature.parameters
    return all(name in params for name in REQUIRED_LAYER_FORWARD_PARAMS)


def inspect_layerwise_recompute_capability(
    model_runner: Any,
) -> LayerwiseCapability:
    """Introspect whether ``model_runner`` can be driven layer-by-layer.

    Returns ``LayerwiseCapability(supported=False, reason=...)`` for any
    model/layout that does not match the expected per-layer forward
    contract, rather than raising or silently degrading.
    """
    layers = decoder_layers(model_runner)
    if not layers:
        return LayerwiseCapability(
            False,
            "model_runner exposes no .model.model.layers sequence",
        )
    for index, layer in enumerate(layers):
        if not _layer_forward_matches_contract(layer):
            return LayerwiseCapability(
                False,
                f"layer {index} forward() does not accept "
                f"{REQUIRED_LAYER_FORWARD_PARAMS}",
            )
    return LayerwiseCapability(
        True,
        "all layers expose the expected forward(positions, hidden_states, "
        "forward_batch, ...) contract",
        len(layers),
    )


def inspect_source_layer_forward_params(
    source_path: str | Path,
    class_name: str,
    method_name: str = "forward",
) -> tuple[str, ...]:
    """Extract ``class_name.method_name``'s parameter names via ``ast``.

    This lets tests validate the capability contract against real upstream
    model source files (e.g. ``sglang/srt/models/qwen3.py``) without
    importing the full ``sglang`` package or its heavy CUDA-only
    dependencies -- the same source-inspection strategy already used by
    ``test_approx_kv_integration_source.py`` for common-core wiring checks.
    """
    tree = ast.parse(Path(source_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    args = item.args
                    names = [arg.arg for arg in args.posonlyargs]
                    names += [arg.arg for arg in args.args]
                    names += [arg.arg for arg in args.kwonlyargs]
                    return tuple(names)
            raise ValueError(f"{class_name}.{method_name} not found in {source_path}")
    raise ValueError(f"class {class_name} not found in {source_path}")
