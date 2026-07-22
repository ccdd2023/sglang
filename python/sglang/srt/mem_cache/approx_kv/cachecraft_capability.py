"""Capability introspection for CacheCraft production server wiring.

CacheCraft's algorithmic core -- real Eq.(3)-(14) CCI/beta/gamma/CFO
decision logic (``cachecraft_metrics.py``), genuine (non-placeholder) dense
causal-attention profile capture (``cachecraft_attention.py``), and
partial-repair execution against a real selected-token recompute hook
(``cachecraft_plugin.py``/``cachecraft_recompute.py``/
``cachecraft_runtime.py``) -- is implemented and CPU-tested, but none of it
is reachable from a real running server yet. Concretely:

1. ``schedule_batch.py`` (the frozen common-core request path) calls the
   *generic* ``runtime.restore_request_prefix`` unconditionally for every
   request carrying ``approx_kv_metadata``, regardless of
   ``metadata.plugin``. There is no dispatch branch routing
   ``plugin == "cachecraft"`` requests to
   ``cachecraft_runtime.restore_request_via_cachecraft`` (contrast with R1
   EPIC's ``epic_enabled``-gated dispatch in that same file). Sending a
   real HTTP request with ``plugin: "cachecraft"`` metadata against an
   unmodified server therefore silently falls through to the generic
   raw-copy reuse path -- not CacheCraft's CCI-driven
   direct-reuse/partial-repair/full-recompute decision -- which would look
   like a "successful" CacheCraft run while actually measuring an unrelated
   code path. Any benchmark runner MUST detect and refuse this before
   attributing GPU numbers to CacheCraft.
2. There is no real attention-profile capture wired to a live model's
   actual forward pass: nothing in ``scheduler.py``/``model_runner.py``
   invokes ``cachecraft_attention.py`` during a real prefill to populate a
   ``ChunkContextProfile`` for a live request.
3. There is no production selected-token recompute hook: as documented in
   ``cachecraft_recompute.py``/``cachecraft_runtime.py``,
   ``ForwardMode.TARGET_VERIFY`` is only reachable from inside the
   speculative-decoding worker pipeline (``eagle_worker_v2.py``/
   ``spec_utils.py``), not as a standalone "recompute these positions
   against this KV context" API.

``inspect_scheduler_dispatch_capability`` reports blocker (1) directly and
deterministically, by inspecting the *actually importable*
``schedule_batch`` module's source for a CacheCraft dispatch call, so a
benchmark runner can refuse to produce "server E2E" numbers instead of
silently measuring the wrong code path. Blockers (2) and (3) are
structural (no hook object exists anywhere in the tree to introspect), so
callers must additionally require an explicit, real
``recompute_hook``/profile-capture callable to be passed in -- ``None`` is
never treated as "capability available".
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

# The symbol that must appear in schedule_batch.py's real request-dispatch
# path for a CacheCraft-tagged request to ever reach CacheCraft's decision
# logic instead of the generic raw-copy reuse path.
CACHECRAFT_DISPATCH_SYMBOL = "restore_request_via_cachecraft"


@dataclass(frozen=True)
class CacheCraftServerCapability:
    supported: bool
    reason: str

    def __bool__(self) -> bool:
        return self.supported


def inspect_scheduler_dispatch_capability() -> CacheCraftServerCapability:
    """Check whether the installed scheduler request path can reach
    CacheCraft's decision/execution logic for a real request.

    This inspects ``sglang.srt.managers.schedule_batch`` (the same module
    ``runtime.restore_request_prefix`` is dispatched from -- see
    ``schedule_batch.py``'s ``restore_request_prefix(tree_cache, self)``
    call) for a call to
    ``cachecraft_runtime.restore_request_via_cachecraft``. If that call is
    absent, any request tagged with CacheCraft metadata silently falls
    through to the generic raw-copy reuse path instead, so callers must not
    interpret a "successful" HTTP response as CacheCraft server evidence.

    This is a deterministic, no-network, no-GPU source-code fact about the
    currently importable ``sglang`` package -- it does not require a
    running server. When a benchmark runner's Python environment is the
    same image as the target server (as documented in the runner's
    ``--help``), this is authoritative; otherwise it is only a client-side
    proxy and the runner must say so.
    """
    try:
        from sglang.srt.managers import schedule_batch as schedule_batch_module
    except ImportError as exc:
        return CacheCraftServerCapability(
            supported=False,
            reason=f"schedule_batch module unavailable: {exc!r}",
        )
    return _inspect_module_source(schedule_batch_module)


def _inspect_module_source(module: Any) -> CacheCraftServerCapability:
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError) as exc:
        return CacheCraftServerCapability(
            supported=False,
            reason=f"unable to introspect module source: {exc!r}",
        )
    if CACHECRAFT_DISPATCH_SYMBOL not in source:
        return CacheCraftServerCapability(
            supported=False,
            reason=(
                "schedule_batch.py has no dispatch to "
                f"{CACHECRAFT_DISPATCH_SYMBOL}(); requests tagged with "
                "plugin='cachecraft' would silently fall through to the "
                "generic runtime.restore_request_prefix raw-copy path "
                "instead of CacheCraft's CCI-driven decision, so no real "
                "server benchmark may be attributed to CacheCraft yet"
            ),
        )
    return CacheCraftServerCapability(
        supported=True,
        reason=("schedule_batch.py dispatches to " f"{CACHECRAFT_DISPATCH_SYMBOL}()"),
    )
