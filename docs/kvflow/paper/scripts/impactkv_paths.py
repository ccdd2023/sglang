"""Resolve ImpactKV frozen-artifact and engine roots off-cluster.

Collaborators do not have the original cluster filesystem. Discovery order:

1. ``IMPACTKV_ARTIFACTS`` / ``IMPACTKV_ENGINE_ROOT``
2. ``<this sglang clone>/impactkv-artifacts`` (created by fetch_impactkv_artifacts.py)
3. ``$HOME/impactkv-artifacts``
4. Cluster path, only if it still exists
"""
from __future__ import annotations

import os
from pathlib import Path

_CLUSTER_ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
_CLUSTER_ENGINE = Path(
    "/home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/template-prefetch"
)
_ENGINE_MARKER = Path("python/sglang/srt/mem_cache/kvcomm_exact.py")


def engine_root() -> Path:
    env = os.environ.get("IMPACTKV_ENGINE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / _ENGINE_MARKER).exists():
            return parent
    if _CLUSTER_ENGINE.exists():
        return _CLUSTER_ENGINE
    return Path(__file__).resolve().parents[4]


def artifact_root() -> Path:
    env = os.environ.get("IMPACTKV_ARTIFACTS")
    if env:
        return Path(env).expanduser().resolve()
    root = engine_root()
    local = root / "impactkv-artifacts"
    if local.exists():
        return local
    home = Path.home() / "impactkv-artifacts"
    if home.exists():
        return home
    if _CLUSTER_ARTIFACTS.exists():
        return _CLUSTER_ARTIFACTS
    return local
