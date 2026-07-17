from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class KVCommFeatureConfig:
    """Independent feature gates for the shared data plane and its clients."""

    core_enabled: bool = False
    coding_aware_lossy_enabled: bool = False
    prefetch_enabled: bool = False
    legacy_flags_used: bool = False

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None
    ) -> "KVCommFeatureConfig":
        env = os.environ if env is None else env
        legacy_compat = _read_bool(env, "SGLANG_KVFLOW_LEGACY_FLAGS", False)
        has_new_flag = any(
            key in env
            for key in (
                "SGLANG_KVCOMM_CORE",
                "SGLANG_CODING_AWARE_LOSSY",
                "SGLANG_KV_PREFETCH",
            )
        )

        if legacy_compat and not has_new_flag and "SGLANG_LOSSY_ENABLED" in env:
            enabled = _read_bool(env, "SGLANG_LOSSY_ENABLED", False)
            logger.warning(
                "SGLANG_LOSSY_ENABLED is deprecated; set the independent "
                "SGLANG_KVCOMM_CORE, SGLANG_CODING_AWARE_LOSSY, and "
                "SGLANG_KV_PREFETCH gates instead"
            )
            return cls(
                core_enabled=enabled,
                coding_aware_lossy_enabled=enabled,
                prefetch_enabled=enabled,
                legacy_flags_used=True,
            )

        core = _read_bool(env, "SGLANG_KVCOMM_CORE", False)
        coding = _read_bool(env, "SGLANG_CODING_AWARE_LOSSY", False)
        prefetch = _read_bool(env, "SGLANG_KV_PREFETCH", False)
        if (coding or prefetch) and not core:
            raise ValueError(
                "SGLANG_KVCOMM_CORE=1 is required when coding-aware lossy "
                "reuse or KV prefetch is enabled"
            )
        return cls(
            core_enabled=core,
            coding_aware_lossy_enabled=coding,
            prefetch_enabled=prefetch,
        )
