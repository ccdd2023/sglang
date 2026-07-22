from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def _read_bool(
    env: Mapping[str, str],
    name: str,
    default: bool = False,
) -> bool:
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
class ApproxKVFeatureConfig:
    core_enabled: bool = False
    lossy_recovery_enabled: bool = False
    prefetch_enabled: bool = False

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "ApproxKVFeatureConfig":
        env = os.environ if env is None else env
        core = _read_bool(env, "SGLANG_APPROX_KV_CORE", False)
        lossy = _read_bool(env, "SGLANG_APPROX_KV_LOSSY", False)
        prefetch = _read_bool(env, "SGLANG_APPROX_KV_PREFETCH", False)
        if (lossy or prefetch) and not core:
            raise ValueError(
                "SGLANG_APPROX_KV_CORE=1 is required when lossy recovery "
                "or prefetch is enabled"
            )
        return cls(
            core_enabled=core,
            lossy_recovery_enabled=lossy,
            prefetch_enabled=prefetch,
        )
