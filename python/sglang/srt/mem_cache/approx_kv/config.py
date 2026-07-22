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
    host_residency_enabled: bool = False
    async_prefetch_enabled: bool = False
    raw_rope_plugin_enabled: bool = False

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "ApproxKVFeatureConfig":
        env = os.environ if env is None else env
        core = _read_bool(env, "SGLANG_APPROX_KV_CORE", False)
        host = _read_bool(env, "SGLANG_APPROX_KV_HOST", False)
        prefetch = _read_bool(env, "SGLANG_APPROX_KV_PREFETCH", False)
        raw_rope = _read_bool(env, "SGLANG_APPROX_KV_RAW_ROPE", False)
        if (host or prefetch) and not core:
            raise ValueError(
                "SGLANG_APPROX_KV_CORE=1 is required when host residency "
                "or prefetch is enabled"
            )
        if prefetch and not host:
            raise ValueError(
                "SGLANG_APPROX_KV_HOST=1 is required when prefetch is enabled"
            )
        if raw_rope and not core:
            raise ValueError(
                "SGLANG_APPROX_KV_CORE=1 is required when the raw+RoPE "
                "recovery plugin is enabled"
            )
        return cls(
            core_enabled=core,
            host_residency_enabled=host,
            async_prefetch_enabled=prefetch,
            raw_rope_plugin_enabled=raw_rope,
        )
