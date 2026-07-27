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


# EPIC fixed leading-k attention-sink repair only supports these window
# sizes (see epic_plugin.EPICLeadingKPlugin / Phase 4 R1 requirements).
# k=0 degenerates to the plain raw-copy (R0) path.
SUPPORTED_EPIC_K_VALUES: tuple[int, ...] = (0, 2, 4, 8, 16, 32)


def _read_epic_k(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed not in SUPPORTED_EPIC_K_VALUES:
        raise ValueError(
            f"{name}={parsed} is not a supported EPIC leading-k value; "
            f"must be one of {SUPPORTED_EPIC_K_VALUES}"
        )
    return parsed


@dataclass(frozen=True)
class ApproxKVFeatureConfig:
    core_enabled: bool = False
    host_residency_enabled: bool = False
    async_prefetch_enabled: bool = False
    epic_enabled: bool = False
    epic_k: int = 0
    epic_attention_sink: bool = True
    cross_store_enabled: bool = False
    cross_store_bytes_per_token: int = 1
    cross_store_host_budget_bytes: int = 0
    cross_store_registration_evicts_exact: bool = False
    test_mode_enabled: bool = False
    cross_store_test_reservation_failure: bool = False

    def __post_init__(self) -> None:
        if self.epic_k not in SUPPORTED_EPIC_K_VALUES:
            raise ValueError(
                f"epic_k={self.epic_k} is not a supported EPIC leading-k "
                f"value; must be one of {SUPPORTED_EPIC_K_VALUES}"
            )
        if self.epic_enabled and not self.core_enabled:
            raise ValueError("core_enabled=True is required when epic_enabled is True")
        if self.cross_store_enabled and not self.core_enabled:
            raise ValueError(
                "core_enabled=True is required when cross_store_enabled is True"
            )
        if self.cross_store_bytes_per_token <= 0:
            raise ValueError("cross_store_bytes_per_token must be positive")
        if self.cross_store_host_budget_bytes < 0:
            raise ValueError("cross_store_host_budget_bytes must be non-negative")
        if self.cross_store_registration_evicts_exact and not self.cross_store_enabled:
            raise ValueError(
                "cross_store_enabled=True is required when registration "
                "may evict exact cache"
            )
        if self.cross_store_test_reservation_failure:
            if not self.test_mode_enabled:
                raise ValueError(
                    "test_mode_enabled=True is required for cross-store "
                    "reservation failure injection"
                )
            if not self.cross_store_enabled:
                raise ValueError(
                    "cross_store_enabled=True is required for cross-store "
                    "reservation failure injection"
                )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> ApproxKVFeatureConfig:
        env = os.environ if env is None else env
        core = _read_bool(env, "SGLANG_APPROX_KV_CORE", False)
        host = _read_bool(env, "SGLANG_APPROX_KV_HOST", False)
        prefetch = _read_bool(env, "SGLANG_APPROX_KV_PREFETCH", False)
        epic_enabled = _read_bool(env, "SGLANG_APPROX_KV_EPIC", False)
        epic_k = _read_epic_k(env, "SGLANG_APPROX_KV_EPIC_K", 0)
        epic_attention_sink = _read_bool(
            env, "SGLANG_APPROX_KV_EPIC_ATTENTION_SINK", True
        )
        cross_store_enabled = _read_bool(env, "SGLANG_APPROX_KV_CROSS_STORE", False)
        cross_store_bytes_per_token = int(
            env.get("SGLANG_APPROX_KV_BYTES_PER_TOKEN", "1")
        )
        cross_store_host_budget_bytes = int(
            env.get("SGLANG_APPROX_KV_HOST_BUDGET_BYTES", "0")
        )
        cross_store_registration_evicts_exact = _read_bool(
            env,
            "SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT",
            False,
        )
        test_mode_enabled = _read_bool(
            env,
            "SGLANG_APPROX_KV_TEST_ONLY",
            False,
        )
        cross_store_test_reservation_failure = _read_bool(
            env,
            "SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE",
            False,
        )
        if (host or prefetch) and not core:
            raise ValueError(
                "SGLANG_APPROX_KV_CORE=1 is required when host residency "
                "or prefetch is enabled"
            )
        if prefetch and not host:
            raise ValueError(
                "SGLANG_APPROX_KV_HOST=1 is required when prefetch is enabled"
            )
        if epic_enabled and not core:
            raise ValueError("SGLANG_APPROX_KV_CORE=1 is required when EPIC is enabled")
        if cross_store_enabled and not core:
            raise ValueError(
                "SGLANG_APPROX_KV_CORE=1 is required when cross-store is enabled"
            )
        return cls(
            core_enabled=core,
            host_residency_enabled=host,
            async_prefetch_enabled=prefetch,
            epic_enabled=epic_enabled,
            epic_k=epic_k,
            epic_attention_sink=epic_attention_sink,
            cross_store_enabled=cross_store_enabled,
            cross_store_bytes_per_token=cross_store_bytes_per_token,
            cross_store_host_budget_bytes=cross_store_host_budget_bytes,
            cross_store_registration_evicts_exact=(
                cross_store_registration_evicts_exact
            ),
            test_mode_enabled=test_mode_enabled,
            cross_store_test_reservation_failure=(cross_store_test_reservation_failure),
        )
