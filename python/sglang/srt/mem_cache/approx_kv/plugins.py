from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .store import ApproxKVSegmentStore
from .types import KVReusePlan, SchedulerMetadata


@dataclass(frozen=True)
class RecoveryRequestContext:
    request_id: str
    target_token_ids: tuple[int, ...]
    exact_prefix_length: int
    custom_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.exact_prefix_length < 0:
            raise ValueError("exact_prefix_length must be non-negative")
        if self.exact_prefix_length > len(self.target_token_ids):
            raise ValueError("exact_prefix_length exceeds target tokens")


class RecoveryPlugin(Protocol):
    @property
    def name(self) -> str: ...

    def build_plan(
        self,
        context: RecoveryRequestContext,
        store: ApproxKVSegmentStore,
    ) -> KVReusePlan: ...

    def scheduler_metadata(
        self,
        context: RecoveryRequestContext,
    ) -> tuple[SchedulerMetadata, ...]: ...


class RecoveryPluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, RecoveryPlugin] = {}

    def register(self, plugin: RecoveryPlugin) -> None:
        name = plugin.name.strip()
        if not name:
            raise ValueError("recovery plugin name must be non-empty")
        if name in self._plugins:
            raise ValueError(f"recovery plugin is already registered: {name}")
        self._plugins[name] = plugin

    def get(self, name: str) -> RecoveryPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise KeyError(f"unknown recovery plugin: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))
