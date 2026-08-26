"""Serving-class template: offline class prior, online fine-tune.

Users do not share files or GitHub issues. The reusable object is a
*task class* (e.g. rolling-6 coding agents), not a content hash and not
a per-issue PLAN. Offline profiling of that class sets Beta pseudo-counts.
Online bind outcomes nudge the same bins with unit weight so one session
fine-tunes the class without overwriting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from sglang.srt.mem_cache.coding_aware.online_admit import (
    BindAction,
    BindResult,
    SourceObservation,
    mechanical_source_gates,
    protocol_later_roles,
)


def task_class_id(policy_label: str) -> str:
    if "coding" in (policy_label or ""):
        return "coding_agent"
    return "general"


def featurize(obs: SourceObservation) -> str:
    """One bin per serving class. Not file bytes, not issue id, not (s, t)."""
    return task_class_id(obs.policy_label)


@dataclass
class ClassBin:
    feature_key: str
    alpha: float = 1.0
    beta: float = 1.0
    leased: int = 0
    copied: int = 0
    wasted: int = 0
    offline_n: int = 0

    @property
    def mean(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    @property
    def observations(self) -> int:
        return self.copied + self.wasted


@dataclass
class OnlineFileTemplate:
    """Class-level serving template. Name kept for the existing flag path."""

    admit_floor: float = 0.60
    skip_ceiling: float = 0.20
    min_obs: int = 8
    online_step: float = 1.0
    _bins: dict[str, ClassBin] = field(default_factory=dict)

    def bin_for(self, obs: SourceObservation) -> ClassBin:
        key = featurize(obs)
        found = self._bins.get(key)
        if found is None:
            found = ClassBin(feature_key=key)
            self._bins[key] = found
        return found

    def admit(self, obs: SourceObservation) -> str | None:
        reason = mechanical_source_gates(obs)
        if reason is not None:
            return reason
        post = self.bin_for(obs)
        if obs.later_roles_in_protocol > 0:
            if post.observations >= self.min_obs and post.mean <= self.skip_ceiling:
                return "learned_low_reuse"
            return None
        if post.copied >= self.min_obs and post.mean >= self.admit_floor:
            return None
        return "no_protocol_reread"

    def observe(self, result: BindResult, obs: SourceObservation) -> None:
        post = self.bin_for(obs)
        post.leased += 1
        step = self.online_step
        if result.action is BindAction.COPY:
            post.copied += 1
            post.alpha += step
            return
        if result.reason in {"not_in_target", "token_ids_mismatch"}:
            post.wasted += 1
            post.beta += step
            return
        if result.reason == "zero_shift":
            post.beta += 0.25 * step

    def prefetch_priority(self, obs: SourceObservation) -> int:
        later = protocol_later_roles(
            obs.policy_label, explicit=obs.later_roles_in_protocol
        )
        bonus = int(round(4.0 * self.bin_for(obs).mean))
        return max(0, later + bonus)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "alpha": bin_.alpha,
                "beta": bin_.beta,
                "mean": bin_.mean,
                "leased": bin_.leased,
                "copied": bin_.copied,
                "wasted": bin_.wasted,
                "offline_n": bin_.offline_n,
            }
            for key, bin_ in self._bins.items()
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "serving_class_template",
            "not_per_issue_plan": True,
            "not_content_hash": True,
            "admit_floor": self.admit_floor,
            "skip_ceiling": self.skip_ceiling,
            "min_obs": self.min_obs,
            "online_step": self.online_step,
            "bins": {
                key: {
                    "alpha": bin_.alpha,
                    "beta": bin_.beta,
                    "offline_n": bin_.offline_n,
                }
                for key, bin_ in self._bins.items()
            },
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "OnlineFileTemplate":
        template = cls(
            admit_floor=float(payload.get("admit_floor", 0.60)),
            skip_ceiling=float(payload.get("skip_ceiling", 0.20)),
            min_obs=int(payload.get("min_obs", 8)),
            online_step=float(payload.get("online_step", 1.0)),
        )
        for key, row in dict(payload.get("bins") or {}).items():
            template._bins[str(key)] = ClassBin(
                feature_key=str(key),
                alpha=float(row.get("alpha", 1.0)),
                beta=float(row.get("beta", 1.0)),
                offline_n=int(row.get("offline_n", 0)),
            )
        return template

    @classmethod
    def from_path(cls, path: str | Path) -> "OnlineFileTemplate":
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(payload)
