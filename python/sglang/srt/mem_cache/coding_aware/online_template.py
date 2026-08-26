"""Session-level online template over file-module content hashes.

The protocol later-roles number is the cold-start prior, not a frozen PLAN.
Each bind outcome updates a Beta-Bernoulli posterior for that module
(content_hash). Subsequent admits use the posterior:

* protocol later-roles > 0 still leases, unless enough wasted binds
  have driven the mean below ``skip_ceiling``;
* protocol later-roles = 0 still skips, unless enough successful copies
  have driven the mean above ``admit_floor``.

The learner never sees a target index at admit time. It is not an
Attention estimator and it does not grow the mechanical admit set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sglang.srt.mem_cache.coding_aware.online_admit import (
    BindAction,
    BindResult,
    SourceObservation,
    mechanical_source_gates,
)


@dataclass
class ModulePosterior:
    content_hash: str
    alpha: float = 1.0
    beta: float = 1.0
    leased: int = 0
    copied: int = 0
    wasted: int = 0

    @property
    def mean(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    @property
    def observations(self) -> int:
        return self.copied + self.wasted


@dataclass
class OnlineFileTemplate:
    """Learned file-module template. Keyed by content hash, not (s, t)."""

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    admit_floor: float = 0.60
    skip_ceiling: float = 0.20
    min_obs: int = 2
    _posteriors: dict[str, ModulePosterior] = field(default_factory=dict)

    def posterior(self, content_hash: str) -> ModulePosterior:
        found = self._posteriors.get(content_hash)
        if found is None:
            found = ModulePosterior(
                content_hash=content_hash,
                alpha=self.prior_alpha,
                beta=self.prior_beta,
            )
            self._posteriors[content_hash] = found
        return found

    def admit(self, obs: SourceObservation) -> str | None:
        reason = mechanical_source_gates(obs)
        if reason is not None:
            return reason
        post = self.posterior(obs.content_hash)
        if obs.later_roles_in_protocol > 0:
            if post.observations >= self.min_obs and post.mean <= self.skip_ceiling:
                return "learned_low_reuse"
            return None
        if post.observations >= self.min_obs and post.mean >= self.admit_floor:
            return None
        return "no_protocol_reread"

    def observe(self, result: BindResult) -> None:
        if not result.content_hash:
            return
        post = self.posterior(result.content_hash)
        post.leased += 1
        if result.action is BindAction.COPY:
            post.copied += 1
            post.alpha += 1.0
            return
        if result.reason in {"not_in_target", "token_ids_mismatch"}:
            post.wasted += 1
            post.beta += 1.0
            return
        if result.reason == "zero_shift":
            post.beta += 0.25

    def observe_all(self, results: tuple[BindResult, ...] | list[BindResult]) -> None:
        for result in results:
            self.observe(result)

    def prefetch_priority(self, content_hash: str, later_roles: int) -> int:
        """Residency rank: protocol later-roles plus learned reuse."""
        bonus = int(round(4.0 * self.posterior(content_hash).mean))
        return max(0, int(later_roles) + bonus)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "alpha": post.alpha,
                "beta": post.beta,
                "mean": post.mean,
                "leased": post.leased,
                "copied": post.copied,
                "wasted": post.wasted,
            }
            for key, post in self._posteriors.items()
        }
