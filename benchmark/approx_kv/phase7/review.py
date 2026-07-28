from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

FINAL_REVIEW_ARTIFACT = "phase7-final-opus-review"
FINAL_REVIEW_SCHEMA_VERSION = 1
ACCEPTED_VERDICTS = ("PASS", "PASS_WITH_CAVEATS")
REQUIRED_REVIEW_RUNNERS = ("ceiling", "scheduler", "capacity_pilot")
FINDING_SEVERITIES = ("P0", "P1", "P2", "P3")
FINDING_DISPOSITIONS = ("closed", "accepted_no_change", "deferred")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def review_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_final_review(
    *,
    reviewer: str,
    model: str,
    verdict: str,
    open_p0: int,
    open_p1: int,
    reviewed_manifest_revision: int,
    reviewed_manifest_sha256: str,
    design_payload_sha256: str,
    reviewed_pinned_implementation_sha: str,
    reviewed_pinned_tree_sha: str,
    runner_sha256: Mapping[str, str],
    findings: list[Mapping[str, Any]],
    disposition: str,
    timestamp: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": FINAL_REVIEW_SCHEMA_VERSION,
        "artifact": FINAL_REVIEW_ARTIFACT,
        "reviewer": reviewer,
        "model": model,
        "verdict": verdict,
        "open_p0": open_p0,
        "open_p1": open_p1,
        "reviewed_manifest_revision": reviewed_manifest_revision,
        "reviewed_manifest_sha256": reviewed_manifest_sha256,
        "design_payload_sha256": design_payload_sha256,
        "reviewed_pinned_implementation_sha": reviewed_pinned_implementation_sha,
        "reviewed_pinned_tree_sha": reviewed_pinned_tree_sha,
        "runner_sha256": dict(runner_sha256),
        "findings": [dict(row) for row in findings],
        "disposition": disposition,
        "timestamp": timestamp,
    }
    payload["artifact_sha256"] = review_payload_sha256(payload)
    validate_final_review(payload)
    return payload


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} is not a lowercase SHA-256")
    return value


def _require_git_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} is not a full git SHA")
    return value


def _require_count(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def validate_final_review(payload: Mapping[str, Any]) -> None:
    """Validate the frozen content contract of the final Opus review artifact."""
    required = (
        "schema_version",
        "artifact",
        "reviewer",
        "model",
        "verdict",
        "open_p0",
        "open_p1",
        "reviewed_manifest_revision",
        "reviewed_manifest_sha256",
        "design_payload_sha256",
        "reviewed_pinned_implementation_sha",
        "reviewed_pinned_tree_sha",
        "runner_sha256",
        "findings",
        "disposition",
        "timestamp",
        "artifact_sha256",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"missing final review fields: {missing}")
    if payload["schema_version"] != FINAL_REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported final review schema")
    if payload["artifact"] != FINAL_REVIEW_ARTIFACT:
        raise ValueError("unexpected final review artifact")
    for field in ("reviewer", "model", "disposition"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if payload["verdict"] not in ACCEPTED_VERDICTS:
        raise ValueError(f"final review verdict must be one of {ACCEPTED_VERDICTS}")
    if _require_count(payload["open_p0"], field="open_p0") != 0:
        raise ValueError("final review must have zero open P0 findings")
    if _require_count(payload["open_p1"], field="open_p1") != 0:
        raise ValueError("final review must have zero open P1 findings")
    revision = payload["reviewed_manifest_revision"]
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 6:
        raise ValueError("reviewed_manifest_revision must be an int >= 6")
    _require_sha256(
        payload["reviewed_manifest_sha256"], field="reviewed_manifest_sha256"
    )
    _require_sha256(payload["design_payload_sha256"], field="design_payload_sha256")
    _require_git_sha(
        payload["reviewed_pinned_implementation_sha"],
        field="reviewed_pinned_implementation_sha",
    )
    _require_git_sha(
        payload["reviewed_pinned_tree_sha"], field="reviewed_pinned_tree_sha"
    )
    runner_sha256 = payload["runner_sha256"]
    if not isinstance(runner_sha256, Mapping):
        raise ValueError("runner_sha256 must be an object")
    missing_runners = [
        name for name in REQUIRED_REVIEW_RUNNERS if name not in runner_sha256
    ]
    if missing_runners:
        raise ValueError(f"final review lacks runner hashes: {missing_runners}")
    for name, value in runner_sha256.items():
        _require_sha256(value, field=f"runner_sha256.{name}")
    findings = payload["findings"]
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")
    finding_ids = []
    for index, row in enumerate(findings):
        if not isinstance(row, Mapping):
            raise ValueError(f"findings[{index}] must be an object")
        for field in ("finding_id", "severity", "summary", "disposition"):
            if field not in row:
                raise ValueError(f"findings[{index}] lacks {field}")
        if row["severity"] not in FINDING_SEVERITIES:
            raise ValueError(f"findings[{index}] has an unsupported severity")
        if row["disposition"] not in FINDING_DISPOSITIONS:
            raise ValueError(f"findings[{index}] has an unsupported disposition")
        if row["severity"] in {"P0", "P1"} and row["disposition"] != "closed":
            raise ValueError(f"findings[{index}] leaves a P0/P1 finding open")
        for field in ("finding_id", "summary"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"findings[{index}].{field} must be a non-empty str")
        finding_ids.append(row["finding_id"])
    if len(set(finding_ids)) != len(finding_ids):
        raise ValueError("findings must have unique finding_id values")
    open_severe = [
        row["finding_id"]
        for row in findings
        if row["severity"] in {"P0", "P1"} and row["disposition"] != "closed"
    ]
    if open_severe:
        raise ValueError(f"open P0/P1 findings are not closed: {open_severe}")
    try:
        parsed_timestamp = datetime.fromisoformat(
            str(payload["timestamp"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("timestamp must be ISO 8601") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    if payload["artifact_sha256"] != review_payload_sha256(payload):
        raise ValueError("final review self-hash mismatch")


def load_final_review(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_final_review(payload)
    return payload


def validate_review_binding(
    review: Mapping[str, Any],
    *,
    design_payload_sha256: str,
    supersedes_manifest_sha256: Any,
    manifest_revision: int,
    pinned_implementation_sha: Any,
    pinned_tree_sha: Any,
    runner_sha256: Mapping[str, Any],
) -> None:
    """Bind a validated review artifact to the manifest that activates it.

    The review is always a review *of the superseded revision*: binding it to
    the new manifest's own self-hash would be a self-containing cycle, so the
    contract is that the design payload is unchanged, the reviewed self-hash is
    exactly what the new manifest supersedes, and the revision strictly grows.
    The reviewed implementation pin is bound as well, so a revision that
    re-pins different implementation code cannot inherit the review.
    """
    validate_final_review(review)
    if review["design_payload_sha256"] != design_payload_sha256:
        raise ValueError("final review design payload hash mismatch")
    if not isinstance(supersedes_manifest_sha256, str) or (
        _SHA256_RE.fullmatch(supersedes_manifest_sha256) is None
    ):
        raise ValueError("activating manifest does not record a superseded hash")
    if review["reviewed_manifest_sha256"] != supersedes_manifest_sha256:
        raise ValueError("final review did not review the superseded manifest revision")
    if not isinstance(manifest_revision, int) or isinstance(manifest_revision, bool):
        raise ValueError("manifest_revision must be an integer")
    if manifest_revision <= int(review["reviewed_manifest_revision"]):
        raise ValueError(
            "activating manifest revision must be greater than the reviewed revision"
        )
    if review["reviewed_pinned_implementation_sha"] != pinned_implementation_sha:
        raise ValueError("final review pinned implementation SHA mismatch")
    if review["reviewed_pinned_tree_sha"] != pinned_tree_sha:
        raise ValueError("final review pinned implementation tree mismatch")
    drifted = sorted(
        name
        for name, value in review["runner_sha256"].items()
        if runner_sha256.get(name) != value
    )
    if drifted:
        raise ValueError(f"final review runner hash mismatch: {drifted}")
    missing = sorted(
        name for name in REQUIRED_REVIEW_RUNNERS if name not in runner_sha256
    )
    if missing:
        raise ValueError(f"manifest lacks reviewed runners: {missing}")


def write_final_review(path: Path, payload: Mapping[str, Any]) -> None:
    validate_final_review(payload)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
