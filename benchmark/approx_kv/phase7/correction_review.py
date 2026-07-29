from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

CORRECTION_REVIEW_ARTIFACT = "phase7-capacity-correction-opus-review"
CORRECTION_REVIEW_SCHEMA_VERSION = 1
ACCEPTED_CORRECTION_VERDICTS = ("PASS", "PASS_WITH_CAVEATS")
FINDING_SEVERITIES = ("P0", "P1", "P2", "P3")
FINDING_DISPOSITIONS = ("closed", "accepted_no_change", "deferred")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def correction_review_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def build_correction_review(
    *,
    reviewer: str,
    model: str,
    verdict: str,
    open_p0: int,
    open_p1: int,
    reviewed_correction_manifest_revision: int,
    reviewed_correction_manifest_sha256: str,
    base_manifest_revision: int,
    base_manifest_self_sha256: str,
    base_manifest_design_sha256: str,
    base_manifest_path: str,
    reviewed_correction_pinned_implementation_sha: str,
    reviewed_correction_pinned_tree_sha: str,
    capacity_runner_sha256: str,
    original_raw_sha256: str,
    scope: str,
    allowed_setting: str,
    restart: int,
    findings: list[Mapping[str, Any]],
    disposition: str,
    timestamp: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": CORRECTION_REVIEW_SCHEMA_VERSION,
        "artifact": CORRECTION_REVIEW_ARTIFACT,
        "reviewer": reviewer,
        "model": model,
        "verdict": verdict,
        "open_p0": open_p0,
        "open_p1": open_p1,
        "reviewed_correction_manifest_status": "pinned_blocked",
        "reviewed_correction_manifest_revision": (
            reviewed_correction_manifest_revision
        ),
        "reviewed_correction_manifest_sha256": (reviewed_correction_manifest_sha256),
        "base_manifest_revision": base_manifest_revision,
        "base_manifest_self_sha256": base_manifest_self_sha256,
        "base_manifest_design_sha256": base_manifest_design_sha256,
        "base_manifest_path": base_manifest_path,
        "reviewed_correction_pinned_implementation_sha": (
            reviewed_correction_pinned_implementation_sha
        ),
        "reviewed_correction_pinned_tree_sha": (reviewed_correction_pinned_tree_sha),
        "capacity_runner_sha256": capacity_runner_sha256,
        "original_raw_sha256": original_raw_sha256,
        "scope": scope,
        "allowed_setting": allowed_setting,
        "restart": restart,
        "findings": [dict(row) for row in findings],
        "disposition": disposition,
        "timestamp": timestamp,
    }
    payload["artifact_sha256"] = correction_review_payload_sha256(payload)
    validate_correction_review(payload)
    return payload


def validate_correction_review(payload: Mapping[str, Any]) -> None:
    required = (
        "schema_version",
        "artifact",
        "reviewer",
        "model",
        "verdict",
        "open_p0",
        "open_p1",
        "reviewed_correction_manifest_status",
        "reviewed_correction_manifest_revision",
        "reviewed_correction_manifest_sha256",
        "base_manifest_revision",
        "base_manifest_self_sha256",
        "base_manifest_design_sha256",
        "base_manifest_path",
        "reviewed_correction_pinned_implementation_sha",
        "reviewed_correction_pinned_tree_sha",
        "capacity_runner_sha256",
        "original_raw_sha256",
        "scope",
        "allowed_setting",
        "restart",
        "findings",
        "disposition",
        "timestamp",
        "artifact_sha256",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"missing correction review fields: {missing}")
    if payload["schema_version"] != CORRECTION_REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported correction review schema")
    if payload["artifact"] != CORRECTION_REVIEW_ARTIFACT:
        raise ValueError("unexpected correction review artifact")
    if payload["reviewed_correction_manifest_status"] != "pinned_blocked":
        raise ValueError("correction review must review pinned_blocked status")
    for field in (
        "reviewer",
        "model",
        "base_manifest_path",
        "scope",
        "allowed_setting",
        "disposition",
    ):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if payload["verdict"] not in ACCEPTED_CORRECTION_VERDICTS:
        raise ValueError(
            "correction review verdict must be one of "
            f"{ACCEPTED_CORRECTION_VERDICTS}"
        )
    if _require_count(payload["open_p0"], field="open_p0") != 0:
        raise ValueError("correction review must have zero open P0 findings")
    if _require_count(payload["open_p1"], field="open_p1") != 0:
        raise ValueError("correction review must have zero open P1 findings")
    for field in (
        "reviewed_correction_manifest_revision",
        "base_manifest_revision",
        "restart",
    ):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if payload["reviewed_correction_manifest_revision"] < 1:
        raise ValueError("reviewed correction manifest revision must be positive")
    for field in (
        "reviewed_correction_manifest_sha256",
        "base_manifest_self_sha256",
        "base_manifest_design_sha256",
        "capacity_runner_sha256",
        "original_raw_sha256",
        "artifact_sha256",
    ):
        _require_sha256(payload[field], field=field)
    _require_git_sha(
        payload["reviewed_correction_pinned_implementation_sha"],
        field="reviewed_correction_pinned_implementation_sha",
    )
    _require_git_sha(
        payload["reviewed_correction_pinned_tree_sha"],
        field="reviewed_correction_pinned_tree_sha",
    )
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
                raise ValueError(f"findings[{index}].{field} must be non-empty")
        finding_ids.append(row["finding_id"])
    if len(set(finding_ids)) != len(finding_ids):
        raise ValueError("findings must have unique finding_id values")
    try:
        parsed_timestamp = datetime.fromisoformat(
            str(payload["timestamp"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("timestamp must be ISO 8601") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    if payload["artifact_sha256"] != correction_review_payload_sha256(payload):
        raise ValueError("correction review self-hash mismatch")


def validate_correction_review_binding(
    review: Mapping[str, Any],
    *,
    activating_manifest: Mapping[str, Any],
) -> None:
    validate_correction_review(review)
    expected = {
        "base_manifest_revision": activating_manifest.get("base_manifest_revision"),
        "base_manifest_self_sha256": activating_manifest.get(
            "base_manifest_self_sha256"
        ),
        "base_manifest_design_sha256": activating_manifest.get(
            "base_manifest_design_sha256"
        ),
        "base_manifest_path": activating_manifest.get("base_manifest_path"),
        "reviewed_correction_pinned_implementation_sha": activating_manifest.get(
            "correction_pinned_implementation_sha"
        ),
        "reviewed_correction_pinned_tree_sha": activating_manifest.get(
            "correction_pinned_tree_sha"
        ),
        "capacity_runner_sha256": activating_manifest.get("capacity_runner_sha256"),
        "original_raw_sha256": activating_manifest.get("original_raw_sha256"),
        "scope": activating_manifest.get("scope"),
        "allowed_setting": activating_manifest.get("allowed_setting"),
        "restart": activating_manifest.get("restart"),
    }
    drifted = {
        field: (review.get(field), value)
        for field, value in expected.items()
        if review.get(field) != value
    }
    if drifted:
        raise ValueError(f"correction review binding mismatch: {drifted}")
    if review["reviewed_correction_manifest_sha256"] != activating_manifest.get(
        "supersedes_correction_manifest_sha256"
    ):
        raise ValueError(
            "correction review did not review the superseded correction manifest"
        )
    activating_revision = activating_manifest.get("correction_manifest_revision")
    if (
        not isinstance(activating_revision, int)
        or isinstance(activating_revision, bool)
        or activating_revision <= int(review["reviewed_correction_manifest_revision"])
    ):
        raise ValueError(
            "authorized correction revision must exceed the reviewed revision"
        )


def load_correction_review(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_correction_review(payload)
    return payload


def write_correction_review(path: Path, payload: Mapping[str, Any]) -> None:
    validate_correction_review(payload)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
