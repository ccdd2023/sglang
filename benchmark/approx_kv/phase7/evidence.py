from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

RUNNER_TEST_EVIDENCE_ARTIFACT = "phase7-runner-cpu-test-evidence"


def evidence_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_runner_test_evidence(
    *,
    runner_key: str,
    runner_module: str,
    runner_path: str,
    runner_sha256: str,
    image_digest: str,
    command: str,
    exit_code: int,
    summary_line: str,
    passed_count: int,
    subtests_passed_count: int,
    subtests: list[str],
    timestamp: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "artifact": RUNNER_TEST_EVIDENCE_ARTIFACT,
        "runner_key": runner_key,
        "runner_module": runner_module,
        "runner_path": runner_path,
        "runner_sha256": runner_sha256,
        "image_digest": image_digest,
        "command": command,
        "exit_code": exit_code,
        "summary_line": summary_line,
        "passed_count": passed_count,
        "subtests": {
            "passed_count": subtests_passed_count,
            "names": list(subtests),
        },
        "timestamp": timestamp,
    }
    payload["artifact_sha256"] = evidence_payload_sha256(payload)
    validate_runner_test_evidence(payload)
    return payload


def validate_runner_test_evidence(payload: Mapping[str, Any]) -> None:
    required = (
        "schema_version",
        "artifact",
        "runner_key",
        "runner_module",
        "runner_path",
        "runner_sha256",
        "image_digest",
        "command",
        "exit_code",
        "summary_line",
        "passed_count",
        "subtests",
        "timestamp",
        "artifact_sha256",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"missing runner test evidence fields: {missing}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported runner test evidence schema")
    if payload["artifact"] != RUNNER_TEST_EVIDENCE_ARTIFACT:
        raise ValueError("unexpected runner test evidence artifact")
    for field in ("runner_key", "runner_module", "runner_path", "command"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    summary_line = payload["summary_line"]
    if not isinstance(summary_line, str) or not summary_line.strip():
        raise ValueError("summary_line must be a non-empty string")
    exit_code = payload["exit_code"]
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError("exit_code must be an integer")
    if exit_code != 0:
        raise ValueError("exit_code must be 0 for passing CPU test evidence")
    runner_sha = payload["runner_sha256"]
    if (
        not isinstance(runner_sha, str)
        or len(runner_sha) != 64
        or any(character not in "0123456789abcdef" for character in runner_sha)
    ):
        raise ValueError("runner_sha256 is not a lowercase SHA-256")
    image_digest = payload["image_digest"]
    if (
        not isinstance(image_digest, str)
        or not image_digest.startswith("sha256:")
        or len(image_digest) != 71
        or any(character not in "0123456789abcdef" for character in image_digest[7:])
    ):
        raise ValueError("image_digest is not a pinned sha256 digest")
    passed_count = payload["passed_count"]
    if not isinstance(passed_count, int) or isinstance(passed_count, bool):
        raise ValueError("passed_count must be an integer")
    if passed_count <= 0:
        raise ValueError("passed_count must be positive")
    subtests = payload["subtests"]
    if not isinstance(subtests, Mapping):
        raise ValueError("subtests must be an object")
    subtest_count = subtests.get("passed_count")
    if not isinstance(subtest_count, int) or isinstance(subtest_count, bool):
        raise ValueError("subtests.passed_count must be an integer")
    if subtest_count < 0:
        raise ValueError("subtests.passed_count must be non-negative")
    names = subtests.get("names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("subtests.names must contain unique non-empty strings")
    try:
        parsed_timestamp = datetime.fromisoformat(
            str(payload["timestamp"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("timestamp must be ISO 8601") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    if payload["artifact_sha256"] != evidence_payload_sha256(payload):
        raise ValueError("runner test evidence self-hash mismatch")


def write_runner_test_evidence(path: Path, payload: Mapping[str, Any]) -> None:
    validate_runner_test_evidence(payload)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
