from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PHASE6_ARTIFACT_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "phase",
    "source_git_sha",
    "source_tree_sha",
    "result_git_sha",
    "result_commit_status",
    "raw_sha256",
    "server_argv",
    "plugin_env",
    "machine",
    "image_digest",
    "requested_capacity",
    "observed_capacity",
    "crosses_chunk_boundary",
    "segment_count",
    "warmup_repeats",
    "formal_repeats",
    "restarts",
    "ledger",
    "rho",
    "status",
)

PHASE6_ARTIFACT_STATUS_VALUES = (
    "valid",
    "negative",
    "inconclusive",
    "invalid",
)

PHASE6_LEDGER_FIELDS = (
    "setup",
    "materialization",
    "recovery",
    "scheduler",
    "transfer",
    "temporary_peak",
)


@dataclass(frozen=True)
class RhoDefinitions:
    logical_demand: str = (
        "logical reusable working set tokens / configured device token capacity"
    )
    physical_demand: str = (
        "requested physical bytes including all representations and scratch / "
        "configured device byte capacity"
    )
    resident: str = (
        "sampled device used plus evictable bytes / configured device byte capacity"
    )
    host: str = "host working set bytes / configured host byte capacity"


@dataclass(frozen=True)
class Phase6RunSettings:
    source_git_sha: str
    source_tree_sha: str
    image_digest: str
    model: str
    model_revision: str
    chunked_prefill_size: int
    chunk_source: str
    warmup_repeats: int
    formal_repeats: int
    restarts: int
    rho: RhoDefinitions = field(default_factory=RhoDefinitions)

    def __post_init__(self) -> None:
        if not self.source_git_sha or not self.source_tree_sha or not self.image_digest:
            raise ValueError(
                "source_git_sha, source_tree_sha and image_digest are required"
            )
        if self.chunked_prefill_size <= 0:
            raise ValueError("chunked_prefill_size must be positive")
        if self.chunk_source not in {"cl2", "provisional_worst_case"}:
            raise ValueError("chunk_source must be cl2 or provisional_worst_case")
        if self.warmup_repeats < 0 or self.formal_repeats <= 0 or self.restarts <= 0:
            raise ValueError("invalid warmup/formal/restart counts")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def settings_payload(settings: Phase6RunSettings) -> dict[str, Any]:
    return asdict(settings)


def artifact_schema_payload() -> dict[str, Any]:
    return {
        "required_fields": list(PHASE6_ARTIFACT_REQUIRED_FIELDS),
        "status_values": list(PHASE6_ARTIFACT_STATUS_VALUES),
        "ledger_fields": list(PHASE6_LEDGER_FIELDS),
        "capacity_fields": ["tokens", "pages", "bytes"],
        "rho_fields": [
            "logical_demand",
            "physical_demand",
            "resident",
            "host",
        ],
    }


def validate_phase6_artifact(payload: dict[str, Any]) -> None:
    missing = [
        field for field in PHASE6_ARTIFACT_REQUIRED_FIELDS if field not in payload
    ]
    if missing:
        raise ValueError(f"missing Phase6 artifact fields: {missing}")
    if payload["status"] not in PHASE6_ARTIFACT_STATUS_VALUES:
        raise ValueError(f"invalid Phase6 artifact status: {payload['status']!r}")
    missing_ledger = [
        field for field in PHASE6_LEDGER_FIELDS if field not in payload["ledger"]
    ]
    if missing_ledger:
        raise ValueError(f"missing Phase6 ledger fields: {missing_ledger}")
