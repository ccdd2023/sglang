from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from benchmark.approx_kv.build_phase7_manifest import (
    DESIGN_KEYS,
    RUNNER_SPECS,
    design_payload_sha256,
    payload_sha256,
)
from benchmark.approx_kv.phase7.correction_review import (
    validate_correction_review_binding,
)
from benchmark.approx_kv.phase7.evidence import validate_runner_test_evidence

BASE_MANIFEST_REVISION = 12
BASE_MANIFEST_SELF_SHA256 = (
    "2d66a1bcdb6dc92a72c59fefc581212fcd541accbc8ededa221495d30d039bef"
)
BASE_MANIFEST_DESIGN_SHA256 = (
    "50003145f2e7f0e866613dbd420e73ba3983a6c182a360d6918098b1d1f7b987"
)
BASE_MANIFEST_PATH = "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json"
CAPACITY_CORRECTION_MANIFEST_PATH = (
    "benchmark/approx_kv/results/phase7/" "phase7-capacity-correction-manifest.json"
)
CAPACITY_CORRECTION_REVIEW_PATH = (
    "benchmark/approx_kv/results/phase7/" "phase7-capacity-correction-opus-review.json"
)
CAPACITY_CORRECTION_CPU_EVIDENCE_PATH = (
    "benchmark/approx_kv/results/phase7/evidence/capacity-correction-cpu.json"
)
RESULT_MANIFEST_PATH = "benchmark/approx_kv/results/phase7/RESULT_MANIFEST.json"
ORIGINAL_RAW_PATH = (
    "benchmark/approx_kv/results/phase7/raw/" "p6delta-s0-rho2-chunk4096-r0.json"
)
ORIGINAL_RAW_SHA256 = "80e8e83d7b587b1ed566889e1603eead82eb2618b58a2f9a1816fb8eae741ff3"
CAPACITY_CORRECTION_ARTIFACT = "phase7-capacity-correction-manifest"
CAPACITY_CORRECTION_SCHEMA_VERSION = 1
CAPACITY_TERMINAL_REASON_CORRECTION_SCOPE = "capacity_terminal_reason"
CAPACITY_CORRECTION_SETTING_ID = "p6delta-s0-rho2-chunk4096"
CAPACITY_CORRECTION_RESTART = 0
CAPACITY_RUNNER_KEY = "capacity_pilot"
CAPACITY_RUNNER_MODULE = RUNNER_SPECS[CAPACITY_RUNNER_KEY]["module"]
CAPACITY_RUNNER_PATH = RUNNER_SPECS[CAPACITY_RUNNER_KEY]["path"]
CAPACITY_CORRECTION_RUNTIME_ROOT = "/results/phase7-capacity-correction"
CAPACITY_CORRECTION_POST_PIN_ALLOWLIST = (
    RESULT_MANIFEST_PATH,
    CAPACITY_CORRECTION_MANIFEST_PATH,
    CAPACITY_CORRECTION_REVIEW_PATH,
    CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correction_manifest_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("correction_manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def design_key_value_bytes(payload: Mapping[str, Any], key: str) -> bytes:
    return json.dumps(
        payload[key],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def design_key_value_sha256(payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: hashlib.sha256(design_key_value_bytes(payload, key)).hexdigest()
        for key in DESIGN_KEYS
    }


def correction_artifact_templates() -> dict[str, str]:
    root = CAPACITY_CORRECTION_RUNTIME_ROOT
    return {
        "runtime_staging_root": root,
        "staging_raw_json": f"{root}/raw/{{run_id}}.json",
        "staging_server_log": f"{root}/logs/{{run_id}}.log",
        "staging_central_log": f"{root}/phase7-runs.jsonl",
        "versioned_destination_root": (
            "benchmark/approx_kv/results/phase7/capacity-correction"
        ),
        "versioned_destination_raw_json": (
            "benchmark/approx_kv/results/phase7/capacity-correction/"
            "raw/{run_id}.json"
        ),
        "versioned_destination_server_log": (
            "benchmark/approx_kv/results/phase7/capacity-correction/"
            "logs/{run_id}.log"
        ),
        "versioned_destination_central_log": (
            "benchmark/approx_kv/results/phase7/capacity-correction/"
            "phase7-runs.jsonl"
        ),
    }


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} is not a lowercase SHA-256")
    return value


def _require_git_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} is not a full git SHA")
    return value


def _repo_relative(path: Path, repo_root: Path, *, field: str) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError as error:
        raise ValueError(f"{field} must be inside the repository") from error


def validate_base_manifest(base_manifest: Mapping[str, Any]) -> None:
    if (
        base_manifest.get("artifact") != "phase7-primary-manifest"
        or base_manifest.get("schema_version") != 1
        or base_manifest.get("manifest_revision") != BASE_MANIFEST_REVISION
        or base_manifest.get("status") != "authorized"
        or base_manifest.get("phase7_execution_authorized") is not True
        or base_manifest.get("execution_blockers") != []
    ):
        raise ValueError("base manifest is not the authorized rev12 primary manifest")
    if (
        base_manifest.get("preregistered_manifest_sha256") != BASE_MANIFEST_SELF_SHA256
        or payload_sha256(dict(base_manifest)) != BASE_MANIFEST_SELF_SHA256
    ):
        raise ValueError("base manifest self-hash mismatch")
    if (
        base_manifest.get("design_payload_sha256") != BASE_MANIFEST_DESIGN_SHA256
        or design_payload_sha256(dict(base_manifest)) != BASE_MANIFEST_DESIGN_SHA256
    ):
        raise ValueError("base manifest design hash mismatch")
    if base_manifest.get("design_keys") != list(DESIGN_KEYS):
        raise ValueError("base manifest DESIGN_KEYS drifted")


def load_capacity_cpu_evidence(
    path: Path,
    *,
    runner_sha256: str,
    image_digest: str,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_runner_test_evidence(payload)
    expected = {
        "runner_key": CAPACITY_RUNNER_KEY,
        "runner_module": CAPACITY_RUNNER_MODULE,
        "runner_path": CAPACITY_RUNNER_PATH,
        "runner_sha256": runner_sha256,
        "image_digest": image_digest,
        "command": RUNNER_SPECS[CAPACITY_RUNNER_KEY]["required_cpu_test"],
        "exit_code": 0,
    }
    drifted = {
        field: (payload.get(field), value)
        for field, value in expected.items()
        if payload.get(field) != value
    }
    if drifted:
        raise ValueError(f"capacity CPU evidence binding mismatch: {drifted}")
    relative = _repo_relative(path, repo_root, field="capacity CPU evidence")
    summary = {
        "status": "passed",
        "path": relative,
        "file_sha256": file_sha256(path),
        "artifact_sha256": payload["artifact_sha256"],
        "runner_sha256": payload["runner_sha256"],
        "image_digest": payload["image_digest"],
        "command": payload["command"],
        "exit_code": payload["exit_code"],
        "summary_line": payload["summary_line"],
        "passed_count": payload["passed_count"],
        "subtests": payload["subtests"],
        "timestamp": payload["timestamp"],
    }
    return payload, summary


def pending_review_evidence() -> dict[str, Any]:
    return {
        "status": "pending",
        "artifact_path": CAPACITY_CORRECTION_REVIEW_PATH,
        "file_sha256": None,
        "artifact_sha256": None,
        "verdict": None,
        "open_p0": None,
        "open_p1": None,
        "reviewed_correction_manifest_status": None,
        "reviewed_correction_manifest_revision": None,
        "reviewed_correction_manifest_sha256": None,
    }


def passed_review_evidence(
    review: Mapping[str, Any],
    *,
    review_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "status": "passed",
        "artifact_path": _repo_relative(
            review_path,
            repo_root,
            field="correction review",
        ),
        "file_sha256": file_sha256(review_path),
        "artifact_sha256": review["artifact_sha256"],
        "verdict": review["verdict"],
        "open_p0": review["open_p0"],
        "open_p1": review["open_p1"],
        "reviewed_correction_manifest_status": review[
            "reviewed_correction_manifest_status"
        ],
        "reviewed_correction_manifest_revision": review[
            "reviewed_correction_manifest_revision"
        ],
        "reviewed_correction_manifest_sha256": review[
            "reviewed_correction_manifest_sha256"
        ],
    }


def build_pinned_capacity_correction_manifest(
    *,
    base_manifest: Mapping[str, Any],
    base_manifest_path: Path,
    base_manifest_file_sha256: str,
    original_raw_file_sha256: str,
    correction_manifest_revision: int,
    correction_pinned_implementation_sha: str,
    correction_pinned_tree_sha: str,
    capacity_runner_sha256: str,
    capacity_cpu_evidence: Mapping[str, Any],
    manifest_generation_sha: str,
    manifest_generation_tree_sha: str,
) -> dict[str, Any]:
    validate_base_manifest(base_manifest)
    manifest: dict[str, Any] = {
        "schema_version": CAPACITY_CORRECTION_SCHEMA_VERSION,
        "artifact": CAPACITY_CORRECTION_ARTIFACT,
        "correction_manifest_revision": correction_manifest_revision,
        "supersedes_correction_manifest_sha256": None,
        "status": "pinned_blocked",
        "phase7_execution_authorized": False,
        "execution_blockers": ["correction_opus_review_pending"],
        "base_manifest_revision": BASE_MANIFEST_REVISION,
        "base_manifest_self_sha256": BASE_MANIFEST_SELF_SHA256,
        "base_manifest_design_sha256": BASE_MANIFEST_DESIGN_SHA256,
        "base_manifest_path": str(base_manifest_path),
        "base_manifest_file_sha256": base_manifest_file_sha256,
        "design_keys": list(DESIGN_KEYS),
        "base_design_key_value_sha256": design_key_value_sha256(base_manifest),
        "design_payload_sha256": BASE_MANIFEST_DESIGN_SHA256,
        "scope": CAPACITY_TERMINAL_REASON_CORRECTION_SCOPE,
        "allowed_setting": CAPACITY_CORRECTION_SETTING_ID,
        "restart": CAPACITY_CORRECTION_RESTART,
        "original_raw_path": ORIGINAL_RAW_PATH,
        "original_raw_sha256": ORIGINAL_RAW_SHA256,
        "original_raw_file_sha256": original_raw_file_sha256,
        "correction_pinned_implementation_sha": (correction_pinned_implementation_sha),
        "correction_pinned_tree_sha": correction_pinned_tree_sha,
        "manifest_generation_sha": manifest_generation_sha,
        "manifest_generation_tree_sha": manifest_generation_tree_sha,
        "capacity_runner": {
            "key": CAPACITY_RUNNER_KEY,
            "module": CAPACITY_RUNNER_MODULE,
            "path": CAPACITY_RUNNER_PATH,
            "sha256": capacity_runner_sha256,
            "required_cpu_test": RUNNER_SPECS[CAPACITY_RUNNER_KEY]["required_cpu_test"],
        },
        "capacity_runner_sha256": capacity_runner_sha256,
        "capacity_cpu_evidence": dict(capacity_cpu_evidence),
        "post_pin_allowlist": list(CAPACITY_CORRECTION_POST_PIN_ALLOWLIST),
        "post_pin_rule": (
            "the correction pin must be an ancestor of the execution HEAD; "
            "the worktree must be clean and read-only; every post-pin change "
            "must be an exact member of post_pin_allowlist"
        ),
        "correction_artifact_templates": correction_artifact_templates(),
        "correction_review_contract": {
            "correction_opus_required": True,
            "reviewed_status": "pinned_blocked",
            "artifact_path": CAPACITY_CORRECTION_REVIEW_PATH,
            "pass_condition": "zero open P0/P1 findings",
            "authorization_activation": (
                "an authorized_correction manifest must supersede the reviewed "
                "pinned_blocked correction manifest"
            ),
        },
        "review_evidence": pending_review_evidence(),
    }
    for key in DESIGN_KEYS:
        manifest[key] = copy.deepcopy(base_manifest[key])
    manifest["correction_manifest_sha256"] = correction_manifest_payload_sha256(
        manifest
    )
    return manifest


def build_authorized_capacity_correction_manifest(
    *,
    reviewed_manifest: Mapping[str, Any],
    review: Mapping[str, Any],
    review_path: Path,
    repo_root: Path,
    correction_manifest_revision: int,
    manifest_generation_sha: str,
    manifest_generation_tree_sha: str,
) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(reviewed_manifest))
    manifest.update(
        {
            "correction_manifest_revision": correction_manifest_revision,
            "supersedes_correction_manifest_sha256": reviewed_manifest[
                "correction_manifest_sha256"
            ],
            "status": "authorized_correction",
            "phase7_execution_authorized": True,
            "execution_blockers": [],
            "manifest_generation_sha": manifest_generation_sha,
            "manifest_generation_tree_sha": manifest_generation_tree_sha,
            "review_evidence": passed_review_evidence(
                review,
                review_path=review_path,
                repo_root=repo_root,
            ),
        }
    )
    manifest["correction_manifest_sha256"] = correction_manifest_payload_sha256(
        manifest
    )
    validate_correction_review_binding(review, activating_manifest=manifest)
    return manifest


def _validate_post_pin_allowlist(payload: Mapping[str, Any]) -> None:
    allowlist = payload.get("post_pin_allowlist")
    if allowlist != list(CAPACITY_CORRECTION_POST_PIN_ALLOWLIST):
        raise ValueError("correction post-pin allowlist drifted")
    if len(set(allowlist)) != len(allowlist):
        raise ValueError("correction post-pin allowlist has duplicates")
    for entry in allowlist:
        path = PurePosixPath(entry)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("correction post-pin allowlist escapes the repository")
        if not entry.startswith("benchmark/approx_kv/results/phase7/"):
            raise ValueError("correction post-pin allowlist escapes Phase7 results")


def validate_capacity_correction_manifest(
    payload: Mapping[str, Any],
    *,
    base_manifest: Mapping[str, Any],
    require_authorized: bool,
    review: Mapping[str, Any] | None = None,
) -> None:
    validate_base_manifest(base_manifest)
    if payload.get("schema_version") != CAPACITY_CORRECTION_SCHEMA_VERSION:
        raise ValueError("unsupported correction manifest schema")
    if payload.get("artifact") != CAPACITY_CORRECTION_ARTIFACT:
        raise ValueError("not a capacity correction manifest")
    revision = payload.get("correction_manifest_revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("correction manifest revision must be a positive integer")
    if payload.get("correction_manifest_sha256") != (
        correction_manifest_payload_sha256(payload)
    ):
        raise ValueError("correction manifest self-hash mismatch")
    expected_base = {
        "base_manifest_revision": BASE_MANIFEST_REVISION,
        "base_manifest_self_sha256": BASE_MANIFEST_SELF_SHA256,
        "base_manifest_design_sha256": BASE_MANIFEST_DESIGN_SHA256,
        "base_manifest_path": BASE_MANIFEST_PATH,
        "design_keys": list(DESIGN_KEYS),
        "base_design_key_value_sha256": design_key_value_sha256(base_manifest),
        "design_payload_sha256": BASE_MANIFEST_DESIGN_SHA256,
    }
    drifted = {
        field: (payload.get(field), value)
        for field, value in expected_base.items()
        if payload.get(field) != value
    }
    if drifted:
        raise ValueError(f"correction base manifest binding mismatch: {drifted}")
    for key in DESIGN_KEYS:
        if key not in payload:
            raise ValueError(f"correction manifest lacks base design key {key}")
        if design_key_value_bytes(payload, key) != design_key_value_bytes(
            base_manifest, key
        ):
            raise ValueError(
                f"correction manifest design key {key} is not byte-identical"
            )
    if design_payload_sha256(dict(payload)) != BASE_MANIFEST_DESIGN_SHA256:
        raise ValueError("correction design payload hash drifted")
    expected_scope = {
        "scope": CAPACITY_TERMINAL_REASON_CORRECTION_SCOPE,
        "allowed_setting": CAPACITY_CORRECTION_SETTING_ID,
        "restart": CAPACITY_CORRECTION_RESTART,
        "original_raw_path": ORIGINAL_RAW_PATH,
        "original_raw_sha256": ORIGINAL_RAW_SHA256,
    }
    scope_drift = {
        field: (payload.get(field), value)
        for field, value in expected_scope.items()
        if payload.get(field) != value
    }
    if scope_drift:
        raise ValueError(f"correction scope binding mismatch: {scope_drift}")
    _require_sha256(
        payload.get("base_manifest_file_sha256"),
        field="base_manifest_file_sha256",
    )
    _require_sha256(
        payload.get("original_raw_file_sha256"),
        field="original_raw_file_sha256",
    )
    pin = _require_git_sha(
        payload.get("correction_pinned_implementation_sha"),
        field="correction_pinned_implementation_sha",
    )
    tree = _require_git_sha(
        payload.get("correction_pinned_tree_sha"),
        field="correction_pinned_tree_sha",
    )
    if pin == base_manifest["implementation"]["phase7_pinned_implementation_sha"]:
        raise ValueError("correction must use a new implementation pin")
    _require_git_sha(
        payload.get("manifest_generation_sha"),
        field="manifest_generation_sha",
    )
    _require_git_sha(
        payload.get("manifest_generation_tree_sha"),
        field="manifest_generation_tree_sha",
    )
    runner = payload.get("capacity_runner")
    expected_runner = {
        "key": CAPACITY_RUNNER_KEY,
        "module": CAPACITY_RUNNER_MODULE,
        "path": CAPACITY_RUNNER_PATH,
        "required_cpu_test": RUNNER_SPECS[CAPACITY_RUNNER_KEY]["required_cpu_test"],
    }
    if not isinstance(runner, Mapping):
        raise ValueError("correction capacity runner binding is missing")
    runner_drift = {
        field: (runner.get(field), value)
        for field, value in expected_runner.items()
        if runner.get(field) != value
    }
    if runner_drift:
        raise ValueError(f"correction capacity runner binding drifted: {runner_drift}")
    runner_sha = _require_sha256(
        runner.get("sha256"),
        field="capacity_runner.sha256",
    )
    if payload.get("capacity_runner_sha256") != runner_sha:
        raise ValueError("correction capacity runner SHA aliases drifted")
    evidence = payload.get("capacity_cpu_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("status") != "passed":
        raise ValueError("correction capacity CPU evidence is not passed")
    for field in ("file_sha256", "artifact_sha256", "runner_sha256"):
        _require_sha256(evidence.get(field), field=f"capacity_cpu_evidence.{field}")
    if (
        evidence.get("path") != CAPACITY_CORRECTION_CPU_EVIDENCE_PATH
        or evidence.get("runner_sha256") != runner_sha
        or evidence.get("image_digest") != base_manifest["environment"]["image_digest"]
        or evidence.get("command")
        != RUNNER_SPECS[CAPACITY_RUNNER_KEY]["required_cpu_test"]
        or evidence.get("exit_code") != 0
    ):
        raise ValueError("correction capacity CPU evidence binding drifted")
    _validate_post_pin_allowlist(payload)
    if payload.get("correction_artifact_templates") != correction_artifact_templates():
        raise ValueError("correction artifact templates drifted")
    review_contract = payload.get("correction_review_contract")
    if (
        not isinstance(review_contract, Mapping)
        or review_contract.get("correction_opus_required") is not True
        or review_contract.get("reviewed_status") != "pinned_blocked"
        or review_contract.get("artifact_path") != CAPACITY_CORRECTION_REVIEW_PATH
    ):
        raise ValueError("correction review contract drifted")
    status = payload.get("status")
    authorized = payload.get("phase7_execution_authorized")
    blockers = payload.get("execution_blockers")
    review_evidence = payload.get("review_evidence")
    if not isinstance(review_evidence, Mapping):
        raise ValueError("correction review evidence is missing")
    if status == "pinned_blocked":
        if (
            authorized is not False
            or blockers != ["correction_opus_review_pending"]
            or payload.get("supersedes_correction_manifest_sha256") is not None
            or review_evidence != pending_review_evidence()
        ):
            raise ValueError("invalid pinned_blocked correction state")
    elif status == "authorized_correction":
        if (
            authorized is not True
            or blockers != []
            or review_evidence.get("status") != "passed"
        ):
            raise ValueError("invalid authorized_correction state")
        _require_sha256(
            payload.get("supersedes_correction_manifest_sha256"),
            field="supersedes_correction_manifest_sha256",
        )
        for field in ("file_sha256", "artifact_sha256"):
            _require_sha256(
                review_evidence.get(field),
                field=f"review_evidence.{field}",
            )
        if review_evidence.get("artifact_path") != CAPACITY_CORRECTION_REVIEW_PATH:
            raise ValueError("correction review evidence path drifted")
        if review is not None:
            validate_correction_review_binding(
                review,
                activating_manifest=payload,
            )
            expected_review_summary = {
                "status": "passed",
                "artifact_path": CAPACITY_CORRECTION_REVIEW_PATH,
                "file_sha256": review_evidence["file_sha256"],
                "artifact_sha256": review["artifact_sha256"],
                "verdict": review["verdict"],
                "open_p0": review["open_p0"],
                "open_p1": review["open_p1"],
                "reviewed_correction_manifest_status": review[
                    "reviewed_correction_manifest_status"
                ],
                "reviewed_correction_manifest_revision": review[
                    "reviewed_correction_manifest_revision"
                ],
                "reviewed_correction_manifest_sha256": review[
                    "reviewed_correction_manifest_sha256"
                ],
            }
            if review_evidence != expected_review_summary:
                raise ValueError("correction review evidence summary drifted")
    else:
        raise ValueError(f"unsupported correction manifest status: {status!r}")
    if require_authorized and status != "authorized_correction":
        raise ValueError("capacity correction execution requires authorization")
    if not _is_setting_byte_identical(payload, base_manifest):
        raise ValueError("correction allowed setting is not byte-identical to rev12")
    if not pin or not tree:
        raise ValueError("correction pin is missing")


def _is_setting_byte_identical(
    payload: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
) -> bool:
    settings = {
        row["setting_id"]: row
        for row in payload.get("settings", [])
        if isinstance(row, Mapping) and isinstance(row.get("setting_id"), str)
    }
    base_settings = {
        row["setting_id"]: row
        for row in base_manifest.get("settings", [])
        if isinstance(row, Mapping) and isinstance(row.get("setting_id"), str)
    }
    setting_id = payload.get("allowed_setting")
    if setting_id not in settings or setting_id not in base_settings:
        return False
    return json.dumps(settings[setting_id], separators=(",", ":")).encode(
        "utf-8"
    ) == json.dumps(base_settings[setting_id], separators=(",", ":")).encode("utf-8")


def verify_capacity_correction_files(
    payload: Mapping[str, Any],
    *,
    base_manifest: Mapping[str, Any],
    manifest_path: Path,
    repo_root: Path,
    verify_git: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base_path = repo_root / payload["base_manifest_path"]
    if file_sha256(base_path) != payload["base_manifest_file_sha256"]:
        raise ValueError("correction base manifest file SHA-256 mismatch")
    original_path = repo_root / payload["original_raw_path"]
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if (
        file_sha256(original_path) != payload["original_raw_file_sha256"]
        or original.get("raw_sha256") != payload["original_raw_sha256"]
    ):
        raise ValueError("correction original raw binding mismatch")
    if _repo_relative(manifest_path, repo_root, field="correction manifest") != (
        CAPACITY_CORRECTION_MANIFEST_PATH
    ):
        raise ValueError("correction manifest path drifted")
    evidence_path = repo_root / payload["capacity_cpu_evidence"]["path"]
    evidence, summary = load_capacity_cpu_evidence(
        evidence_path,
        runner_sha256=payload["capacity_runner_sha256"],
        image_digest=base_manifest["environment"]["image_digest"],
        repo_root=repo_root,
    )
    if summary != payload["capacity_cpu_evidence"]:
        raise ValueError("correction capacity CPU evidence summary mismatch")
    review = None
    if payload["status"] == "authorized_correction":
        review_path = repo_root / payload["review_evidence"]["artifact_path"]
        review = json.loads(review_path.read_text(encoding="utf-8"))
        validate_correction_review_binding(review, activating_manifest=payload)
        if (
            file_sha256(review_path) != payload["review_evidence"]["file_sha256"]
            or review.get("artifact_sha256")
            != payload["review_evidence"]["artifact_sha256"]
        ):
            raise ValueError("correction review artifact hash mismatch")
    validate_capacity_correction_manifest(
        payload,
        base_manifest=base_manifest,
        require_authorized=False,
        review=review,
    )
    if verify_git:
        pin = payload["correction_pinned_implementation_sha"]
        resolved_pin = subprocess.run(
            ("git", "rev-parse", f"{pin}^{{commit}}"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        resolved_tree = subprocess.run(
            ("git", "rev-parse", f"{pin}^{{tree}}"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if resolved_pin != pin:
            raise ValueError("correction implementation pin does not resolve exactly")
        if resolved_tree != payload["correction_pinned_tree_sha"]:
            raise ValueError("correction implementation tree mismatch")
        ancestry = subprocess.run(
            ("git", "merge-base", "--is-ancestor", pin, "HEAD"),
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if ancestry.returncode != 0:
            raise ValueError("correction implementation pin is not an ancestor of HEAD")
        runner_blob = subprocess.run(
            ("git", "show", f"{pin}:{CAPACITY_RUNNER_PATH}"),
            cwd=repo_root,
            capture_output=True,
            check=True,
        ).stdout
        if hashlib.sha256(runner_blob).hexdigest() != payload["capacity_runner_sha256"]:
            raise ValueError("correction pinned capacity runner hash mismatch")
        if (
            file_sha256(repo_root / CAPACITY_RUNNER_PATH)
            != payload["capacity_runner_sha256"]
        ):
            raise ValueError("current capacity runner hash mismatch")
        for field, relative in (
            ("base_manifest_file_sha256", payload["base_manifest_path"]),
            ("original_raw_file_sha256", payload["original_raw_path"]),
        ):
            blob = subprocess.run(
                ("git", "show", f"{pin}:{relative}"),
                cwd=repo_root,
                capture_output=True,
                check=True,
            ).stdout
            if hashlib.sha256(blob).hexdigest() != payload[field]:
                raise ValueError(f"correction pin {field} binding mismatch")
    return evidence, review
