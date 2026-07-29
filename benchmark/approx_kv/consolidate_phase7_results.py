from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.approx_kv.phase7.correction import (
    BASE_MANIFEST_DESIGN_SHA256 as AUTHORIZED_DESIGN_SHA256,
)
from benchmark.approx_kv.phase7.correction import (
    BASE_MANIFEST_REVISION as AUTHORIZED_MANIFEST_REVISION,
)
from benchmark.approx_kv.phase7.correction import (
    BASE_MANIFEST_SELF_SHA256 as AUTHORIZED_MANIFEST_SHA256,
)
from benchmark.approx_kv.phase7.correction import (
    CAPACITY_CORRECTION_MANIFEST_PATH,
    CAPACITY_CORRECTION_REVIEW_PATH,
    CAPACITY_CORRECTION_RESTART,
    CAPACITY_CORRECTION_SETTING_ID,
    CAPACITY_TERMINAL_REASON_CORRECTION_SCOPE,
)
from benchmark.approx_kv.phase7.correction import (
    validate_capacity_correction_manifest as validate_correction_schema,
)
from benchmark.approx_kv.phase7.correction import (
    verify_capacity_correction_files,
)
from benchmark.approx_kv.phase7.correction_review import (
    validate_correction_review_binding,
)

ENGINEERING_STATUS_VALID = "VALID"
MECHANISM_STATUS_NEGATIVE = "NEGATIVE"
SYSTEM_BEHAVIOUR_STATUS = "INCONCLUSIVE/DESCRIPTIVE"
ALLOWED_RAW_STATUSES = {"valid", "inconclusive"}
RESET_ZERO_GAUGES = {
    "sglang:approx_kv_provisional_tokens",
    "sglang:approx_kv_store_device_bytes",
    "sglang:approx_kv_store_host_bytes",
    "sglang:approx_kv_store_leases",
    "sglang:approx_kv_store_orphans",
    "sglang:approx_kv_store_records",
    "sglang:cross_store_reserved_device_bytes",
}
W_DENOMINATORS = ("all_reusable", "workflow_only")
EXPECTED_PHASE_BY_RUNNER = {
    "benchmark.approx_kv.run_p6_4_capacity_pilot": "Phase7-capacity",
    "benchmark.approx_kv.run_p7_ceiling": "Phase7-ceiling",
    "benchmark.approx_kv.run_p7_scheduler": "Phase7-scheduler",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


class ConsolidationError(ValueError):
    pass


def canonical_sha256(payload: Mapping[str, Any], hash_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(hash_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialized_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def normalized_diagnostic(value: Any) -> Any:
    if value == "diagnostic-unavailable":
        return "diagnostic_unavailable"
    return value


def _positive_terminal_reasons(
    observations: Mapping[str, Any] | None,
) -> dict[str, float]:
    if not isinstance(observations, Mapping):
        return {}
    return {
        reason: float(value)
        for reason, value in observations.get("mapped", {}).items()
        if value is not None and float(value) > 0
    }


def _record_outcome_composition(
    records: Sequence[Mapping[str, Any]],
    *,
    outcome_taxonomy: Sequence[str],
    terminal_reasons: Sequence[str],
) -> dict[str, Any]:
    counts = {
        outcome: sum(record.get("outcome") == outcome for record in records)
        for outcome in outcome_taxonomy
    }
    reason_counts = {
        reason: sum(record.get("terminal_reason") == reason for record in records)
        for reason in terminal_reasons
    }
    total = len(records)
    recovery = counts.get("approximate_gpu_recovery", 0)
    fallback = counts.get("approximate_recovery_failed_dense", 0)
    return {
        "requests": total,
        "counts": counts,
        "recovery": {
            "count": recovery,
            "rate": recovery / total if total else None,
        },
        "dense_fallback": {
            "count": fallback,
            "rate": fallback / total if total else None,
        },
        "terminal_reason_counts": reason_counts,
    }


def attach_self_hash(payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    payload.pop(hash_field, None)
    payload[hash_field] = canonical_sha256(payload, hash_field)
    return payload


def _manifest_setting_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    settings = manifest.get("settings")
    if not isinstance(settings, list):
        raise ConsolidationError("manifest settings must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for setting in settings:
        setting_id = setting.get("setting_id")
        if not isinstance(setting_id, str) or setting_id in result:
            raise ConsolidationError("manifest setting IDs are missing or duplicated")
        result[setting_id] = setting
    return result


def validate_authorized_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_revision") != AUTHORIZED_MANIFEST_REVISION:
        raise ConsolidationError("Phase7 manifest is not authorized revision 12")
    if manifest.get("status") != "authorized":
        raise ConsolidationError("Phase7 manifest status is not authorized")
    if manifest.get("phase7_execution_authorized") is not True:
        raise ConsolidationError("Phase7 execution authorization is not active")
    if manifest.get("execution_blockers") != []:
        raise ConsolidationError("authorized manifest still has execution blockers")
    manifest_sha = manifest.get("preregistered_manifest_sha256")
    if manifest_sha != AUTHORIZED_MANIFEST_SHA256:
        raise ConsolidationError("authorized Phase7 manifest SHA-256 mismatch")
    if canonical_sha256(manifest, "preregistered_manifest_sha256") != manifest_sha:
        raise ConsolidationError("authorized Phase7 manifest self-hash mismatch")
    if manifest.get("design_payload_sha256") != AUTHORIZED_DESIGN_SHA256:
        raise ConsolidationError("authorized Phase7 design hash mismatch")
    if manifest.get("r2_strategy") != "disabled_not_comparable":
        raise ConsolidationError("Phase7 R2 disposition drifted")
    resolution = manifest.get("conditional_resolution", {})
    if resolution.get("CR-P6DELTA-RHO3") != "disabled_scoped_chunk1024":
        raise ConsolidationError("Phase7 rho3 conditional disposition drifted")
    if resolution.get("CR-R2-ADAPTER") != "disabled_not_comparable":
        raise ConsolidationError("Phase7 R2 conditional disposition drifted")
    budget = manifest.get("budget", {})
    if budget.get("committed_server_starts") != 30:
        raise ConsolidationError("Phase7 committed-start budget drifted")
    if budget.get("hard_cap_server_starts") != 36:
        raise ConsolidationError("Phase7 hard start cap drifted")
    if float(budget.get("expected_gpu_hours_total", -1)) != 3.8:
        raise ConsolidationError("Phase7 expected GPU-hour budget drifted")
    if float(budget.get("hard_cap_gpu_hours", -1)) != 6.0:
        raise ConsolidationError("Phase7 GPU-hour hard cap drifted")

    expected_setting_ids = {
        "p6delta-s4-rho2-chunk4096",
        "p6delta-s0-rho2-chunk4096",
        "p6delta-s4-rho3-chunk4096",
        "p7-a8-r0-body1024-rho1.5",
        "p7-a8-r0-body1024-rho2.0",
        "p7-a8-r0-body2048-rho1.5",
        "p7-a8-r0-body2048-rho2.0",
        "p7-a8-r0-body2048-rho2-chunk1024-sensitivity",
        "p7-w-r0-lru-rho1.5",
        "p7-w-r0-hierarchical-rho1.5",
        "p7-w-r0-lru-rho2.0",
        "p7-w-r0-hierarchical-rho2.0",
        "p7-w-r4like-lru-rho2",
        "p7-w-r4like-hierarchical-rho2",
    }
    if set(_manifest_setting_map(manifest)) != expected_setting_ids:
        raise ConsolidationError("Phase7 setting matrix drifted from authorized rev12")


def validate_correction_manifest(
    manifest: Mapping[str, Any],
    *,
    primary_manifest: Mapping[str, Any],
    manifest_path: Path | None = None,
    repo_root: Path | None = None,
    verify_git: bool = False,
) -> None:
    try:
        validate_correction_schema(
            manifest,
            base_manifest=primary_manifest,
            require_authorized=True,
        )
        if manifest_path is not None and repo_root is not None:
            verify_capacity_correction_files(
                manifest,
                base_manifest=primary_manifest,
                manifest_path=manifest_path,
                repo_root=repo_root,
                verify_git=verify_git,
            )
    except (
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise ConsolidationError(f"correction manifest is invalid: {error}") from error


def expected_execution_plan(
    manifest: Mapping[str, Any],
) -> dict[str, list[tuple[str, int]]]:
    validate_authorized_manifest(manifest)
    settings = _manifest_setting_map(manifest)

    wave0 = [
        ("p6delta-s4-rho2-chunk4096", 0),
        ("p6delta-s0-rho2-chunk4096", 0),
    ]
    a8_primary_restart0 = [
        ("p7-a8-r0-body1024-rho1.5", 0),
        ("p7-a8-r0-body1024-rho2.0", 0),
        ("p7-a8-r0-body2048-rho1.5", 0),
        ("p7-a8-r0-body2048-rho2.0", 0),
    ]
    a8_primary_supplements_skipped = [
        (setting_id, restart)
        for setting_id, _ in a8_primary_restart0
        for restart in (1, 2)
    ]
    sensitivity = [
        ("p7-a8-r0-body2048-rho2-chunk1024-sensitivity", restart) for restart in (0, 1)
    ]
    w_main = [
        (setting_id, restart)
        for setting_id in (
            "p7-w-r0-lru-rho1.5",
            "p7-w-r0-hierarchical-rho1.5",
            "p7-w-r0-lru-rho2.0",
            "p7-w-r0-hierarchical-rho2.0",
        )
        for restart in (0, 1, 2)
    ]
    r4 = [
        ("p7-w-r4like-lru-rho2", 0),
        ("p7-w-r4like-hierarchical-rho2", 0),
    ]
    rho3_disabled = [("p6delta-s4-rho3-chunk4096", 0)]
    executed = wave0 + a8_primary_restart0 + sensitivity + w_main + r4

    for setting_id, restart in executed + a8_primary_supplements_skipped:
        setting = settings.get(setting_id)
        if setting is None or restart not in setting.get("restart_indices", []):
            raise ConsolidationError(
                f"authorized setting/restart missing: {setting_id} restart {restart}"
            )
    rho3_setting = settings[rho3_disabled[0][0]]
    if rho3_setting.get("conditional") is not True:
        raise ConsolidationError("rho3 setting is no longer conditional")
    if len(executed) != 22:
        raise ConsolidationError("internal Phase7 executed-start contract is not 22")
    return {
        "wave0_required": wave0,
        "a8_primary_restart0": a8_primary_restart0,
        "a8_primary_supplements_skipped_es_r0_mde": (a8_primary_supplements_skipped),
        "chunk1024_sensitivity": sensitivity,
        "w_main": w_main,
        "r4_diagnostic": r4,
        "rho3_conditional_disabled": rho3_disabled,
        "executed": executed,
    }


def _require_passed(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        raise ConsolidationError(f"{label} did not pass")


def _validate_inactive_counter_assertion(
    assertion: Any,
    manifest: Mapping[str, Any],
    label: str,
) -> None:
    _require_passed(assertion, label)
    required = manifest.get("required_inactive_counters")
    if assertion.get("required_counters") != required:
        raise ConsolidationError(f"{label} required-counter list mismatch")
    rows = assertion.get("assertions")
    if not isinstance(rows, Mapping) or set(rows) != set(required):
        raise ConsolidationError(f"{label} counter evidence is incomplete")
    for counter in required:
        row = rows[counter]
        if (
            not isinstance(row, Mapping)
            or row.get("manifest_pinned_disabled") is not True
        ):
            raise ConsolidationError(f"{label} {counter} is not manifest-pinned")
        verification = row.get("verification")
        value = row.get("value")
        if verification == "direct":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or float(value) != 0.0
            ):
                raise ConsolidationError(f"{label} {counter} is nonzero or invalid")
        elif verification == "indirectly_verified":
            if value is not None:
                raise ConsolidationError(
                    f"{label} {counter} fabricated an indirect zero"
                )
        else:
            raise ConsolidationError(f"{label} {counter} verification is invalid")


def _validate_execution_provenance(
    raw: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_hash: str,
    runner_entry: Mapping[str, Any],
    label: str,
) -> None:
    implementation = manifest["implementation"]
    envelope = raw.get("execution_envelope")
    if not isinstance(envelope, Mapping):
        raise ConsolidationError(f"{label} execution envelope is missing")
    if envelope.get("worktree_clean") is not True:
        raise ConsolidationError(f"{label} execution worktree was not clean")
    if envelope.get("pinned_is_ancestor_of_execution_head") is not True:
        raise ConsolidationError(f"{label} pinned source is not an execution ancestor")
    if (
        envelope.get("pinned_source_git_sha")
        != implementation["phase7_pinned_implementation_sha"]
        or envelope.get("pinned_source_tree_sha")
        != implementation["phase7_pinned_tree_sha"]
    ):
        raise ConsolidationError(f"{label} pinned source provenance mismatch")
    allowlist = implementation["post_pin_envelope_allowlist"]
    if envelope.get("post_pin_envelope_allowlist") != allowlist:
        raise ConsolidationError(f"{label} post-pin allowlist mismatch")
    changed_paths = envelope.get("post_pin_changed_paths")
    if not isinstance(changed_paths, list) or not set(changed_paths).issubset(
        set(allowlist)
    ):
        raise ConsolidationError(f"{label} has non-allowlisted post-pin changes")
    envelope_hashes = envelope.get("post_pin_envelope_sha256")
    if (
        not isinstance(envelope_hashes, Mapping)
        or envelope_hashes.get(
            "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json"
        )
        != manifest_file_hash
    ):
        raise ConsolidationError(f"{label} execution manifest file hash mismatch")

    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ConsolidationError(f"{label} provenance is missing")
    if provenance.get("manifest_file_sha256") != manifest_file_hash:
        raise ConsolidationError(f"{label} provenance manifest hash mismatch")
    if provenance.get("runner_sha256") != runner_entry["sha256"]:
        raise ConsolidationError(f"{label} provenance runner hash mismatch")
    if provenance.get("implementation") != implementation:
        raise ConsolidationError(f"{label} implementation provenance mismatch")
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        raise ConsolidationError(f"{label} source provenance is missing")
    if (
        source.get("source_git_sha")
        != implementation["phase7_pinned_implementation_sha"]
        or source.get("source_tree_sha") != implementation["phase7_pinned_tree_sha"]
        or raw.get("source_git_sha")
        != implementation["phase7_pinned_implementation_sha"]
        or raw.get("source_tree_sha") != implementation["phase7_pinned_tree_sha"]
    ):
        raise ConsolidationError(f"{label} source SHA provenance mismatch")


def _validate_correction_execution_provenance(
    raw: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_hash: str,
    runner_entry: Mapping[str, Any],
    label: str,
) -> None:
    envelope = raw.get("execution_envelope")
    if not isinstance(envelope, Mapping):
        raise ConsolidationError(f"{label} correction execution envelope is missing")
    if (
        envelope.get("evidence_correction_scope")
        != CAPACITY_TERMINAL_REASON_CORRECTION_SCOPE
        or envelope.get("execution_kind") != "capacity_correction"
        or envelope.get("primary_execution_envelope") is not False
        or envelope.get("pinned_is_ancestor_of_execution_head") is not True
    ):
        raise ConsolidationError(f"{label} correction execution envelope is invalid")
    if (
        envelope.get("worktree_clean") is not True
        or envelope.get("worktree_status_entries") != []
    ):
        raise ConsolidationError(f"{label} correction worktree must be clean")
    if (
        envelope.get("pinned_source_git_sha")
        != manifest["correction_pinned_implementation_sha"]
        or envelope.get("pinned_source_tree_sha")
        != manifest["correction_pinned_tree_sha"]
        or not _is_git_sha(envelope.get("execution_head_git_sha"))
        or not _is_git_sha(envelope.get("execution_head_tree_sha"))
    ):
        raise ConsolidationError(f"{label} correction pin/head provenance mismatch")
    allowlist = manifest["post_pin_allowlist"]
    envelope_hashes = envelope.get("post_pin_envelope_sha256")
    if (
        not isinstance(allowlist, list)
        or envelope.get("post_pin_envelope_allowlist") != allowlist
        or not isinstance(envelope_hashes, Mapping)
        or set(envelope_hashes) != set(allowlist)
        or any(not _is_sha256(value) for value in envelope_hashes.values())
    ):
        raise ConsolidationError(f"{label} correction manifest envelope mismatch")
    changed_paths = envelope.get("post_pin_changed_paths")
    if not isinstance(changed_paths, list) or any(
        not isinstance(path, str) for path in changed_paths
    ):
        raise ConsolidationError(f"{label} correction changed-path list is invalid")
    unexpected = sorted(set(changed_paths) - set(allowlist))
    if unexpected:
        raise ConsolidationError(
            f"{label} correction has unlisted post-pin changes: {unexpected}"
        )
    runner = raw.get("runner")
    if not isinstance(runner, Mapping):
        raise ConsolidationError(f"{label} correction runner binding is missing")
    if (
        envelope.get("correction_runner_path") != runner.get("path")
        or envelope.get("correction_runner_sha256") != runner.get("sha256")
        or envelope.get("pinned_runner_sha256") != runner.get("sha256")
        or runner.get("sha256") != runner_entry.get("sha256")
        or not _is_sha256(runner.get("sha256"))
    ):
        raise ConsolidationError(f"{label} correction runner binding mismatch")

    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ConsolidationError(f"{label} correction provenance is missing")
    manifest_path = provenance.get("manifest_path")
    expected_manifest_path = CAPACITY_CORRECTION_MANIFEST_PATH
    if (
        not isinstance(manifest_path, str)
        or not (
            manifest_path == expected_manifest_path
            or manifest_path.endswith(f"/{expected_manifest_path}")
        )
        or envelope_hashes.get(expected_manifest_path) != manifest_file_hash
        or raw.get("manifest_file_sha256") != manifest_file_hash
        or raw.get("correction_manifest_file_sha256") != manifest_file_hash
    ):
        raise ConsolidationError(f"{label} correction manifest file binding mismatch")
    source = provenance.get("source")
    expected_implementation = {
        "correction_pinned_implementation_sha": manifest[
            "correction_pinned_implementation_sha"
        ],
        "correction_pinned_tree_sha": manifest["correction_pinned_tree_sha"],
        "capacity_runner_sha256": manifest["capacity_runner_sha256"],
        "post_pin_allowlist": manifest["post_pin_allowlist"],
    }
    if (
        provenance.get("manifest_file_sha256") != manifest_file_hash
        or provenance.get("runner_sha256") != runner.get("sha256")
        or provenance.get("implementation") != expected_implementation
        or not isinstance(source, Mapping)
        or source.get("source_git_sha")
        != manifest["correction_pinned_implementation_sha"]
        or source.get("source_tree_sha") != manifest["correction_pinned_tree_sha"]
        or source.get("execution_head_git_sha")
        != envelope.get("execution_head_git_sha")
        or source.get("execution_head_tree_sha")
        != envelope.get("execution_head_tree_sha")
        or source.get("source_binding") != "dedicated_capacity_correction_pin"
        or raw.get("source_git_sha") != manifest["correction_pinned_implementation_sha"]
        or raw.get("source_tree_sha") != manifest["correction_pinned_tree_sha"]
    ):
        raise ConsolidationError(f"{label} correction source provenance mismatch")
    review_evidence = manifest["review_evidence"]
    cpu_evidence = manifest["capacity_cpu_evidence"]
    if (
        envelope_hashes.get(review_evidence["artifact_path"])
        != review_evidence["file_sha256"]
        or envelope_hashes.get(cpu_evidence["path"]) != cpu_evidence["file_sha256"]
    ):
        raise ConsolidationError(f"{label} correction evidence envelope mismatch")


def capacity_terminal_reason_correction_required(
    raw: Mapping[str, Any],
) -> bool:
    outcome = raw.get("outcome", {})
    fallback_count = int(
        outcome.get("counts", {}).get("approximate_recovery_failed_dense", 0)
    )
    reason_counts = outcome.get("terminal_reason_counts", {})
    return fallback_count > 0 and not any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) > 0
        for value in reason_counts.values()
    )


def validate_capacity_terminal_reason_correction(
    correction: Mapping[str, Any],
    *,
    original: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, int]:
    expected_binding = {
        "scope": CAPACITY_TERMINAL_REASON_CORRECTION_SCOPE,
        "original_raw_sha256": original["raw_sha256"],
        "setting_id": original["setting_id"],
        "restart_index": original["restart_index"],
    }
    if correction.get("correction") != expected_binding:
        raise ConsolidationError("capacity correction binding mismatch")
    if original.get("raw_sha256") != manifest.get("original_raw_sha256"):
        raise ConsolidationError(
            "capacity correction manifest targets a different original raw"
        )
    if (
        _manifest_setting_map(manifest).get(original["setting_id"])
        != original["setting"]
    ):
        raise ConsolidationError("capacity correction manifest setting drifted")
    if (
        correction.get("phase") != "Phase7-capacity"
        or correction.get("setting_id") != original["setting_id"]
        or correction.get("restart_index") != original["restart_index"]
        or correction.get("setting") != original["setting"]
        or correction.get("base_manifest_revision")
        != manifest["base_manifest_revision"]
        or correction.get("base_manifest_self_sha256")
        != manifest["base_manifest_self_sha256"]
        or correction.get("base_manifest_design_sha256")
        != manifest["base_manifest_design_sha256"]
        or correction.get("correction_manifest_revision")
        != manifest["correction_manifest_revision"]
        or correction.get("correction_manifest_sha256")
        != manifest["correction_manifest_sha256"]
        or correction.get("design_payload_sha256")
        != manifest["base_manifest_design_sha256"]
        or correction.get("plan") != manifest["plan"]
        or correction.get("status") not in ALLOWED_RAW_STATUSES
    ):
        raise ConsolidationError(
            "capacity correction setting/manifest/design/status mismatch"
        )
    if "error" in correction or correction.get("execution_status") in {
        "invalid",
        "error",
    }:
        raise ConsolidationError("capacity correction contains failure evidence")

    outcome = correction.get("outcome", {})
    counts = outcome.get("counts", {})
    taxonomy = tuple(manifest["outcome_taxonomy"])
    if (
        outcome.get("taxonomy") != list(taxonomy)
        or set(counts) != set(taxonomy)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts.values()
        )
    ):
        raise ConsolidationError("capacity correction outcome taxonomy is invalid")
    exclusive = tuple(manifest["exclusive_terminal_reasons"])
    reason_counts = {reason: 0 for reason in exclusive}
    fallback_rows = []
    exact_rows = []
    cells = correction.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise ConsolidationError("capacity correction must contain one full cell")
    profiles = cells[0].get("profiles", [])
    if {profile.get("profile") for profile in profiles} != {
        "exact_only",
        "r0_like",
        "r1_like_k32",
        "r2_like",
        "r4_like",
    } or len(profiles) != 5:
        raise ConsolidationError(
            "capacity correction does not execute the full setting"
        )
    for profile in profiles:
        formal = profile.get("formal", [])
        if len(formal) != 2 or any(
            len(repeat.get("replay", [])) != 5 for repeat in formal
        ):
            raise ConsolidationError(
                "capacity correction formal repeat structure drifted"
            )
        if profile.get("profile") == "exact_only":
            exact_rows.extend(
                replay for repeat in formal for replay in repeat["replay"]
            )
            continue
        profile_fallbacks = [
            replay for repeat in formal for replay in repeat.get("replay", [])
        ]
        if len(profile_fallbacks) != 10:
            raise ConsolidationError(
                f"capacity correction {profile.get('profile')} lacks 10 fallbacks"
            )
        fallback_rows.extend(profile_fallbacks)
    if len(exact_rows) != 10:
        raise ConsolidationError(
            "capacity correction exact-only profile must contain 10 requests"
        )
    if any(
        row.get("outcome") not in {"exact_gpu_hit", "exact_cache_miss"}
        or row.get("terminal_reason") is not None
        or row.get("terminal_reason_valid") is False
        for row in exact_rows
    ):
        raise ConsolidationError(
            "capacity correction exact-only outcome is outside the taxonomy"
        )
    if len(fallback_rows) != 40:
        raise ConsolidationError(
            "capacity correction must retain 40 approximate replays"
        )
    for row in fallback_rows:
        observations = row.get("terminal_reason_observations")
        positives = _positive_terminal_reasons(observations)
        if (
            row.get("outcome") != "dense_fallback"
            or not isinstance(observations, Mapping)
            or observations.get("verification") != "direct"
            or observations.get("value_unit") != "tokens"
            or not isinstance(observations.get("raw"), Mapping)
            or not isinstance(observations.get("mapped"), Mapping)
            or not isinstance(observations.get("mapped_from"), Mapping)
            or observations.get("unmapped_raw_reasons")
            or len(positives) != 1
        ):
            raise ConsolidationError(
                "capacity correction fallback lacks one direct exclusive reason"
            )
        reason = next(iter(positives))
        if (
            reason not in exclusive
            or row.get("terminal_reason") != reason
            or row.get("terminal_reason_verification") != "direct"
            or row.get("terminal_reason_valid") is not True
        ):
            raise ConsolidationError(
                "capacity correction terminal reason is not directly exclusive"
            )
        reason_counts[reason] += 1
    computed_counts = {name: 0 for name in taxonomy}
    for row in exact_rows:
        computed_counts[
            (
                "exact_gpu_hit"
                if row["outcome"] == "exact_gpu_hit"
                else "ordinary_exact_cache_miss"
            )
        ] += 1
    computed_counts["approximate_recovery_failed_dense"] = len(fallback_rows)
    if counts != computed_counts:
        raise ConsolidationError(
            "capacity correction top-level outcome counts do not match requests"
        )
    if outcome.get("terminal_reason_counts") != reason_counts:
        raise ConsolidationError("capacity correction top-level reason counts mismatch")
    validate_capacity_terminal_reason_contract(correction)
    return reason_counts


def _validate_reset_snapshot(snapshot: Any, label: str) -> None:
    _require_passed(snapshot, label)
    components = snapshot.get("components")
    if isinstance(components, Mapping):
        failed = [
            name
            for name, component in components.items()
            if isinstance(component, Mapping) and component.get("passed") is False
        ]
        if failed:
            raise ConsolidationError(f"{label} failed components: {failed}")
    gauges = snapshot.get("store_gauges")
    if isinstance(gauges, Mapping):
        nonzero = {
            name: value
            for name, value in gauges.items()
            if name in RESET_ZERO_GAUGES
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) != 0.0
        }
        if nonzero:
            raise ConsolidationError(f"{label} has nonzero reset gauges: {nonzero}")


def validate_reset_invariants(raw: Mapping[str, Any]) -> None:
    if raw.get("phase") == "Phase7-capacity":
        cells = raw.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ConsolidationError("capacity artifact is missing cells")
        for cell_index, cell in enumerate(cells):
            if float(cell.get("capacity_relative_error", 1.0)) > float(
                cell.get("capacity_relative_error_tolerance", -1.0)
            ):
                raise ConsolidationError(
                    f"capacity cell {cell_index} exceeds capacity tolerance"
                )
            for profile in cell.get("profiles", []):
                if profile.get("valid") is not True:
                    raise ConsolidationError(
                        f"capacity profile {profile.get('profile')} is invalid"
                    )
                for repeat_index, repeat in enumerate(profile.get("formal", [])):
                    _require_passed(
                        repeat.get("reset_invariant"),
                        (
                            f"capacity {profile.get('profile')} repeat "
                            f"{repeat_index} reset invariant"
                        ),
                    )
                    gauges = repeat.get("store_reset_gauges")
                    if not isinstance(gauges, Mapping):
                        raise ConsolidationError("capacity reset gauges are missing")
                    nonzero = {
                        name: value
                        for name, value in gauges.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and float(value) != 0.0
                    }
                    if nonzero:
                        raise ConsolidationError(
                            "capacity reset gauges are nonzero: "
                            f"{profile.get('profile')} {nonzero}"
                        )
        return

    reset = raw.get("reset")
    if not isinstance(reset, Mapping):
        raise ConsolidationError("artifact reset section is missing")
    _validate_reset_snapshot(reset.get("startup"), "startup reset")
    formal = raw.get("formal")
    if not isinstance(formal, list) or not formal:
        raise ConsolidationError("artifact formal repeats are missing")
    for repeat in formal:
        repeat_index = repeat.get("repeat_index")
        arms = repeat.get("arms")
        if not isinstance(arms, Mapping) or not arms:
            raise ConsolidationError("artifact formal arms are missing")
        for arm, data in arms.items():
            _validate_reset_snapshot(
                data.get("pre_reset"),
                f"repeat {repeat_index} arm {arm} pre-reset",
            )
            _validate_reset_snapshot(
                data.get("post_reset"),
                f"repeat {repeat_index} arm {arm} post-reset",
            )


def _validate_a8_contract(raw: Mapping[str, Any]) -> None:
    expected_counts = {
        "dense_no_reuse_baseline": 16,
        "exact_gpu_hit": 16,
        "approximate_gpu_recovery": 16,
        "ordinary_exact_cache_miss": 0,
        "host_demand_load": 0,
        "approximate_recovery_failed_dense": 0,
    }
    if raw.get("outcome", {}).get("counts") != expected_counts:
        raise ConsolidationError(
            f"{raw.get('setting_id')} A8 outcome counts are not the frozen 16/16/16"
        )
    if len(raw.get("formal", [])) != 2:
        raise ConsolidationError("A8 must contain two formal repeats")
    early_stop = raw.get("early_stop", {})
    is_sensitivity = raw["setting"]["chunked_prefill_size"] == 1024
    expected_gate = "ES-W-UNCONDITIONAL" if is_sensitivity else "ES-R0-MDE"
    if early_stop.get("supplement_gate") != expected_gate:
        raise ConsolidationError("A8 supplement-gate evidence mismatch")
    for repeat in raw["formal"]:
        arms = repeat["arms"]
        if set(arms) != {"D0", "E0", "R0"}:
            raise ConsolidationError("A8 formal arms are incomplete")
        canary = arms["R0"].get("same_context_canary")
        if (
            not isinstance(canary, Mapping)
            or canary.get("complete_8_tokens") is not True
            or canary.get("matched") is not True
            or canary.get("engineering_status") != "valid"
        ):
            raise ConsolidationError("A8 same-context canary failed")
        for arm, expected in (
            ("D0", "dense_no_reuse_baseline"),
            ("E0", "exact_gpu_hit"),
            ("R0", "approximate_gpu_recovery"),
        ):
            targets = arms[arm].get("targets")
            if not isinstance(targets, list) or len(targets) != 8:
                raise ConsolidationError(f"A8 {arm} does not have eight targets")
            if any(
                target.get("outcome") != expected
                or target.get("expected_outcome") is not True
                for target in targets
            ):
                raise ConsolidationError(f"A8 {arm} target outcome mismatch")


def _validate_w_contract(raw: Mapping[str, Any]) -> None:
    if len(raw.get("formal", [])) != 2:
        raise ConsolidationError("W artifact must contain two formal repeats")
    if len(raw.get("paired_per_repeat", [])) != 2:
        raise ConsolidationError("W artifact lacks per-repeat paired summaries")
    if raw.get("paired_E0_R0", {}).get("pair_count") != 122:
        raise ConsolidationError("W aggregate E0/R0 pair count is not 122")
    for repeat, paired in zip(raw["formal"], raw["paired_per_repeat"]):
        if set(repeat.get("arms", {})) != {"E0", "R0"}:
            raise ConsolidationError("W formal E0/R0 arms are incomplete")
        if paired.get("repeat_index") != repeat.get("repeat_index"):
            raise ConsolidationError("W paired repeat index drifted")
        if paired.get("paired", {}).get("pair_count") != 61:
            raise ConsolidationError("W formal E0/R0 pair count is not 61")
        if any(
            len(repeat["arms"][arm].get("records", [])) != 61 for arm in ("E0", "R0")
        ):
            raise ConsolidationError("W formal arm did not complete 61 requests")


def validate_r4_contract(raw: Mapping[str, Any]) -> None:
    contract = raw.get("performance_contract", {})
    if contract.get("arm_label") != "R4-like-5x":
        raise ConsolidationError("R4 artifact is missing the 5x proxy label")
    if (
        contract.get("claim")
        != "synthetic_footprint_and_victim_diagnostic_only_not_kvcomm"
    ):
        raise ConsolidationError("R4 artifact claim drifted")
    if contract.get("performance_ranking_enabled") is not False:
        raise ConsolidationError("R4 performance ranking must remain disabled")

    policy = raw.get("setting", {}).get("policy")
    if policy not in {"lru", "hierarchical"}:
        raise ConsolidationError("R4 policy is neither S0 nor S4")
    expected_diagnostic = "available" if policy == "lru" else "diagnostic_unavailable"
    expected_status = "valid" if policy == "lru" else "inconclusive"
    if raw.get("status") != expected_status:
        raise ConsolidationError(f"R4 {policy} status mismatch")
    if len(raw.get("formal", [])) != 2:
        raise ConsolidationError("R4 diagnostic must contain two formal repeats")
    expected_kinds = ["canonical_base", "anchor", "delta", "anchor", "delta"]
    for repeat in raw["formal"]:
        arm = repeat.get("arms", {}).get("R4-like-5x")
        if not isinstance(arm, Mapping):
            raise ConsolidationError("R4 formal proxy arm is missing")
        setup = arm.get("setup", {})
        if setup.get("profile") != "r4_like":
            raise ConsolidationError("R4 formal profile label drifted")
        if setup.get("representation_multiplicity") != 5:
            raise ConsolidationError("R4 formal proxy is not 5x")
        if setup.get("representation_kinds") != expected_kinds:
            raise ConsolidationError("R4 formal proxy representation kinds drifted")
        if arm.get("diagnostic_status") != expected_diagnostic:
            raise ConsolidationError(f"R4 {policy} diagnostic semantics mismatch")
        if policy == "lru":
            if setup.get("registration_failed") is not False:
                raise ConsolidationError("R4 S0 unexpectedly failed registration")
            if len(arm.get("records", [])) != 61:
                raise ConsolidationError("R4 S0 did not complete the W trace")
        else:
            if setup.get("registration_failed") is not True:
                raise ConsolidationError(
                    "R4 S4 diagnostic_unavailable lacks registration failure"
                )
            if arm.get("records") != []:
                raise ConsolidationError(
                    "R4 S4 diagnostic_unavailable unexpectedly has requests"
                )


def validate_capacity_terminal_reason_contract(raw: Mapping[str, Any]) -> None:
    outcome = raw.get("outcome", {})
    expected_counts = {
        reason: 0 for reason in outcome.get("exclusive_terminal_reasons", [])
    }
    observed_fallbacks = 0
    for cell in raw.get("cells", []):
        for profile in cell.get("profiles", []):
            for repeat in profile.get("formal", []):
                for replay in repeat.get("replay", []):
                    if replay.get("outcome") != "dense_fallback":
                        continue
                    observed_fallbacks += 1
                    observations = replay.get("terminal_reason_observations")
                    positives = _positive_terminal_reasons(observations)
                    if (
                        not isinstance(observations, Mapping)
                        or observations.get("verification") != "direct"
                        or observations.get("value_unit") != "tokens"
                        or not isinstance(observations.get("raw"), Mapping)
                        or not isinstance(observations.get("mapped"), Mapping)
                        or not isinstance(observations.get("mapped_from"), Mapping)
                        or observations.get("unmapped_raw_reasons")
                        or len(positives) != 1
                    ):
                        raise ConsolidationError(
                            "capacity dense fallback lacks one mapped terminal reason"
                        )
                    reason = next(iter(positives))
                    if (
                        replay.get("terminal_reason") != reason
                        or replay.get("terminal_reason_verification") != "direct"
                        or replay.get("terminal_reason_valid") is not True
                        or reason not in expected_counts
                    ):
                        raise ConsolidationError(
                            "capacity dense fallback reason binding is invalid"
                        )
                    expected_counts[reason] += 1
    expected_fallbacks = int(
        outcome.get("counts", {}).get("approximate_recovery_failed_dense", 0)
    )
    if observed_fallbacks != expected_fallbacks:
        raise ConsolidationError("capacity fallback row/count mismatch")
    if outcome.get("terminal_reason_counts") != expected_counts:
        raise ConsolidationError("capacity terminal-reason count mismatch")


def validate_raw_artifact(
    raw: Mapping[str, Any],
    *,
    path: Path,
    staging_dir: Path,
    manifest: Mapping[str, Any],
    manifest_file_hash: str,
    settings: Mapping[str, Mapping[str, Any]],
    allow_missing_capacity_terminal_reason_correction: bool = False,
) -> dict[str, str]:
    stored_raw_sha = raw.get("raw_sha256")
    if canonical_sha256(raw, "raw_sha256") != stored_raw_sha:
        raise ConsolidationError(f"{path.name} internal raw SHA-256 mismatch")
    if raw.get("manifest_revision") != AUTHORIZED_MANIFEST_REVISION:
        raise ConsolidationError(f"{path.name} manifest revision mismatch")
    if raw.get("preregistered_manifest_sha256") != AUTHORIZED_MANIFEST_SHA256:
        raise ConsolidationError(f"{path.name} authorized manifest binding mismatch")
    if raw.get("manifest_file_sha256") != manifest_file_hash:
        raise ConsolidationError(f"{path.name} manifest file SHA-256 mismatch")

    setting_id = raw.get("setting_id")
    if setting_id not in settings:
        raise ConsolidationError(f"{path.name} references an unknown setting")
    setting = settings[setting_id]
    restart_index = raw.get("restart_index")
    if restart_index not in setting.get("restart_indices", []):
        raise ConsolidationError(f"{path.name} restart is not authorized")
    if raw.get("setting") != setting:
        raise ConsolidationError(f"{path.name} embedded setting drifted")
    runner = raw.get("runner")
    if not isinstance(runner, Mapping):
        raise ConsolidationError(f"{path.name} runner provenance is missing")
    if runner.get("module") != setting.get("runner"):
        raise ConsolidationError(f"{path.name} runner module mismatch")
    if raw.get("phase") != EXPECTED_PHASE_BY_RUNNER.get(setting["runner"]):
        raise ConsolidationError(f"{path.name} runner/phase mismatch")
    runner_entry = next(
        (
            entry
            for entry in manifest.get("runners", {}).values()
            if entry.get("module") == setting.get("runner")
        ),
        None,
    )
    if (
        runner_entry is None
        or runner.get("sha256") != runner_entry.get("sha256")
        or runner.get("path") != runner_entry.get("path")
    ):
        raise ConsolidationError(f"{path.name} runner SHA-256 mismatch")
    if raw.get("status") not in ALLOWED_RAW_STATUSES:
        raise ConsolidationError(
            f"{path.name} has forbidden status {raw.get('status')}"
        )
    if "error" in raw or raw.get("execution_status") in {"invalid", "error"}:
        raise ConsolidationError(f"{path.name} contains execution failure evidence")
    _validate_inactive_counter_assertion(
        raw.get("inactive_counter_assertion"),
        manifest,
        f"{path.name} inactive-counter assertion",
    )
    _validate_execution_provenance(
        raw,
        manifest,
        manifest_file_hash,
        runner_entry,
        path.name,
    )
    validate_reset_invariants(raw)

    fallback_count = int(
        raw.get("outcome", {})
        .get("counts", {})
        .get("approximate_recovery_failed_dense", 0)
    )
    if raw.get("phase") == "Phase7-capacity" and fallback_count > 0:
        try:
            validate_capacity_terminal_reason_contract(raw)
        except ConsolidationError:
            if not (
                allow_missing_capacity_terminal_reason_correction
                and capacity_terminal_reason_correction_required(raw)
            ):
                raise
    if setting_id.startswith("p7-a8-"):
        _validate_a8_contract(raw)
    if setting_id.startswith("p7-w-r0-"):
        _validate_w_contract(raw)
    if setting_id.startswith("p7-w-r4like-"):
        validate_r4_contract(raw)

    expected_log = staging_dir / "logs" / f"{path.stem}.log"
    expected_runtime_log = (
        f"{manifest['artifact_templates']['runtime_staging_root']}/logs/"
        f"{expected_log.name}"
    )
    if raw.get("server_log_path") != expected_runtime_log:
        raise ConsolidationError(f"{path.name} server log path mismatch")
    if not expected_log.is_file():
        raise ConsolidationError(f"{path.name} server log is missing")
    expected_log_sha = file_sha256(expected_log)
    if raw.get("server_log_sha256") != expected_log_sha:
        raise ConsolidationError(f"{path.name} server log SHA-256 mismatch")
    return {
        "raw_sha256": stored_raw_sha,
        "log_sha256": expected_log_sha,
    }


def validate_correction_artifact(
    correction: Mapping[str, Any],
    *,
    path: Path,
    correction_dir: Path,
    original: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_hash: str,
) -> dict[str, Any]:
    stored_raw_sha = correction.get("raw_sha256")
    if (
        not _is_sha256(stored_raw_sha)
        or canonical_sha256(correction, "raw_sha256") != stored_raw_sha
    ):
        raise ConsolidationError(f"{path.name} correction raw SHA-256 mismatch")
    if correction.get("manifest_file_sha256") != manifest_file_hash:
        raise ConsolidationError(f"{path.name} correction manifest file mismatch")
    if (
        correction.get("correction_manifest_file_sha256") != manifest_file_hash
        or correction.get("correction_manifest_sha256")
        != manifest["correction_manifest_sha256"]
        or correction.get("correction_manifest_revision")
        != manifest["correction_manifest_revision"]
        or correction.get("base_manifest_revision")
        != manifest["base_manifest_revision"]
        or correction.get("base_manifest_self_sha256")
        != manifest["base_manifest_self_sha256"]
        or correction.get("base_manifest_design_sha256")
        != manifest["base_manifest_design_sha256"]
    ):
        raise ConsolidationError(f"{path.name} correction manifest provenance mismatch")
    if (
        original.get("setting_id") != CAPACITY_CORRECTION_SETTING_ID
        or original.get("restart_index") != CAPACITY_CORRECTION_RESTART
        or original.get("raw_sha256") != manifest["original_raw_sha256"]
    ):
        raise ConsolidationError("correction does not target the authorized S0 raw")
    setting = _manifest_setting_map(manifest)[original["setting_id"]]
    runner_entry = manifest["capacity_runner"]
    runner = correction.get("runner")
    if (
        not isinstance(runner, Mapping)
        or runner.get("module") != setting["runner"]
        or runner.get("path") != runner_entry["path"]
        or runner.get("sha256") != runner_entry["sha256"]
    ):
        raise ConsolidationError(f"{path.name} correction runner mismatch")
    _validate_inactive_counter_assertion(
        correction.get("inactive_counter_assertion"),
        manifest,
        f"{path.name} correction inactive-counter assertion",
    )
    _validate_correction_execution_provenance(
        correction,
        manifest,
        manifest_file_hash,
        runner_entry,
        path.name,
    )
    validate_reset_invariants(correction)
    reason_counts = validate_capacity_terminal_reason_correction(
        correction,
        original=original,
        manifest=manifest,
    )

    runtime_log = correction.get("server_log_path")
    if not isinstance(runtime_log, str):
        raise ConsolidationError(f"{path.name} correction server log path is missing")
    runtime_log_path = Path(runtime_log)
    runtime_root = Path(
        manifest["correction_artifact_templates"]["runtime_staging_root"]
    )
    if (
        not runtime_log_path.is_absolute()
        or not runtime_log_path.is_relative_to(runtime_root)
        or runtime_log_path.parent != runtime_root / "logs"
    ):
        raise ConsolidationError(
            f"{path.name} correction log is outside the dedicated staging root"
        )
    local_log = correction_dir / "logs" / runtime_log_path.name
    if not local_log.is_file():
        raise ConsolidationError(f"{path.name} correction server log is missing")
    log_sha = file_sha256(local_log)
    if correction.get("server_log_sha256") != log_sha:
        raise ConsolidationError(f"{path.name} correction server log hash mismatch")
    return {
        "raw_sha256": stored_raw_sha,
        "file_sha256": file_sha256(path),
        "log_path": str(local_log.relative_to(correction_dir)),
        "log_sha256": log_sha,
        "terminal_reason_counts": reason_counts,
    }


def load_evidence_correction(
    *,
    correction_dir: Path,
    correction_manifest_path: Path,
    original: Mapping[str, Any],
    manifest: Mapping[str, Any],
    original_file_sha256: str | None = None,
    fallback_central_events: Sequence[Mapping[str, Any]] = (),
    verify_git: bool = True,
) -> dict[str, Any]:
    correction_dir = correction_dir.resolve()
    correction_manifest_path = correction_manifest_path.resolve()
    if not correction_dir.is_dir():
        raise ConsolidationError(
            f"correction directory does not exist: {correction_dir}"
        )
    raw_dir = correction_dir / "raw"
    candidates = sorted(raw_dir.glob("*.json")) if raw_dir.is_dir() else []
    if len(candidates) != 1:
        raise ConsolidationError(
            "correction directory must contain exactly one S0 supplementary "
            f"raw JSON, got "
            f"{len(candidates)}"
        )
    path = candidates[0]
    correction = json.loads(path.read_text(encoding="utf-8"))
    if not correction_manifest_path.is_file():
        raise ConsolidationError(
            f"correction manifest does not exist: {correction_manifest_path}"
        )
    correction_manifest = json.loads(
        correction_manifest_path.read_text(encoding="utf-8")
    )
    validate_correction_manifest(
        correction_manifest,
        primary_manifest=manifest,
        manifest_path=correction_manifest_path,
        repo_root=correction_manifest_path.parents[4],
        verify_git=verify_git,
    )
    if (
        original_file_sha256 is not None
        and original_file_sha256 != correction_manifest["original_raw_file_sha256"]
    ):
        raise ConsolidationError(
            "correction manifest original raw file SHA-256 mismatch"
        )
    correction_manifest_file_hash = file_sha256(correction_manifest_path)
    review_path = (
        correction_manifest_path.parents[4]
        / correction_manifest["review_evidence"]["artifact_path"]
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    try:
        validate_correction_review_binding(
            review,
            activating_manifest=correction_manifest,
        )
    except ValueError as error:
        raise ConsolidationError(
            f"correction review binding is invalid: {error}"
        ) from error
    if (
        file_sha256(review_path)
        != correction_manifest["review_evidence"]["file_sha256"]
    ):
        raise ConsolidationError("correction review file SHA-256 mismatch")
    hashes = validate_correction_artifact(
        correction,
        path=path,
        correction_dir=correction_dir,
        original=original,
        manifest=correction_manifest,
        manifest_file_hash=correction_manifest_file_hash,
    )
    central_path = correction_dir / "phase7-runs.jsonl"
    if central_path.is_file():
        events = parse_central_events(
            central_path.read_text(encoding="utf-8").splitlines()
        )
    else:
        events = list(fallback_central_events)
    if any(event.get("correction") != correction["correction"] for event in events):
        raise ConsolidationError(
            "correction central log contains a non-correction or extra run"
        )
    durations = central_run_durations(
        events,
        {(correction["setting_id"], correction["restart_index"])},
        excluded_run_classes=("primary execution runs",),
        expected_manifest_sha256=correction_manifest["correction_manifest_sha256"],
    )
    central_row = durations["runs"][0]
    if (
        central_row["raw_sha256"] != correction["raw_sha256"]
        or central_row["run_id"] != correction.get("run_id")
        or central_row["phase"] != correction.get("phase")
    ):
        raise ConsolidationError("correction central/raw binding mismatch")
    return {
        "artifact": correction,
        "manifest": correction_manifest,
        "source": {
            "path": str(path.relative_to(correction_dir)),
            "manifest_path": CAPACITY_CORRECTION_MANIFEST_PATH,
            "manifest_file_sha256": correction_manifest_file_hash,
            "review_path": CAPACITY_CORRECTION_REVIEW_PATH,
            "review_file_sha256": file_sha256(review_path),
            "cpu_evidence_path": correction_manifest["capacity_cpu_evidence"]["path"],
            "cpu_evidence_file_sha256": correction_manifest["capacity_cpu_evidence"][
                "file_sha256"
            ],
            **hashes,
        },
        "elapsed": durations,
    }


def require_capacity_terminal_reason_correction(
    *,
    original: Mapping[str, Any],
    correction_dir: Path | None,
) -> None:
    if (
        capacity_terminal_reason_correction_required(original)
        and correction_dir is None
    ):
        raise ConsolidationError(
            "engineering VALID requires --correction-dir for S0 wave0 "
            "terminal reasons"
        )


def parse_central_events(lines: Sequence[str]) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConsolidationError(
                f"central JSONL line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(event, dict):
            raise ConsolidationError(
                f"central JSONL line {line_number} is not an object"
            )
        events.append(event)
    return events


def central_run_durations(
    events: Sequence[Mapping[str, Any]],
    expected_runs: set[tuple[str, int]],
    *,
    excluded_run_classes: Sequence[str] = (),
    expected_manifest_sha256: str = AUTHORIZED_MANIFEST_SHA256,
) -> dict[str, Any]:
    by_run_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        status = event.get("status")
        if status == "failed":
            raise ConsolidationError("central JSONL contains a failed event")
        if status not in {"running", "completed"}:
            raise ConsolidationError(f"central JSONL has unknown status {status}")
        run_id = event.get("run_id")
        if not isinstance(run_id, str):
            raise ConsolidationError("central JSONL event is missing run_id")
        by_run_id[run_id].append(event)

    seen_pairs: set[tuple[str, int]] = set()
    rows = []
    for run_id, run_events in by_run_id.items():
        statuses = Counter(event["status"] for event in run_events)
        if statuses != Counter({"running": 1, "completed": 1}):
            raise ConsolidationError(
                f"central JSONL run {run_id} must have one running and one completed"
            )
        running = next(event for event in run_events if event["status"] == "running")
        completed = next(
            event for event in run_events if event["status"] == "completed"
        )
        pair = (running.get("setting_id"), running.get("restart_index"))
        completed_pair = (
            completed.get("setting_id"),
            completed.get("restart_index"),
        )
        if pair != completed_pair or pair not in expected_runs:
            raise ConsolidationError(
                f"central JSONL run {run_id} setting/restart mismatch"
            )
        if pair in seen_pairs:
            raise ConsolidationError(f"central JSONL duplicates setting/restart {pair}")
        seen_pairs.add(pair)
        if running.get("manifest_sha256") != expected_manifest_sha256:
            raise ConsolidationError(
                f"central JSONL run {run_id} manifest binding mismatch"
            )
        if not _is_sha256(completed.get("raw_sha256")):
            raise ConsolidationError(
                f"central JSONL run {run_id} completion lacks raw SHA-256"
            )
        expected_phase = (
            "Phase7-capacity"
            if pair[0].startswith("p6delta-")
            else (
                "Phase7-ceiling" if pair[0].startswith("p7-a8-") else "Phase7-scheduler"
            )
        )
        if (
            running.get("phase") != expected_phase
            or completed.get("phase") != expected_phase
        ):
            raise ConsolidationError(f"central JSONL run {run_id} phase mismatch")
        if running.get("correction") != completed.get("correction"):
            raise ConsolidationError(
                f"central JSONL run {run_id} correction binding mismatch"
            )
        try:
            started = datetime.fromisoformat(running["timestamp"])
            ended = datetime.fromisoformat(completed["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConsolidationError(
                f"central JSONL run {run_id} has invalid timestamps"
            ) from exc
        if started.utcoffset() is None or ended.utcoffset() is None:
            raise ConsolidationError(
                f"central JSONL run {run_id} timestamps lack timezone offsets"
            )
        elapsed_seconds = (ended - started).total_seconds()
        if elapsed_seconds < 0:
            raise ConsolidationError(
                f"central JSONL run {run_id} has negative duration"
            )
        rows.append(
            {
                "run_id": run_id,
                "phase": expected_phase,
                "setting_id": pair[0],
                "restart_index": pair[1],
                "started_at": running["timestamp"],
                "completed_at": completed["timestamp"],
                "elapsed_seconds": elapsed_seconds,
                "raw_sha256": completed["raw_sha256"],
                "output": completed.get("output") or running.get("output"),
                **(
                    {"correction": running["correction"]}
                    if "correction" in running
                    else {}
                ),
            }
        )
    if seen_pairs != expected_runs:
        missing = sorted(expected_runs - seen_pairs)
        extra = sorted(seen_pairs - expected_runs)
        raise ConsolidationError(
            f"central JSONL execution set mismatch: missing={missing}, extra={extra}"
        )
    rows.sort(key=lambda row: (row["started_at"], row["setting_id"]))
    total_seconds = sum(row["elapsed_seconds"] for row in rows)
    first_started = min(datetime.fromisoformat(row["started_at"]) for row in rows)
    last_completed = max(datetime.fromisoformat(row["completed_at"]) for row in rows)
    wall_clock_span_seconds = (last_completed - first_started).total_seconds()
    excluded = ["inter-run gaps", *excluded_run_classes]
    return {
        "runs": rows,
        "total_elapsed_seconds": total_seconds,
        "total_elapsed_gpu_equivalent_hours": total_seconds / 3600.0,
        "wall_clock_span_seconds": wall_clock_span_seconds,
        "wall_clock_span_hours": wall_clock_span_seconds / 3600.0,
        "sum_of_run_intervals_exclusion_note": (
            f"sum of run intervals excludes {' and '.join(excluded)}; "
            "wall_clock_span_hours includes inter-run gaps"
        ),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ConsolidationError("cannot summarize an empty numeric view")
    return {
        "n": len(values),
        "min": min(values),
        "p50": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _a8_primary_views(raw: Mapping[str, Any]) -> dict[str, Any]:
    request_path_ratios = []
    request_path_deltas = []
    target_only_ratios = []
    target_only_deltas = []
    full_lifecycle_ratios = []
    full_lifecycle_deltas = []
    cached_tokens: dict[str, list[float]] = {arm: [] for arm in ("D0", "E0", "R0")}
    for repeat in raw["formal"]:
        arms = repeat["arms"]
        dense_targets = arms["D0"]["targets"]
        recovery_targets = arms["R0"]["targets"]
        for dense, recovery in zip(dense_targets, recovery_targets):
            dense_request_ms = float(dense["request_path_ms"])
            recovery_request_ms = float(recovery["request_path_ms"])
            request_path_ratios.append(dense_request_ms / recovery_request_ms)
            request_path_deltas.append(recovery_request_ms - dense_request_ms)
            dense_ms = float(dense["target_only_ms"])
            recovery_ms = float(recovery["target_only_ms"])
            target_only_ratios.append(dense_ms / recovery_ms)
            target_only_deltas.append(recovery_ms - dense_ms)
        dense_lifecycle = float(arms["D0"]["ledger"]["full_lifecycle_ms"])
        recovery_lifecycle = float(arms["R0"]["ledger"]["full_lifecycle_ms"])
        full_lifecycle_ratios.append(dense_lifecycle / recovery_lifecycle)
        full_lifecycle_deltas.append(recovery_lifecycle - dense_lifecycle)
        for arm in cached_tokens:
            cached_tokens[arm].extend(
                float(target["cached_tokens"]) for target in arms[arm]["targets"]
            )
    cold_start_ms = float(raw["ledger"]["setup"]["server_cold_start_ms"])
    return {
        "request_path": {
            "paired_dense_over_recovery_ratio": _numeric_summary(
                request_path_ratios
            ),
            "paired_recovery_minus_dense_ms": _numeric_summary(
                request_path_deltas
            ),
            "definition": "seed_head_ms + target_only_ms",
            "is_preregistered_mde_metric": True,
        },
        "target_only": {
            "paired_dense_over_recovery_ratio": _numeric_summary(target_only_ratios),
            "paired_recovery_minus_dense_ms": _numeric_summary(target_only_deltas),
            "excludes": "seed_head_ms",
            "is_preregistered_mde_metric": False,
        },
        "full_lifecycle": {
            "dense_over_recovery_ratio": _numeric_summary(full_lifecycle_ratios),
            "recovery_minus_dense_ms": _numeric_summary(full_lifecycle_deltas),
            "unit": "formal_repeat",
        },
        "cached_tokens": {
            arm: _numeric_summary(values) for arm, values in cached_tokens.items()
        },
        "cold_start": {
            "server_cold_start_ms": cold_start_ms,
            "shared_by_all_arms": True,
            "included_in_arm_latency_ratios": False,
        },
    }


def aggregate_a8(raws: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        raws,
        key=lambda raw: (
            int(raw["setting"]["body_tokens"]),
            float(raw["setting"]["rho_logical_demand"]),
        ),
    )
    table = []
    request_medians = []
    for raw in ordered:
        if raw.get("restart_index") != 0:
            raise ConsolidationError("A8 primary aggregation includes a supplement")
        summary = raw.get("summary", {})
        request_median = summary.get("paired_target_request_path_median_speedup")
        amortization = summary.get("amortization_median_speedup")
        if not isinstance(request_median, (int, float)) or not isinstance(
            amortization, Mapping
        ):
            raise ConsolidationError("A8 summary metrics are missing")
        canaries = [
            repeat["arms"]["R0"]["same_context_canary"] for repeat in raw["formal"]
        ]
        canary_rows = []
        for canary in canaries:
            distinct_output_tokens = len(
                set(canary["dense_output_ids"]) | set(canary["recovery_output_ids"])
            )
            canary_rows.append(
                {
                    "complete_8_tokens": canary["complete_8_tokens"],
                    "matched": canary["matched"],
                    "engineering_status": canary["engineering_status"],
                    "distinct_output_tokens": distinct_output_tokens,
                    "body_tokens": raw["setting"]["body_tokens"],
                    "discriminative_power": (
                        "limited"
                        if distinct_output_tokens <= 1
                        or int(raw["setting"]["body_tokens"]) == 1024
                        else "bounded"
                    ),
                }
            )
        row = {
            "setting_id": raw["setting_id"],
            "body_tokens": raw["setting"]["body_tokens"],
            "rho_logical_demand": raw["setting"]["rho_logical_demand"],
            "chunked_prefill_size": raw["setting"]["chunked_prefill_size"],
            "restart_index": raw["restart_index"],
            "request_path_median_speedup": request_median,
            "amortization_median_speedup": {
                str(n): {
                    "full_setup": amortization[str(n)]["full_setup"],
                    "incremental_setup": amortization[str(n)]["incremental_setup"],
                }
                for n in (1, 2, 4, 8)
            },
            "break_even": {
                "full_setup": [
                    repeat["amortization"]["full_setup_break_even_observed_N"]
                    for repeat in raw["formal"]
                ],
                "incremental_setup": [
                    repeat["amortization"]["incremental_setup_break_even_observed_N"]
                    for repeat in raw["formal"]
                ],
            },
            "canary": {
                "repeat_values": canary_rows,
                "passed": all(canary["matched"] for canary in canaries),
                "limitation": (
                    "distinct_output_tokens and body1024 canaries have limited "
                    "discriminative power"
                ),
            },
            "primary_views": _a8_primary_views(raw),
            "reset_passed": True,
            "outcome_counts": raw["outcome"]["counts"],
            "outcome": "NEGATIVE",
        }
        request_medians.append(float(request_median))
        table.append(row)
    if len(table) != 4 or any(value >= 1.05 for value in request_medians):
        raise ConsolidationError("A8 restart-0 matrix does not support NEGATIVE")
    return {
        "mechanism_status": MECHANISM_STATUS_NEGATIVE,
        "mde_fraction": 0.05,
        "independent_replicate_unit": "server_restart",
        "independent_restarts_per_setting": 1,
        "n_per_setting": 1,
        "formal_repeats_are_not_independent_replicates": True,
        "targets_are_not_independent_replicates": True,
        "three_restart_range": {
            "available": False,
            "reason": "ES-R0-MDE",
        },
        "primary_supplement_disposition": {
            "rule": "ES-R0-MDE",
            "skipped_starts": 8,
            "reason": "restart-0 request-path medians did not reach the 5% MDE",
        },
        "table": table,
        "headline_speedup_allowed": False,
    }


def aggregate_chunk1024_sensitivity(
    raws: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(raws, key=lambda raw: raw["restart_index"])
    if [raw["restart_index"] for raw in ordered] != [0, 1]:
        raise ConsolidationError("chunk1024 sensitivity requires restarts 0 and 1")
    restart_values = [
        {
            "restart_index": raw["restart_index"],
            "request_path_median_speedup": raw["summary"][
                "paired_target_request_path_median_speedup"
            ],
            "amortization_median_speedup": raw["summary"][
                "amortization_median_speedup"
            ],
        }
        for raw in ordered
    ]
    return {
        "chunked_prefill_size": 1024,
        "restart_values": restart_values,
        "median_request_path_speedup": statistics.median(
            row["request_path_median_speedup"] for row in restart_values
        ),
        "amortization_median_across_restarts": {
            str(n): {
                "full_setup": statistics.median(
                    row["amortization_median_speedup"][str(n)]["full_setup"]
                    for row in restart_values
                ),
                "incremental_setup": statistics.median(
                    row["amortization_median_speedup"][str(n)]["incremental_setup"]
                    for row in restart_values
                ),
            }
            for n in (1, 2, 4, 8)
        },
        "interpretation": (
            "chunk-coupled sensitivity diagnostic; not a "
            "mechanism-intrinsic headline"
        ),
        "headline": False,
    }


def _paired_denominator_view(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    ratio_of_marginal_p95s = row["p95_ratio"]
    return {
        "e0": row["e0"],
        "r0": row["r0"],
        "e0_over_r0_mean_speedup": row["mean_speedup"],
        "e0_over_r0_p50_speedup": row["p50_speedup"],
        "e0_over_r0_wall_speedup": row["wall_clock_speedup"],
        "ratio_of_marginal_p95s": ratio_of_marginal_p95s,
        "ratio_of_marginal_p95s_direction": "r0_over_e0",
        "p95_pairing": "nonpaired",
        "paired_delta_median_ms": row["paired_delta_median_ms"],
        "per_role": row["per_role"],
    }


def _restart_w_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    policy = raw["setting"]["policy"]
    rho = float(raw["setting"]["rho_logical_demand"])
    result: dict[str, Any] = {
        "setting_id": raw["setting_id"],
        "policy": policy,
        "rho_logical_demand": rho,
        "restart_index": raw["restart_index"],
        "formal_repeats": [],
        "r0_outcomes": _record_outcome_composition(
            [
                record
                for repeat in raw["formal"]
                for record in repeat["arms"]["R0"]["records"]
            ],
            outcome_taxonomy=raw["outcome"]["taxonomy"],
            terminal_reasons=raw["outcome"]["exclusive_terminal_reasons"],
        ),
    }
    for repeat, paired_row in zip(raw["formal"], raw["paired_per_repeat"]):
        if repeat["repeat_index"] != paired_row["repeat_index"]:
            raise ConsolidationError("W repeat pairing drifted")
        repeat_row: dict[str, Any] = {
            "repeat_index": repeat["repeat_index"],
            "denominators": {},
        }
        for denominator in W_DENOMINATORS:
            e0 = repeat["arms"]["E0"]["statistics"]["denominators"][denominator]
            r0 = repeat["arms"]["R0"]["statistics"]["denominators"][denominator]
            paired = paired_row["paired"]["denominators"][denominator]
            repeat_row["denominators"][denominator] = {
                **_paired_denominator_view(paired),
                "e0": e0,
                "r0": r0,
            }
        peaks = raw["rho"]["arm_interval_peak_by_repeat_arm"]["values"][
            str(repeat["repeat_index"])
        ]
        repeat_row["peak_device_bytes"] = peaks
        repeat_row["r0_over_e0_peak_ratio"] = peaks["R0"] / peaks["E0"]
        result["formal_repeats"].append(repeat_row)

    aggregate: dict[str, Any] = {"denominators": {}}
    for denominator in W_DENOMINATORS:
        e0 = raw["arm_statistics"]["E0"]["denominators"][denominator]
        r0 = raw["arm_statistics"]["R0"]["denominators"][denominator]
        paired = raw["paired_E0_R0"]["denominators"][denominator]
        aggregate["denominators"][denominator] = {
            **_paired_denominator_view(paired),
            "e0": e0,
            "r0": r0,
        }
    aggregate["r0_peak_device_bytes"] = statistics.median(
        repeat["peak_device_bytes"]["R0"] for repeat in result["formal_repeats"]
    )
    aggregate["e0_peak_device_bytes"] = statistics.median(
        repeat["peak_device_bytes"]["E0"] for repeat in result["formal_repeats"]
    )
    aggregate["r0_over_e0_peak_ratio"] = (
        aggregate["r0_peak_device_bytes"] / aggregate["e0_peak_device_bytes"]
    )
    result["aggregate_two_formals"] = aggregate
    return result


def aggregate_w_cross_policy(
    restart_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    indexed = {
        (
            float(row["rho_logical_demand"]),
            row["policy"],
            int(row["restart_index"]),
        ): row
        for row in restart_rows
    }
    expected_keys = {
        (rho, policy, restart)
        for rho in (1.5, 2.0)
        for policy in ("lru", "hierarchical")
        for restart in (0, 1, 2)
    }
    if len(indexed) != len(restart_rows):
        raise ConsolidationError("W aggregation contains duplicate starts")
    if set(indexed) != expected_keys:
        raise ConsolidationError("W aggregation requires 12 independent starts")
    result = {}
    for rho in (1.5, 2.0):
        seed_matched_restarts = []
        for restart in (0, 1, 2):
            s0 = indexed[(rho, "lru", restart)]["aggregate_two_formals"]
            s4 = indexed[(rho, "hierarchical", restart)]["aggregate_two_formals"]
            row: dict[str, Any] = {
                "restart_index": restart,
                "comparison_design": ("seed-matched_non_adjacent_restart_comparison"),
                "all_reusable": {},
                "workflow_only": {},
                "miss_delta_s4_minus_s0": {},
                "peak_ratio_s4_over_s0": (
                    s4["r0_peak_device_bytes"] / s0["r0_peak_device_bytes"]
                ),
            }
            for denominator in W_DENOMINATORS:
                s0_r0 = s0["denominators"][denominator]["r0"]
                s4_r0 = s4["denominators"][denominator]["r0"]
                row[denominator] = {
                    "mean_speedup_s0_over_s4": (
                        s0_r0["ttft_mean_ms"] / s4_r0["ttft_mean_ms"]
                    ),
                    "wall_speedup_s0_over_s4": (
                        s0_r0["wall_clock_ms"] / s4_r0["wall_clock_ms"]
                    ),
                    "ratio_of_marginal_p95s": (
                        s4_r0["ttft_p95_ms"] / s0_r0["ttft_p95_ms"]
                    ),
                    "ratio_of_marginal_p95s_direction": "s4_over_s0",
                    "p95_pairing": "nonpaired",
                    "s0_r0": s0_r0,
                    "s4_r0": s4_r0,
                }
                row["miss_delta_s4_minus_s0"][denominator] = (
                    s4_r0["partial_or_full_miss_requests"]
                    - s0_r0["partial_or_full_miss_requests"]
                )
            seed_matched_restarts.append(row)

        aggregate = {}
        for denominator in W_DENOMINATORS:
            aggregate[denominator] = {
                field: statistics.median(
                    row[denominator][field] for row in seed_matched_restarts
                )
                for field in (
                    "mean_speedup_s0_over_s4",
                    "wall_speedup_s0_over_s4",
                    "ratio_of_marginal_p95s",
                )
            }
            aggregate[denominator].update(
                {
                    "ratio_of_marginal_p95s_direction": "s4_over_s0",
                    "p95_pairing": "nonpaired",
                }
            )
            for policy_label in ("s0_r0", "s4_r0"):
                aggregate[denominator][f"{policy_label}_median"] = {
                    field: statistics.median(
                        row[denominator][policy_label][field]
                        for row in seed_matched_restarts
                    )
                    for field in (
                        "ttft_mean_ms",
                        "wall_clock_ms",
                        "ttft_p50_ms",
                        "ttft_p95_ms",
                        "partial_or_full_miss_requests",
                    )
                }
        aggregate["miss_delta_s4_minus_s0"] = {
            denominator: statistics.median(
                row["miss_delta_s4_minus_s0"][denominator]
                for row in seed_matched_restarts
            )
            for denominator in W_DENOMINATORS
        }
        aggregate["s0_r0_peak_device_bytes_median"] = statistics.median(
            indexed[(rho, "lru", restart)]["aggregate_two_formals"][
                "r0_peak_device_bytes"
            ]
            for restart in (0, 1, 2)
        )
        aggregate["s4_r0_peak_device_bytes_median"] = statistics.median(
            indexed[(rho, "hierarchical", restart)]["aggregate_two_formals"][
                "r0_peak_device_bytes"
            ]
            for restart in (0, 1, 2)
        )
        aggregate["peak_ratio_s4_over_s0"] = statistics.median(
            row["peak_ratio_s4_over_s0"] for row in seed_matched_restarts
        )
        result[str(rho)] = {
            "independent_replicate_unit": "server_restart",
            "request_rows_are_not_independent_replicates": True,
            "comparison_design": "seed-matched_non_adjacent_restart_comparison",
            "not_a_paired_launch_block": True,
            "per_restart": seed_matched_restarts,
            "median_across_restarts": aggregate,
        }
    return result


def aggregate_within_policy_latency(
    restart_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in restart_rows:
        for denominator in W_DENOMINATORS:
            grouped[
                (
                    row["policy"],
                    float(row["rho_logical_demand"]),
                    denominator,
                )
            ].append(row["aggregate_two_formals"]["denominators"][denominator])
    result: dict[str, Any] = {}
    for (policy, rho, denominator), rows in sorted(grouped.items()):
        mean_ratios = [1.0 / float(row["e0_over_r0_mean_speedup"]) for row in rows]
        p50_ratios = [1.0 / float(row["e0_over_r0_p50_speedup"]) for row in rows]
        p95_ratios = [float(row["ratio_of_marginal_p95s"]) for row in rows]
        paired_deltas = [float(row["paired_delta_median_ms"]) for row in rows]
        per_role: dict[str, Any] = {}
        roles = sorted({role for row in rows for role in row["per_role"]})
        for role in roles:
            role_rows = [row["per_role"][role] for row in rows]
            per_role[role] = {
                "speedup_median_across_restarts": statistics.median(
                    role_row["speedup_median"] for role_row in role_rows
                ),
                "paired_delta_median_ms_across_restarts": statistics.median(
                    role_row["paired_delta_median_ms"] for role_row in role_rows
                ),
                "e0_misses_median": statistics.median(
                    role_row["e0_misses"] for role_row in role_rows
                ),
                "r0_misses_median": statistics.median(
                    role_row["r0_misses"] for role_row in role_rows
                ),
            }
        result.setdefault(policy, {}).setdefault(str(rho), {})[denominator] = {
            "r0_over_e0_mean_latency_ratio": _numeric_summary(mean_ratios),
            "r0_over_e0_p50_latency_ratio": _numeric_summary(p50_ratios),
            "ratio_of_marginal_p95s": _numeric_summary(p95_ratios),
            "ratio_of_marginal_p95s_direction": "r0_over_e0",
            "p95_pairing": "nonpaired",
            "paired_delta_median_ms": _numeric_summary(paired_deltas),
            "per_role": per_role,
            "independent_replicate_unit": "server_restart",
            "interpretation": "R0 is slower than request-paired E0",
        }
    return result


def _aggregate_victim_classes(
    arms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    totals: Counter[tuple[str, str, str]] = Counter()
    for arm in arms:
        for row in arm["metrics"]["victim_evict_bytes"]["rows"]:
            key = (
                str(row["requester"]),
                str(row["provenance"]),
                str(row["object_kind"]),
            )
            totals[key] += float(row["bytes_or_count"])
    return [
        {
            "requester": requester,
            "provenance": provenance,
            "object_kind": object_kind,
            "bytes": value,
        }
        for (requester, provenance, object_kind), value in sorted(totals.items())
    ]


def _optional_metric_summary(
    arms: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    rows = [arm["metrics"][metric] for arm in arms]
    values = [
        float(row["value"])
        for row in rows
        if isinstance(row.get("value"), (int, float))
        and not isinstance(row.get("value"), bool)
    ]
    return {
        "values": [row.get("value") for row in rows],
        "sum": sum(values) if len(values) == len(rows) else None,
        "verifications": sorted({str(row.get("verification")) for row in rows}),
        "definitions": sorted(
            {str(row.get("definition")) for row in rows if row.get("definition")}
        ),
    }


def _arm_victim_footprint(
    repeat_arms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    peaks = [
        float(arm["metrics"]["arm_interval_peak_device_bytes"]) for arm in repeat_arms
    ]
    memory_rows = [arm["memory_footprint_after"] for arm in repeat_arms]
    numeric_memory_fields = sorted(
        {
            key
            for row in memory_rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    victim_class_counts = Counter(
        (
            str(event["requester"]),
            str(event["provenance"]),
            str(event["object_kind"]),
        )
        for arm in repeat_arms
        for event in arm["victim_sequence"]
    )
    return {
        "victim_evict_bytes_by_requester_provenance_object_kind": (
            _aggregate_victim_classes(repeat_arms)
        ),
        "victim_sequence_event_count": sum(
            len(arm["victim_sequence"]) for arm in repeat_arms
        ),
        "victim_sequence_class_counts": [
            {
                "requester": requester,
                "provenance": provenance,
                "object_kind": object_kind,
                "events": count,
            }
            for (requester, provenance, object_kind), count in sorted(
                victim_class_counts.items()
            )
        ],
        "wasted_bytes": _optional_metric_summary(repeat_arms, "wasted_bytes"),
        "churn_bytes": _optional_metric_summary(repeat_arms, "churn_bytes"),
        "peak_device_bytes": {
            "per_repeat": peaks,
            "max": max(peaks),
            "median": statistics.median(peaks),
            "semantics": repeat_arms[0]["metrics"]["peak_semantics"],
        },
        "memory_footprint_after": {
            "per_repeat": memory_rows,
            "max_by_field": {
                field: max(float(row[field]) for row in memory_rows)
                for field in numeric_memory_fields
            },
        },
    }


def summarize_w_victim_footprint(
    raws: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for raw in raws:
        rows.append(
            {
                "setting_id": raw["setting_id"],
                "policy": raw["setting"]["policy"],
                "rho_logical_demand": float(raw["setting"]["rho_logical_demand"]),
                "restart_index": int(raw["restart_index"]),
                "arms": {
                    arm: _arm_victim_footprint(
                        [repeat["arms"][arm] for repeat in raw["formal"]]
                    )
                    for arm in ("E0", "R0")
                },
            }
        )
    return {
        "primary_axis_after_a8_negative": True,
        "primary_axis_reason": (
            "A8 is NEGATIVE, so scheduler victim/accounting and memory footprint "
            "are primary; latency ratios are secondary descriptive evidence"
        ),
        "aggregation_unit": "policy/rho/server_restart with formal repeats retained",
        "rows": sorted(
            rows,
            key=lambda row: (
                row["rho_logical_demand"],
                row["policy"],
                row["restart_index"],
            ),
        ),
    }


def summarize_wave0(
    raws: Sequence[Mapping[str, Any]],
    *,
    correction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {}
    for raw in raws:
        policy = raw["setting"]["policy"]
        cell = raw["cells"][0]
        profiles = {
            profile["profile"]: {
                "reachability": normalized_diagnostic(profile["reachability"]),
                "valid": profile["valid"],
                "representation_kinds": profile["representation_kinds"],
            }
            for profile in cell["profiles"]
        }
        profile_outcomes = {}
        for profile in cell["profiles"]:
            counts: Counter[str] = Counter()
            for repeat in profile["formal"]:
                counts.update(
                    {
                        outcome: int(value)
                        for outcome, value in repeat["cache_outcomes"].items()
                    }
                )
            profile_outcomes[profile["profile"]] = dict(counts)
        approximate_profiles = [
            name for name in profile_outcomes if name != "exact_only"
        ]
        approximate_counts = {
            outcome: sum(
                profile_outcomes[name].get(outcome, 0) for name in approximate_profiles
            )
            for outcome in (
                "approximate_gpu_recovery",
                "dense_fallback",
                "exact_gpu_hit",
                "host_demand_load",
                "unknown",
            )
        }
        approximate_requests = sum(approximate_counts.values())
        policy_label = "S4" if policy == "hierarchical" else "S0"
        correction_reason_counts = None
        artifact_status = "accepted"
        if policy_label == "S0" and approximate_counts["dense_fallback"] > 0:
            original_reason_counts = raw["outcome"].get("terminal_reason_counts", {})
            if not any(original_reason_counts.values()):
                if correction is None:
                    raise ConsolidationError(
                        "wave0 S0 dense fallback requires terminal-reason correction"
                    )
                correction_reason_counts = correction["outcome"][
                    "terminal_reason_counts"
                ]
                artifact_status = "accepted_with_evidence_correction"
        result[policy_label] = {
            "setting_id": raw["setting_id"],
            "artifact_status": artifact_status,
            "raw_status": normalized_diagnostic(raw["status"]),
            "cell_status": normalized_diagnostic(cell["status"]),
            "requested_capacity_tokens": raw["requested_capacity"]["tokens"][0],
            "observed_capacity_tokens": raw["observed_capacity"]["tokens"][0],
            "profile_registration_reachability": profiles,
            "registration_reachability_is_not_recovery_success": True,
            "formal_outcomes": {
                "by_profile": profile_outcomes,
                "approximate_profiles": approximate_profiles,
                "approximate_replay_requests": approximate_requests,
                **approximate_counts,
                "terminal_reason_counts": (
                    correction_reason_counts
                    if correction_reason_counts is not None
                    else raw["outcome"].get("terminal_reason_counts", {})
                ),
            },
            "fallback_reachability": {
                **raw["fallback_reachability"],
                "semantics": (
                    "reservation-failure-specific reachability; distinct from "
                    "observed dense-fallback outcomes"
                ),
            },
            "status_reason": (
                (
                    f"{approximate_counts['approximate_gpu_recovery']}/"
                    f"{approximate_requests} approximate replays recovered; "
                    "R4-like registration reachability is diagnostic_unavailable"
                )
                if policy_label == "S4"
                else (
                    f"{approximate_counts['approximate_gpu_recovery']}/"
                    f"{approximate_requests} approximate replays recovered and "
                    f"{approximate_counts['dense_fallback']}/"
                    f"{approximate_requests} used dense fallback; direct terminal "
                    "reasons come from the bound evidence correction"
                )
            ),
            "r2_like_semantics": (
                "2x synthetic representation multiplicity footprint only; "
                "not R2 execution"
            ),
        }
    if set(result) != {"S0", "S4"}:
        raise ConsolidationError("wave0 requires one S0 and one S4 artifact")
    s4 = result["S4"]["profile_registration_reachability"]
    if (
        any(
            s4[name]["reachability"] != "reachable"
            for name in ("exact_only", "r0_like", "r1_like_k32", "r2_like")
        )
        or s4["r4_like"]["reachability"] != "diagnostic_unavailable"
    ):
        raise ConsolidationError("wave0 S4 reachability semantics mismatch")
    if any(
        profile["reachability"] != "reachable"
        for profile in result["S0"]["profile_registration_reachability"].values()
    ):
        raise ConsolidationError("wave0 S0 must have five reachable profiles")
    s0_outcomes = result["S0"]["formal_outcomes"]
    s4_outcomes = result["S4"]["formal_outcomes"]
    if (
        s0_outcomes["approximate_replay_requests"] != 40
        or s0_outcomes["approximate_gpu_recovery"] != 0
        or s0_outcomes["dense_fallback"] != 40
    ):
        raise ConsolidationError("wave0 S0 formal outcome composition drifted")
    if (
        s4_outcomes["approximate_replay_requests"] != 40
        or s4_outcomes["approximate_gpu_recovery"] != 40
        or s4_outcomes["dense_fallback"] != 0
    ):
        raise ConsolidationError("wave0 S4 formal outcome composition drifted")
    return result


def summarize_r4(raws: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for raw in raws:
        policy = raw["setting"]["policy"]
        arms = [repeat["arms"]["R4-like-5x"] for repeat in raw["formal"]]
        policy_label = "S0" if policy == "lru" else "S4"
        records = [record for arm in arms for record in arm["records"]]
        composition = _record_outcome_composition(
            records,
            outcome_taxonomy=raw["outcome"]["taxonomy"],
            terminal_reasons=raw["outcome"]["exclusive_terminal_reasons"],
        )
        if composition["counts"] != raw["outcome"]["counts"]:
            raise ConsolidationError(f"R4 {policy_label} outcome counts drifted")
        result[policy_label] = {
            "status": raw["status"],
            "diagnostic_statuses": [
                normalized_diagnostic(arm["diagnostic_status"]) for arm in arms
            ],
            "representation_multiplicity": 5,
            "representation_kinds": arms[0]["setup"]["representation_kinds"],
            "representation_metadata": [arm["representation_metadata"] for arm in arms],
            "registration_failed": [
                arm["setup"]["registration_failed"] for arm in arms
            ],
            "completed_request_records": [len(arm["records"]) for arm in arms],
            "outcomes": composition,
            "outcome_counts": raw["outcome"]["counts"],
            "terminal_reason_counts": raw["outcome"]["terminal_reason_counts"],
            "recovery_success_fraction": composition["recovery"],
            "fallback_fraction": composition["dense_fallback"],
            "peak_device_bytes": [
                arm["metrics"]["arm_interval_peak_device_bytes"] for arm in arms
            ],
            "memory_footprint_after": [arm["memory_footprint_after"] for arm in arms],
            "victim_sequence": [
                {
                    "repeat_index": index,
                    "events": arm["victim_sequence"],
                }
                for index, arm in enumerate(arms)
            ],
            "victim_class_accounting": {
                "victim_evict_bytes_by_requester_provenance_object_kind": (
                    _aggregate_victim_classes(arms)
                ),
                "wasted_bytes": _optional_metric_summary(arms, "wasted_bytes"),
                "churn_bytes": _optional_metric_summary(arms, "churn_bytes"),
                "store_gauges_after": [
                    arm["metrics"]["store_gauges_after"] for arm in arms
                ],
            },
        }
    if (
        result["S0"]["recovery_success_fraction"]["count"] != 12
        or result["S0"]["fallback_fraction"]["count"] != 110
    ):
        raise ConsolidationError(
            "R4 S0 must retain 12/122 recovery and 110/122 fallback"
        )
    if result["S0"]["outcomes"]["requests"] != 122:
        raise ConsolidationError("R4 S0 outcome denominator must be 122")
    return {
        "proxy": "R4-like-5x synthetic footprint proxy",
        "not_kvcomm": True,
        "performance_ranking_enabled": False,
        "ranking": "disabled",
        "policies": result,
    }


def _compact_capacity(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "requested_capacity": raw["requested_capacity"],
        "observed_capacity": raw["observed_capacity"],
        "cells": [
            {
                "policy": cell["policy"],
                "capacity_relative_error": cell["capacity_relative_error"],
                "status": normalized_diagnostic(cell["status"]),
                "profiles": [
                    {
                        "profile": profile["profile"],
                        "reachability": normalized_diagnostic(profile["reachability"]),
                        "valid": profile["valid"],
                        "representation_kinds": profile["representation_kinds"],
                        "formal_cache_outcomes": [
                            repeat["cache_outcomes"] for repeat in profile["formal"]
                        ],
                    }
                    for profile in cell["profiles"]
                ],
            }
            for cell in raw["cells"]
        ],
        "bidirectional_pressure": raw["bidirectional_pressure"],
        "fallback_reachability": raw["fallback_reachability"],
    }


def _compact_ceiling(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "workload": raw["workload"],
        "summary": raw["summary"],
        "early_stop": raw["early_stop"],
        "canary": [
            repeat["arms"]["R0"]["same_context_canary"] for repeat in raw["formal"]
        ],
        "per_repeat": [
            {
                "repeat_index": repeat["repeat_index"],
                "amortization": repeat["amortization"],
                "arms": {
                    arm: {
                        "ledger": data["ledger"],
                        "metrics": data["metrics"],
                        "target_outcomes": [
                            {
                                "target_id": target["target_id"],
                                "outcome": target["outcome"],
                                "expected_outcome": target["expected_outcome"],
                                "request_path_ms": target["request_path_ms"],
                            }
                            for target in data["targets"]
                        ],
                    }
                    for arm, data in repeat["arms"].items()
                },
            }
            for repeat in raw["formal"]
        ],
    }


def _compact_scheduler(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw["setting_id"].startswith("p7-w-r4like-"):
        return {
            "performance_contract": raw["performance_contract"],
            "outcome": raw["outcome"],
            "formal": [
                {
                    "repeat_index": repeat["repeat_index"],
                    "diagnostic": {
                        "diagnostic_status": normalized_diagnostic(
                            arm["diagnostic_status"]
                        ),
                        "diagnostic_error": arm["diagnostic_error"],
                        "diagnostic_error_stage": arm["diagnostic_error_stage"],
                        "request_diagnostic": arm["request_diagnostic"],
                        "representation_multiplicity": arm["setup"][
                            "representation_multiplicity"
                        ],
                        "representation_kinds": arm["setup"]["representation_kinds"],
                        "registration_failed": arm["setup"]["registration_failed"],
                        "completed_request_records": len(arm["records"]),
                        "metrics": arm["metrics"],
                        "victim_sequence": arm["victim_sequence"],
                        "memory_footprint_after": arm["memory_footprint_after"],
                        "representation_metadata": arm["representation_metadata"],
                        "outcomes": _record_outcome_composition(
                            arm["records"],
                            outcome_taxonomy=raw["outcome"]["taxonomy"],
                            terminal_reasons=raw["outcome"][
                                "exclusive_terminal_reasons"
                            ],
                        ),
                    },
                }
                for repeat in raw["formal"]
                for arm in [repeat["arms"]["R4-like-5x"]]
            ],
        }
    return {
        "workload": raw["workload"],
        "performance_contract": raw["performance_contract"],
        "arm_statistics": raw["arm_statistics"],
        "outcome": raw["outcome"],
        "paired_E0_R0": {
            "pair_count": raw["paired_E0_R0"]["pair_count"],
            "denominators": {
                denominator: _paired_denominator_view(row)
                for denominator, row in raw["paired_E0_R0"]["denominators"].items()
            },
        },
        "paired_per_repeat": [
            {
                "repeat_index": row["repeat_index"],
                "paired": {
                    "pair_count": row["paired"]["pair_count"],
                    "denominators": {
                        denominator: _paired_denominator_view(data)
                        for denominator, data in row["paired"]["denominators"].items()
                    },
                },
            }
            for row in raw["paired_per_repeat"]
        ],
        "formal": [
            {
                "repeat_index": repeat["repeat_index"],
                "arms": {
                    arm: {
                        "metrics": data["metrics"],
                        "victim_sequence": data["victim_sequence"],
                        "memory_footprint_after": data["memory_footprint_after"],
                        "representation_metadata": data["representation_metadata"],
                        "outcomes": _record_outcome_composition(
                            data["records"],
                            outcome_taxonomy=raw["outcome"]["taxonomy"],
                            terminal_reasons=raw["outcome"][
                                "exclusive_terminal_reasons"
                            ],
                        ),
                    }
                    for arm, data in repeat["arms"].items()
                },
            }
            for repeat in raw["formal"]
        ],
        "peak_device_bytes": raw["rho"]["arm_interval_peak_by_repeat_arm"],
    }


def build_compact_artifact(
    raw: Mapping[str, Any],
    *,
    raw_relative_path: str,
    log_relative_path: str,
    raw_file_sha256: str,
    log_sha256: str,
) -> dict[str, Any]:
    if raw["phase"] == "Phase7-capacity":
        key_metrics = _compact_capacity(raw)
    elif raw["phase"] == "Phase7-ceiling":
        key_metrics = _compact_ceiling(raw)
    elif raw["phase"] == "Phase7-scheduler":
        key_metrics = _compact_scheduler(raw)
    else:
        raise ConsolidationError(f"unknown Phase7 phase {raw['phase']}")
    payload = {
        "schema_version": 1,
        "artifact": "phase7-compact-result",
        "phase": raw["phase"],
        "run_id": raw["run_id"],
        "setting_id": raw["setting_id"],
        "setting": raw["setting"],
        "restart_index": raw["restart_index"],
        "status": raw["status"],
        "engineering_valid": raw["status"] in ALLOWED_RAW_STATUSES,
        "engineering_valid_derivation": {
            "observed_raw_status": raw["status"],
            "allowed_raw_statuses": sorted(ALLOWED_RAW_STATUSES),
            "rule": (
                "true only after raw validation succeeds and status is in the "
                "allowed non-invalid set"
            ),
        },
        "runner": raw["runner"],
        "outcome": raw["outcome"],
        "inactive_counter_assertion": raw["inactive_counter_assertion"],
        "reset_validation": {"passed": True},
        "provenance": raw["provenance"],
        "source_hashes": {
            "raw": {
                "path": raw_relative_path,
                "internal_raw_sha256": raw["raw_sha256"],
                "file_sha256": raw_file_sha256,
            },
            "server_log": {
                "path": log_relative_path,
                "sha256": log_sha256,
            },
        },
        "key_metrics": key_metrics,
    }
    return attach_self_hash(payload, "compact_sha256")


def _local_phase7_path(publication_dir: Path, declared_path: str) -> Path:
    marker = "benchmark/approx_kv/results/phase7/"
    if not declared_path.startswith(marker):
        raise ConsolidationError(
            f"publication artifact path is outside Phase7: {declared_path}"
        )
    return publication_dir / declared_path.removeprefix(marker)


def publication_provenance(
    *,
    manifest: Mapping[str, Any],
    publication_dir: Path,
    raws: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cpu_evidence = {}
    publication_inputs = []
    for runner_key in ("capacity_pilot", "ceiling", "scheduler"):
        evidence_summary = manifest["runners"][runner_key]["cpu_test_evidence"]
        evidence_path = _local_phase7_path(publication_dir, evidence_summary["path"])
        if (
            not evidence_path.is_file()
            or file_sha256(evidence_path) != evidence_summary["file_sha256"]
        ):
            raise ConsolidationError(f"{runner_key} CPU evidence file mismatch")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (
            canonical_sha256(evidence, "artifact_sha256")
            != evidence.get("artifact_sha256")
            or evidence.get("artifact_sha256") != evidence_summary["artifact_sha256"]
            or evidence.get("runner_sha256")
            != manifest["runners"][runner_key]["sha256"]
            or evidence.get("exit_code") != 0
        ):
            raise ConsolidationError(f"{runner_key} CPU evidence binding mismatch")
        cpu_evidence[runner_key] = {
            "path": str(evidence_path.relative_to(publication_dir)),
            "file_sha256": file_sha256(evidence_path),
            "artifact_sha256": evidence["artifact_sha256"],
            "command": evidence["command"],
            "passed_count": evidence["passed_count"],
            "summary_line": evidence["summary_line"],
            "runner_sha256": evidence["runner_sha256"],
        }
        publication_inputs.append(evidence_summary["path"])

    review_summary = manifest["review_evidence"]
    review_path = _local_phase7_path(
        publication_dir, manifest["review_contract"]["artifact_path"]
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if file_sha256(review_path) != review_summary[
        "artifact_sha256"
    ] or canonical_sha256(review, "artifact_sha256") != review.get("artifact_sha256"):
        raise ConsolidationError("final entry-review artifact binding mismatch")

    result_manifest_path = publication_dir / "RESULT_MANIFEST.json"
    if not result_manifest_path.is_file():
        raise ConsolidationError("Phase7 RESULT_MANIFEST.json is missing")
    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    entries = {
        entry["file"]: entry
        for entry in result_manifest.get("files", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("file"), str)
    }
    publication_inputs.extend(
        [
            "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json",
            manifest["review_contract"]["artifact_path"],
        ]
    )
    missing_entries = sorted(set(publication_inputs) - set(entries))
    if missing_entries:
        raise ConsolidationError(
            f"RESULT_MANIFEST lacks publication inputs: {missing_entries}"
        )
    drifted_entries = []
    for declared in publication_inputs:
        local_path = _local_phase7_path(publication_dir, declared)
        if entries[declared].get("sha256") != file_sha256(local_path):
            drifted_entries.append(declared)
    if drifted_entries:
        raise ConsolidationError(
            f"RESULT_MANIFEST publication input hashes drifted: {drifted_entries}"
        )

    execution_heads = sorted(
        {
            (
                raw["execution_envelope"]["execution_head_git_sha"],
                raw["execution_envelope"]["execution_head_tree_sha"],
            )
            for raw in raws
        }
    )
    design_preserving = (
        review["reviewed_manifest_revision"] == 11
        and manifest["manifest_revision"] == 12
        and manifest["supersedes_manifest_sha256"] == review["reviewed_manifest_sha256"]
        and manifest["design_payload_sha256"] == review["design_payload_sha256"]
        and manifest["supersedes_design_payload_sha256"]
        == manifest["design_payload_sha256"]
    )
    if not design_preserving:
        raise ConsolidationError("rev11 to rev12 entry-review transition drifted")
    return {
        "environment": manifest["environment"],
        "code_pin": {
            "implementation_git_sha": manifest["implementation"][
                "phase7_pinned_implementation_sha"
            ],
            "implementation_tree_sha": manifest["implementation"][
                "phase7_pinned_tree_sha"
            ],
            "runner_sha256": {
                runner_key: manifest["runners"][runner_key]["sha256"]
                for runner_key in ("capacity_pilot", "ceiling", "scheduler")
            },
        },
        "execution_heads": [
            {"git_sha": git_sha, "tree_sha": tree_sha}
            for git_sha, tree_sha in execution_heads
        ],
        "cpu_evidence": cpu_evidence,
        "result_manifest": {
            "path": "RESULT_MANIFEST.json",
            "hash_omitted_to_avoid_summary_result_manifest_cycle": True,
            "publication_inputs_validated": sorted(publication_inputs),
            "authority": result_manifest.get("authority"),
        },
        "entry_review": {
            "path": str(review_path.relative_to(publication_dir)),
            "file_sha256": file_sha256(review_path),
            "artifact_sha256": review["artifact_sha256"],
            "verdict": review["verdict"],
            "open_p0": review["open_p0"],
            "open_p1": review["open_p1"],
            "reviewed_manifest_revision": 11,
            "authorized_manifest_revision": 12,
            "design_payload_sha256": manifest["design_payload_sha256"],
            "design_preserving_rev11_to_rev12": True,
            "explanation": (
                "rev12 activates authorization and supersedes reviewed rev11 "
                "without changing the reviewed design payload"
            ),
        },
    }


def _w_interpretation(comparison: Mapping[str, Any]) -> dict[str, Any]:
    median = comparison["median_across_restarts"]
    all_reusable = median["all_reusable"]["mean_speedup_s0_over_s4"]
    workflow_only = median["workflow_only"]["mean_speedup_s0_over_s4"]
    miss_delta_all = median["miss_delta_s4_minus_s0"]["all_reusable"]
    miss_delta_workflow = median["miss_delta_s4_minus_s0"]["workflow_only"]
    peak_ratio = median["peak_ratio_s4_over_s0"]
    return {
        "comparison_design": comparison["comparison_design"],
        "all_reusable_mean_speedup_s0_over_s4": all_reusable,
        "workflow_only_mean_speedup_s0_over_s4": workflow_only,
        "miss_delta_s4_minus_s0": {
            "all_reusable": miss_delta_all,
            "workflow_only": miss_delta_workflow,
        },
        "peak_ratio_s4_over_s0": peak_ratio,
        "statement": (
            f"seed-matched non-adjacent restart medians: all-reusable "
            f"S0/S4 mean={all_reusable:.4f}, workflow-only={workflow_only:.4f}, "
            f"miss deltas S4-S0 all={miss_delta_all:+g}, "
            f"workflow={miss_delta_workflow:+g}, peak S4/S0={peak_ratio:.4f}"
        ),
    }


def build_summary(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    staging_dir: Path,
    plan: Mapping[str, Sequence[tuple[str, int]]],
    raws: Sequence[Mapping[str, Any]],
    raw_sources: Sequence[Mapping[str, Any]],
    compact_sources: Sequence[Mapping[str, Any]],
    central_path: Path,
    central_durations: Mapping[str, Any],
    evidence_corrections: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in raws:
        by_setting[raw["setting_id"]].append(raw)

    correction_artifact = (
        evidence_corrections[0]["artifact"] if evidence_corrections else None
    )
    wave0 = summarize_wave0(
        [
            *by_setting["p6delta-s4-rho2-chunk4096"],
            *by_setting["p6delta-s0-rho2-chunk4096"],
        ],
        correction=correction_artifact,
    )
    a8 = aggregate_a8(
        [
            raw
            for setting_id in (
                "p7-a8-r0-body1024-rho1.5",
                "p7-a8-r0-body1024-rho2.0",
                "p7-a8-r0-body2048-rho1.5",
                "p7-a8-r0-body2048-rho2.0",
            )
            for raw in by_setting[setting_id]
        ]
    )
    sensitivity = aggregate_chunk1024_sensitivity(
        by_setting["p7-a8-r0-body2048-rho2-chunk1024-sensitivity"]
    )
    w_raws = [
        raw
        for setting_id in (
            "p7-w-r0-lru-rho1.5",
            "p7-w-r0-hierarchical-rho1.5",
            "p7-w-r0-lru-rho2.0",
            "p7-w-r0-hierarchical-rho2.0",
        )
        for raw in by_setting[setting_id]
    ]
    w_restart_rows = [_restart_w_metrics(raw) for raw in w_raws]
    w_cross_policy = aggregate_w_cross_policy(w_restart_rows)
    within_policy = aggregate_within_policy_latency(w_restart_rows)
    victim_footprint = summarize_w_victim_footprint(w_raws)
    r4 = summarize_r4(
        [
            *by_setting["p7-w-r4like-lru-rho2"],
            *by_setting["p7-w-r4like-hierarchical-rho2"],
        ]
    )
    budget = manifest["budget"]
    elapsed_hours = central_durations["total_elapsed_gpu_equivalent_hours"]
    publication = publication_provenance(
        manifest=manifest,
        publication_dir=manifest_path.parent,
        raws=raws,
    )
    correction_elapsed_seconds = sum(
        correction["elapsed"]["total_elapsed_seconds"]
        for correction in evidence_corrections
    )
    correction_elapsed_hours = correction_elapsed_seconds / 3600.0
    w_fallback_min = min(
        row["r0_outcomes"]["dense_fallback"]["rate"] for row in w_restart_rows
    )
    w_fallback_max = max(
        row["r0_outcomes"]["dense_fallback"]["rate"] for row in w_restart_rows
    )
    correction_rows = [
        {
            "scope": correction["artifact"]["correction"]["scope"],
            "setting_id": correction["artifact"]["setting_id"],
            "restart_index": correction["artifact"]["restart_index"],
            "original_raw_sha256": correction["artifact"]["correction"][
                "original_raw_sha256"
            ],
            "correction_raw_sha256": correction["artifact"]["raw_sha256"],
            "status": correction["artifact"]["status"],
            "manifest_binding": {
                "base_manifest_revision": correction["artifact"][
                    "base_manifest_revision"
                ],
                "base_manifest_self_sha256": correction["artifact"][
                    "base_manifest_self_sha256"
                ],
                "correction_manifest_revision": correction["artifact"][
                    "correction_manifest_revision"
                ],
                "correction_manifest_sha256": correction["artifact"][
                    "correction_manifest_sha256"
                ],
                "design_payload_sha256": manifest["design_payload_sha256"],
            },
            "terminal_reason_counts": correction["artifact"]["outcome"][
                "terminal_reason_counts"
            ],
            "runner": correction["artifact"]["runner"],
            "execution_head": {
                "git_sha": correction["artifact"]["execution_envelope"][
                    "execution_head_git_sha"
                ],
                "tree_sha": correction["artifact"]["execution_envelope"][
                    "execution_head_tree_sha"
                ],
                "worktree_clean": correction["artifact"]["execution_envelope"][
                    "worktree_clean"
                ],
                "worktree_status_entries": correction["artifact"]["execution_envelope"][
                    "worktree_status_entries"
                ],
                "post_pin_changed_paths": correction["artifact"]["execution_envelope"][
                    "post_pin_changed_paths"
                ],
            },
            "source": correction["source"],
            "elapsed": correction["elapsed"],
        }
        for correction in evidence_corrections
    ]
    engineering_status = ENGINEERING_STATUS_VALID
    mechanism_status = a8["mechanism_status"]
    system_behaviour_status = SYSTEM_BEHAVIOUR_STATUS
    execution_counts = {
        "executed_starts": len(plan["executed"]),
        "wave0_required": len(plan["wave0_required"]),
        "a8_primary_restart0": len(plan["a8_primary_restart0"]),
        "a8_primary_supplements_skipped": len(
            plan["a8_primary_supplements_skipped_es_r0_mde"]
        ),
        "chunk1024_sensitivity": len(plan["chunk1024_sensitivity"]),
        "w_main": len(plan["w_main"]),
        "r4_diagnostic": len(plan["r4_diagnostic"]),
        "rho3_conditional_disabled": len(plan["rho3_conditional_disabled"]),
        "committed_manifest_starts": budget["committed_server_starts"],
        "conditional_manifest_starts": budget["conditional_server_starts"],
    }
    summary = {
        "schema_version": 1,
        "artifact": "phase7-consolidated-summary",
        "offline_only": True,
        "engineering_status": engineering_status,
        "mechanism_status": mechanism_status,
        "system_behaviour_status": system_behaviour_status,
        "environment": publication["environment"],
        "provenance": {
            "code_pin": publication["code_pin"],
            "execution_heads": publication["execution_heads"],
            "cpu_evidence": publication["cpu_evidence"],
            "result_manifest": publication["result_manifest"],
            "entry_review": publication["entry_review"],
        },
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": file_sha256(manifest_path),
            "manifest_revision": manifest["manifest_revision"],
            "preregistered_manifest_sha256": manifest["preregistered_manifest_sha256"],
            "design_payload_sha256": manifest["design_payload_sha256"],
            "status": manifest["status"],
        },
        "execution": {
            "counts": execution_counts,
            "executed": [
                {"setting_id": setting_id, "restart_index": restart}
                for setting_id, restart in plan["executed"]
            ],
            "skips": {
                "a8_primary_supplements": {
                    "count": len(plan["a8_primary_supplements_skipped_es_r0_mde"]),
                    "rule": "ES-R0-MDE",
                    "runs": [
                        {"setting_id": setting_id, "restart_index": restart}
                        for setting_id, restart in plan[
                            "a8_primary_supplements_skipped_es_r0_mde"
                        ]
                    ],
                },
                "rho3_conditional": {
                    "count": 1,
                    "disposition": "disabled_scoped_chunk1024",
                    "runs": [
                        {"setting_id": setting_id, "restart_index": restart}
                        for setting_id, restart in plan["rho3_conditional_disabled"]
                    ],
                },
            },
            "elapsed": {
                "source": "central JSONL running/completed timestamps",
                "actual_elapsed_seconds": central_durations["total_elapsed_seconds"],
                "actual_elapsed_gpu_equivalent_hours": elapsed_hours,
                "wall_clock_span_hours": central_durations["wall_clock_span_hours"],
                "sum_of_run_intervals_exclusion_note": central_durations[
                    "sum_of_run_intervals_exclusion_note"
                ],
                "per_run": central_durations["runs"],
            },
            "budget_comparison": {
                "expected_gpu_hours_total": budget["expected_gpu_hours_total"],
                "hard_cap_gpu_hours": budget["hard_cap_gpu_hours"],
                "actual_minus_expected_gpu_hours": (
                    elapsed_hours - budget["expected_gpu_hours_total"]
                ),
                "actual_fraction_of_expected": (
                    elapsed_hours / budget["expected_gpu_hours_total"]
                ),
                "actual_fraction_of_hard_cap": (
                    elapsed_hours / budget["hard_cap_gpu_hours"]
                ),
                "within_hard_cap": elapsed_hours <= budget["hard_cap_gpu_hours"],
                "executed_minus_committed_starts": (
                    len(plan["executed"]) - budget["committed_server_starts"]
                ),
                "hard_cap_starts": budget["hard_cap_server_starts"],
            },
        },
        "evidence_corrections": {
            "count": len(correction_rows),
            "excluded_from_executed_starts": True,
            "excluded_from_primary_elapsed_gpu_equivalent_hours": True,
            "runs": correction_rows,
            "elapsed": {
                "total_elapsed_seconds": correction_elapsed_seconds,
                "total_elapsed_gpu_equivalent_hours": correction_elapsed_hours,
                "wall_clock_span_hours": sum(
                    correction["elapsed"]["wall_clock_span_hours"]
                    for correction in evidence_corrections
                ),
            },
            "budget": {
                "counts_against_authorized_22_starts": False,
                "counts_against_preregistered_gpu_hour_budget": False,
                "classification": "post_hoc_evidence_correction",
            },
        },
        "wave0_profile_reachability": wave0,
        "a8_ceiling": a8,
        "chunk1024_sensitivity": sensitivity,
        "w_scheduler": {
            "independent_replicate_unit": "server_restart",
            "requests_are_not_independent_replicates": True,
            "per_restart": sorted(
                w_restart_rows,
                key=lambda row: (
                    row["rho_logical_demand"],
                    row["policy"],
                    row["restart_index"],
                ),
            ),
            "cross_policy_s4_vs_s0": w_cross_policy,
            "within_policy_r0_vs_e0": within_policy,
            "victim_footprint": victim_footprint,
            "interpretation": {
                "rho1.5": _w_interpretation(w_cross_policy["1.5"]),
                "rho2.0": _w_interpretation(w_cross_policy["2.0"]),
                "claim_rule": (
                    "the preregistered rules do not permit a practical benefit "
                    "claim; latency is secondary after A8 NEGATIVE"
                ),
                "comparison_design": ("seed-matched_non_adjacent_restart_comparison"),
                "latency_ratio_fallback_mix": {
                    "minimum_dense_fallback_rate": w_fallback_min,
                    "maximum_dense_fallback_rate": w_fallback_max,
                    "explanation": (
                        "W latency ratios mix approximate recoveries with "
                        f"{w_fallback_min:.1%}-{w_fallback_max:.1%} dense fallback"
                    ),
                },
                "system_behaviour_status": system_behaviour_status,
            },
        },
        "r4_diagnostic": r4,
        "status_separation": {
            "engineering_status": engineering_status,
            "engineering_basis": (
                "all 22 primary artifacts passed hash/provenance/reset/inactive "
                "validation and the required S0 terminal-reason correction passed"
            ),
            "a8_mechanism_status": mechanism_status,
            "w_system_behaviour_status": system_behaviour_status,
            "r4_status": (
                "S0 available/valid; S4 diagnostic_unavailable; both are "
                "synthetic 5x footprint proxies and neither executes KVCOMM"
            ),
        },
        "status_derivation": {
            "engineering_status": {
                "value": engineering_status,
                "rule": (
                    "VALID is emitted only after all expected primary artifacts, "
                    "central bindings, hashes, resets, inactive counters, and any "
                    "required evidence correction validate"
                ),
            },
            "mechanism_status": {
                "value": mechanism_status,
                "rule": (
                    "NEGATIVE is derived from all four A8 restart-0 request-path "
                    "medians remaining below the preregistered 1.05 MDE"
                ),
            },
            "system_behaviour_status": {
                "value": system_behaviour_status,
                "rule": (
                    "W has three independent restarts per policy/rho but is "
                    "descriptive after A8 NEGATIVE and mixes dense fallback"
                ),
            },
        },
        "scope_caveats": list(manifest["scope_caveats"]),
        "additional_caveats": [
            "R2 is disabled_not_comparable and was not executed",
            "r2_like is a synthetic 2x footprint profile and is not R2 execution",
            "R4-like-5x is a synthetic footprint proxy and is not KVCOMM",
            "chunk1024 sensitivity is chunk-coupled and is not a headline result",
            "request rows within W traces are not independent timing replicates",
            "R0 is a ceiling path, not a practical promoted candidate",
        ],
        "source_hashes": {
            "central_jsonl": {
                "path": str(central_path.relative_to(staging_dir)),
                "sha256": file_sha256(central_path),
            },
            "raw": sorted(raw_sources, key=lambda row: row["path"]),
            "logs": sorted(
                [
                    {
                        "path": row["log_path"],
                        "sha256": row["log_sha256"],
                    }
                    for row in raw_sources
                ],
                key=lambda row: row["path"],
            ),
            "compact": sorted(compact_sources, key=lambda row: row["path"]),
            "evidence_corrections": [
                correction["source"] for correction in evidence_corrections
            ],
        },
    }
    return attach_self_hash(summary, "summary_sha256")


def _serialized_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _serialized_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_serialized_json(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialized_json(payload), encoding="utf-8")


def _ensure_output_paths(
    compact_paths: Sequence[Path],
    output: Path,
    *,
    force: bool,
    protected_paths: Sequence[Path] = (),
) -> None:
    paths = [*compact_paths, output]
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(paths):
        raise ConsolidationError("compact and summary output paths must be distinct")
    protected = {path.resolve() for path in protected_paths}
    collisions = [str(path) for path in resolved_paths if path in protected]
    if collisions:
        raise ConsolidationError(
            f"output paths must not overwrite source artifacts: {collisions}"
        )
    if not force:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing paths: {existing}")
    invalid_types = [
        str(path) for path in paths if path.exists() and not path.is_file()
    ]
    if invalid_types:
        raise ConsolidationError(
            f"output paths exist but are not regular files: {invalid_types}"
        )


def consolidate(
    *,
    staging_dir: Path,
    manifest_path: Path,
    compact_dir: Path,
    output: Path,
    correction_dir: Path | None = None,
    correction_manifest_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    staging_dir = staging_dir.resolve()
    manifest_path = manifest_path.resolve()
    compact_dir = compact_dir.resolve()
    output = output.resolve()
    correction_dir = correction_dir.resolve() if correction_dir is not None else None
    correction_manifest_path = (
        correction_manifest_path.resolve()
        if correction_manifest_path is not None
        else (REPO_ROOT / CAPACITY_CORRECTION_MANIFEST_PATH).resolve()
    )
    if not staging_dir.is_dir():
        raise ConsolidationError(f"staging directory does not exist: {staging_dir}")
    if not manifest_path.is_file():
        raise ConsolidationError(f"manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_authorized_manifest(manifest)
    plan = expected_execution_plan(manifest)
    expected_runs = set(plan["executed"])
    settings = _manifest_setting_map(manifest)

    raw_dir = staging_dir / "raw"
    log_dir = staging_dir / "logs"
    central_path = staging_dir / "phase7-runs.jsonl"
    if not raw_dir.is_dir() or not log_dir.is_dir() or not central_path.is_file():
        raise ConsolidationError("staging raw/log/central inputs are incomplete")
    raw_paths = sorted(raw_dir.glob("*.json"))
    log_paths = sorted(log_dir.glob("*.log"))
    if len(raw_paths) != 22 or len(log_paths) != 22:
        raise ConsolidationError(
            f"staging must contain exactly 22 raw JSON and 22 logs, got "
            f"{len(raw_paths)} and {len(log_paths)}"
        )

    raws = []
    raw_sources = []
    actual_runs: set[tuple[str, int]] = set()
    central_by_run: dict[tuple[str, int], Mapping[str, Any]] = {}
    manifest_file_hash = file_sha256(manifest_path)
    all_central_events = parse_central_events(
        central_path.read_text(encoding="utf-8").splitlines()
    )
    primary_central_events = [
        event for event in all_central_events if "correction" not in event
    ]
    correction_central_events = [
        event for event in all_central_events if "correction" in event
    ]
    central = central_run_durations(
        primary_central_events,
        expected_runs,
        excluded_run_classes=("evidence-correction runs",),
    )
    for row in central["runs"]:
        central_by_run[(row["setting_id"], row["restart_index"])] = row

    for raw_path in raw_paths:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        pair = (raw.get("setting_id"), raw.get("restart_index"))
        if pair in actual_runs:
            raise ConsolidationError(f"duplicate raw setting/restart: {pair}")
        if pair not in expected_runs:
            raise ConsolidationError(f"unexpected raw setting/restart: {pair}")
        expected_name = f"{pair[0]}-r{pair[1]}.json"
        if raw_path.name != expected_name:
            raise ConsolidationError(
                f"raw filename mismatch for {pair}: {raw_path.name}"
            )
        is_known_capacity_correction_target = pair == (
            "p6delta-s0-rho2-chunk4096",
            0,
        )
        needs_capacity_correction = (
            is_known_capacity_correction_target
            and capacity_terminal_reason_correction_required(raw)
        )
        if needs_capacity_correction:
            require_capacity_terminal_reason_correction(
                original=raw,
                correction_dir=correction_dir,
            )
        hashes = validate_raw_artifact(
            raw,
            path=raw_path,
            staging_dir=staging_dir,
            manifest=manifest,
            manifest_file_hash=manifest_file_hash,
            settings=settings,
            allow_missing_capacity_terminal_reason_correction=(
                needs_capacity_correction and correction_dir is not None
            ),
        )
        central_row = central_by_run[pair]
        if central_row["raw_sha256"] != raw["raw_sha256"]:
            raise ConsolidationError(f"central completion/raw hash mismatch for {pair}")
        if central_row["run_id"] != raw.get("run_id"):
            raise ConsolidationError(f"central/raw run_id mismatch for {pair}")
        if central_row["phase"] != raw.get("phase"):
            raise ConsolidationError(f"central/raw phase mismatch for {pair}")
        central_output = central_row.get("output")
        if (
            isinstance(central_output, str)
            and Path(central_output).name != raw_path.name
        ):
            raise ConsolidationError(f"central/raw output path mismatch for {pair}")
        raw_file_hash = file_sha256(raw_path)
        log_path = log_dir / f"{raw_path.stem}.log"
        raw_sources.append(
            {
                "setting_id": pair[0],
                "restart_index": pair[1],
                "path": str(raw_path.relative_to(staging_dir)),
                "internal_raw_sha256": hashes["raw_sha256"],
                "file_sha256": raw_file_hash,
                "log_path": str(log_path.relative_to(staging_dir)),
                "log_sha256": hashes["log_sha256"],
            }
        )
        actual_runs.add(pair)
        raws.append(raw)
    if actual_runs != expected_runs:
        raise ConsolidationError("raw execution set does not match the expected 22")

    s0_wave0 = next(
        raw
        for raw in raws
        if raw["setting_id"] == "p6delta-s0-rho2-chunk4096"
        and raw["restart_index"] == 0
    )
    evidence_corrections = []
    if correction_dir is not None:
        if not capacity_terminal_reason_correction_required(s0_wave0):
            raise ConsolidationError(
                "capacity correction is not permitted when the original S0 raw "
                "already has terminal-reason evidence"
            )
        evidence_corrections.append(
            load_evidence_correction(
                correction_dir=correction_dir,
                correction_manifest_path=correction_manifest_path,
                original=s0_wave0,
                original_file_sha256=next(
                    row["file_sha256"]
                    for row in raw_sources
                    if row["setting_id"] == CAPACITY_CORRECTION_SETTING_ID
                    and row["restart_index"] == CAPACITY_CORRECTION_RESTART
                ),
                manifest=manifest,
                fallback_central_events=correction_central_events,
            )
        )

    compact_paths = [
        compact_dir / f"{raw['setting_id']}-r{raw['restart_index']}.compact.json"
        for raw in raws
    ]
    _ensure_output_paths(
        compact_paths,
        output,
        force=force,
        protected_paths=[
            manifest_path,
            correction_manifest_path,
            central_path,
            *raw_paths,
            *log_paths,
            *[
                correction_dir / correction["source"]["path"]
                for correction in evidence_corrections
            ],
            *[
                correction_dir / correction["source"]["log_path"]
                for correction in evidence_corrections
            ],
            *[
                Path(correction["source"]["review_path"])
                for correction in evidence_corrections
            ],
            *[
                correction_manifest_path.parents[4]
                / correction["source"]["cpu_evidence_path"]
                for correction in evidence_corrections
            ],
        ],
    )

    compact_payloads = []
    compact_sources = []
    for raw, raw_source, compact_path in zip(raws, raw_sources, compact_paths):
        compact = build_compact_artifact(
            raw,
            raw_relative_path=raw_source["path"],
            log_relative_path=raw_source["log_path"],
            raw_file_sha256=raw_source["file_sha256"],
            log_sha256=raw_source["log_sha256"],
        )
        compact_payloads.append((compact_path, compact))
        compact_sources.append(
            {
                "setting_id": raw["setting_id"],
                "restart_index": raw["restart_index"],
                "path": f"compact/{compact_path.name}",
                "compact_sha256": compact["compact_sha256"],
                "file_sha256": _serialized_json_sha256(compact),
            }
        )

    summary = build_summary(
        manifest=manifest,
        manifest_path=manifest_path,
        staging_dir=staging_dir,
        plan=plan,
        raws=raws,
        raw_sources=raw_sources,
        compact_sources=compact_sources,
        central_path=central_path,
        central_durations=central,
        evidence_corrections=evidence_corrections,
    )
    for compact_path, compact in compact_payloads:
        _write_json(compact_path, compact)
    _write_json(output, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and consolidate offline Phase7 staging results."
    )
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correction-dir", type=Path)
    parser.add_argument(
        "--correction-manifest",
        type=Path,
        default=REPO_ROOT / CAPACITY_CORRECTION_MANIFEST_PATH,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly allow overwriting compact and summary JSON files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = consolidate(
        staging_dir=args.staging_dir,
        manifest_path=args.manifest,
        compact_dir=args.compact_dir,
        output=args.output,
        correction_dir=args.correction_dir,
        correction_manifest_path=args.correction_manifest,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "summary_sha256": summary["summary_sha256"],
                "engineering_status": summary["engineering_status"],
                "mechanism_status": summary["mechanism_status"],
                "system_behaviour_status": summary["system_behaviour_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
