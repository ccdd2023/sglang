from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from benchmark.approx_kv.build_phase7_manifest import (
    build_a8_workload,
    build_filler_pool,
    build_settings,
    build_w_workload,
    design_payload_sha256,
)
from benchmark.approx_kv.build_phase7_manifest import (
    payload_sha256 as phase7_payload_sha256,
)
from benchmark.approx_kv.build_phase7_manifest import (
    token_list_sha,
)
from benchmark.approx_kv.metrics import (
    clean_cache_invariant,
    clean_pool_reset_invariant,
    counter_delta,
    max_total_num_tokens,
)
from benchmark.approx_kv.phase6.manifest import fixed_object_token_ids
from benchmark.approx_kv.phase6.runner import REPO_ROOT
from benchmark.approx_kv.phase6.schema import (
    RhoDefinitions,
    file_sha256,
    payload_sha256,
    validate_phase6_artifact,
)
from benchmark.approx_kv.run_p6_4_capacity_pilot import labeled_metric_delta

CEILING_RUNNER = "benchmark.approx_kv.run_p7_ceiling"
SCHEDULER_RUNNER = "benchmark.approx_kv.run_p7_scheduler"

POST_PIN_ENVELOPE_PREFIX = "benchmark/approx_kv/results/phase7/"

OUTCOME_TAXONOMY = (
    "dense_no_reuse_baseline",
    "exact_gpu_hit",
    "ordinary_exact_cache_miss",
    "approximate_gpu_recovery",
    "host_demand_load",
    "approximate_recovery_failed_dense",
)
TERMINAL_REASONS = (
    "cross_store_reservation_failed",
    "device_allocation_failed",
    "unsupported",
    "registration_failed",
    "prefix_gap",
)

STORE_GAUGES = (
    "sglang:approx_kv_store_records",
    "sglang:approx_kv_store_device_bytes",
    "sglang:approx_kv_store_host_bytes",
    "sglang:approx_kv_store_leases",
    "sglang:approx_kv_store_orphans",
    "sglang:approx_kv_provisional_tokens",
    "sglang:cross_store_reserved_device_bytes",
)
# Only exported once the cross-store coordinator or a reset has run; a missing
# series is never interpreted as an explicit zero.
LATE_EXPORTED_GAUGES = ("sglang:cross_store_reserved_device_bytes",)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')

# Whitelist only: an unmapped raw reason is reported verbatim instead of being
# collapsed into ``unsupported``.
_RAW_TERMINAL_REASON_MAP = {
    "cross_store_reservation_failed": "cross_store_reservation_failed",
    "device_allocation_failed": "device_allocation_failed",
    "prefix_gap": "prefix_gap",
    "registration_store_capacity": "registration_failed",
    "store_miss": "unsupported",
    "source_pin_stale": "unsupported",
    "residency_load_failed": "unsupported",
    "rope_config_unavailable": "unsupported",
    "cross_store_error": "unsupported",
    "stale_handle": "unsupported",
    "residency_miss": "unsupported",
    "source_slice_mismatch": "unsupported",
}
# Exact-side pressure signals are never approximate-recovery terminal reasons.
EXCLUDED_RAW_TERMINAL_REASONS = (
    "cross_store_exact_pressure_error",
    "cross_store_exact_pressure_failed",
)


class Phase7ContractError(ValueError):
    """Raised before server launch when a frozen Phase 7 contract is violated."""


class DisabledSettingError(Phase7ContractError):
    reason = "disabled_not_comparable"


class Phase7RunError(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        *,
        server_argv: Sequence[str],
        plugin_env: Mapping[str, str],
    ) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.server_argv = list(server_argv)
        self.plugin_env = dict(plugin_env)


@dataclass(frozen=True)
class Phase7ExecutionContext:
    manifest_path: Path
    manifest_file_sha256: str
    manifest: dict[str, Any]
    setting: dict[str, Any]
    restart_index: int
    runner_key: str
    runner_module: str
    runner_path: str
    runner_sha256: str
    source: dict[str, str]
    envelope: dict[str, Any]


def manifest_self_sha256(manifest: Mapping[str, Any]) -> str:
    return phase7_payload_sha256(dict(manifest))


def nested_manifest_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    return phase7_payload_sha256(canonical)


def pending_result_provenance() -> dict[str, Any]:
    return {
        "result_git_sha": None,
        "result_commit_status": "pending_result_commit",
    }


def _require_sha(value: Any, *, field: str, git: bool = False) -> str:
    pattern = _GIT_SHA_RE if git else _SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise Phase7ContractError(f"{field} is not a valid pinned hash")
    return value


def validate_manifest_envelope(
    manifest: Mapping[str, Any],
    *,
    require_authorized: bool,
) -> None:
    if manifest.get("artifact") != "phase7-primary-manifest":
        raise Phase7ContractError("not a Phase 7 primary manifest")
    if manifest.get("schema_version") != 1:
        raise Phase7ContractError("unsupported Phase 7 manifest schema")
    revision = manifest.get("manifest_revision")
    if not isinstance(revision, int) or revision < 6:
        raise Phase7ContractError("Phase 7 manifest revision must be at least 6")
    expected_self_hash = manifest_self_sha256(manifest)
    if manifest.get("preregistered_manifest_sha256") != expected_self_hash:
        raise Phase7ContractError("Phase 7 manifest self-hash mismatch")
    if manifest.get("design_payload_sha256") != design_payload_sha256(dict(manifest)):
        raise Phase7ContractError("Phase 7 immutable design hash mismatch")
    if tuple(manifest.get("outcome_taxonomy", ())) != OUTCOME_TAXONOMY:
        raise Phase7ContractError("Phase 7 outcome taxonomy drift")
    if tuple(manifest.get("exclusive_terminal_reasons", ())) != TERMINAL_REASONS:
        raise Phase7ContractError("Phase 7 terminal-reason taxonomy drift")

    for workload_name, workload in manifest.get("workloads", {}).items():
        if workload.get("manifest_sha256") != nested_manifest_sha256(workload):
            raise Phase7ContractError(
                f"{workload_name} nested manifest self-hash mismatch"
            )

    status = manifest.get("status")
    authorized = manifest.get("phase7_execution_authorized")
    blockers = manifest.get("execution_blockers")
    implementation = manifest.get("implementation", {})
    pinned_sha = implementation.get("phase7_pinned_implementation_sha")
    pinned_tree = implementation.get("phase7_pinned_tree_sha")
    if not isinstance(blockers, list):
        raise Phase7ContractError("execution_blockers must be a list")
    if status == "preregistered_blocked":
        if authorized or pinned_sha is not None or not blockers:
            raise Phase7ContractError("invalid preregistered_blocked state")
    elif status == "pinned_blocked":
        if authorized or pinned_sha is None or not blockers:
            raise Phase7ContractError("invalid pinned_blocked state")
    elif status == "authorized":
        if authorized is not True or pinned_sha is None or blockers:
            raise Phase7ContractError("invalid authorized state")
    else:
        raise Phase7ContractError(f"unsupported Phase 7 state: {status!r}")

    if pinned_sha is not None:
        _require_sha(pinned_sha, field="phase7_pinned_implementation_sha", git=True)
        _require_sha(pinned_tree, field="phase7_pinned_tree_sha", git=True)
    if require_authorized and status != "authorized":
        raise Phase7ContractError("Phase 7 execution requires an authorized manifest")

    plan = manifest.get("plan", {})
    if plan.get("version") != "V6":
        raise Phase7ContractError("Phase 7 execution requires the V6 plan")
    _require_sha(plan.get("plan_sha256"), field="plan_sha256")
    _require_sha(plan.get("plan_commit"), field="plan_commit", git=True)
    if manifest.get("r2_strategy") != "disabled_not_comparable":
        raise Phase7ContractError("V6 requires R2 disabled_not_comparable")
    if manifest.get("conditional_resolution", {}).get(
        "CR-R2-ADAPTER"
    ) != "disabled_not_comparable":
        raise Phase7ContractError("V6 R2 resolution is not disabled_not_comparable")
    if any("R2" in row.get("arms", ()) for row in manifest.get("settings", ())):
        raise Phase7ContractError("V6 must not contain R2 GPU settings")
    if manifest.get("conditional_user_authorization_recorded") is not True:
        raise Phase7ContractError("conditional user authorization is not recorded")
    review_contract = manifest.get("review_contract", {})
    review_evidence = manifest.get("review_evidence", {})
    if review_contract.get("final_opus_required") is not True:
        raise Phase7ContractError("final Opus review is not required")
    if review_evidence.get("status") not in {"pending", "passed"}:
        raise Phase7ContractError("invalid final Opus review status")
    if status == "authorized" and review_evidence.get("status") != "passed":
        raise Phase7ContractError("authorized manifest lacks final Opus approval")
    flags = manifest.get("server_template", {}).get("test_only_injection_flags")
    if flags != {
        "SGLANG_APPROX_KV_TEST_ONLY": "0",
        "SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE": "0",
    }:
        raise Phase7ContractError("test-only injection flags are not pinned off")
    plugin_env = manifest.get("server_template", {}).get("plugin_env", {})
    if plugin_env.get("SGLANG_APPROX_KV_ALLOW_PERSISTENT_PINS") != "1":
        raise Phase7ContractError("Phase 7 persistent pins are not enabled")
    if plugin_env.get("SGLANG_APPROX_KV_MAX_PERSISTENT_PINS") != "16":
        raise Phase7ContractError("Phase 7 persistent pin cap is not 16")


def select_setting(
    manifest: Mapping[str, Any],
    *,
    setting_id: str,
    restart_index: int,
    runner_module: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in manifest.get("settings", ())
        if row.get("setting_id") == setting_id
    ]
    if len(matches) != 1:
        raise Phase7ContractError(
            f"expected exactly one setting {setting_id!r}, found {len(matches)}"
        )
    setting = dict(matches[0])
    frozen = {row["setting_id"]: row for row in build_settings()}.get(setting_id)
    if frozen is None or setting != frozen:
        raise Phase7ContractError(f"{setting_id}: setting differs from frozen builder")
    if setting.get("runner") != runner_module:
        raise Phase7ContractError(
            f"{setting_id}: expected runner {runner_module}, "
            f"got {setting.get('runner')}"
        )
    if restart_index not in setting.get("restart_indices", ()):
        raise Phase7ContractError(
            f"{setting_id}: restart {restart_index} is not preregistered"
        )

    arms = tuple(setting.get("arms", ()))
    if "R2" in arms:
        strategy = manifest.get("r2_strategy")
        resolution = manifest.get("conditional_resolution", {}).get("CR-R2-ADAPTER")
        if strategy == "disabled_not_comparable" and resolution == (
            "disabled_not_comparable"
        ):
            raise DisabledSettingError(f"{setting_id}: R2 is disabled_not_comparable")
        if strategy != "adapter" or resolution != "enabled":
            raise Phase7ContractError(
                f"{setting_id}: R2 adapter is not enabled and pinned"
            )
    if runner_module == CEILING_RUNNER and not set(arms).issubset(
        {"D0", "E0", "R0", "R2"}
    ):
        raise Phase7ContractError(f"{setting_id}: invalid ceiling arms")
    if runner_module == SCHEDULER_RUNNER:
        if arms != ("R4-like-5x",) and set(arms) != {"E0", "R0"}:
            raise Phase7ContractError(f"{setting_id}: invalid scheduler arms")
    for repeat in range(int(setting["formal_repeats"])):
        order = setting["arm_order_by_repeat"].get(str(repeat))
        if order is None or sorted(order) != sorted(setting["arms"]):
            raise Phase7ContractError(
                f"{setting_id}: invalid arm order for repeat {repeat}"
            )
    return setting


def validate_runner_binding(
    manifest: Mapping[str, Any],
    *,
    runner_key: str,
    runner_module: str,
    runner_path: str,
    current_runner_sha256: str,
    pinned_runner_sha256: str,
    observed_pinned_sha: str,
    observed_pinned_tree: str,
) -> None:
    entry = manifest.get("runners", {}).get(runner_key)
    if not isinstance(entry, Mapping):
        raise Phase7ContractError(f"missing runner binding for {runner_key}")
    if entry.get("exists") is not True:
        raise Phase7ContractError(f"{runner_key} runner is not marked ready")
    if entry.get("cpu_test_status") != "passed":
        raise Phase7ContractError(f"{runner_key} CPU tests are not marked passed")
    if entry.get("review_status") != "reviewed":
        raise Phase7ContractError(f"{runner_key} review is not marked complete")
    if entry.get("path") != runner_path:
        raise Phase7ContractError(f"{runner_key} runner path mismatch")
    expected_runner_sha = _require_sha(
        entry.get("sha256"), field=f"{runner_key}.sha256"
    )
    if current_runner_sha256 != expected_runner_sha:
        raise Phase7ContractError(f"{runner_key} current runner blob hash mismatch")
    if pinned_runner_sha256 != expected_runner_sha:
        raise Phase7ContractError(f"{runner_key} pinned runner blob hash mismatch")

    implementation = manifest["implementation"]
    pinned_source_sha = implementation["phase7_pinned_implementation_sha"]
    pinned_source_tree = implementation["phase7_pinned_tree_sha"]
    if observed_pinned_sha != pinned_source_sha:
        raise Phase7ContractError("pinned source SHA mismatch")
    if observed_pinned_tree != pinned_source_tree:
        raise Phase7ContractError("pinned source tree mismatch")
    if runner_module not in {CEILING_RUNNER, SCHEDULER_RUNNER}:
        raise Phase7ContractError(f"unknown Phase 7 runner {runner_module}")


def post_pin_envelope_allowlist(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Repo-relative paths that may legally change after the code pin."""
    raw = manifest.get("implementation", {}).get("post_pin_envelope_allowlist")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise Phase7ContractError(
            "manifest does not declare a post-pin envelope allowlist"
        )
    allowlist = []
    for entry in raw:
        if not isinstance(entry, str) or not entry:
            raise Phase7ContractError("post-pin envelope allowlist entries must be str")
        path = PurePosixPath(entry)
        if path.is_absolute() or ".." in path.parts:
            raise Phase7ContractError(
                f"post-pin envelope path escapes the repository: {entry}"
            )
        if not entry.startswith(POST_PIN_ENVELOPE_PREFIX):
            raise Phase7ContractError(
                f"post-pin envelope path is outside the result envelope: {entry}"
            )
        allowlist.append(entry)
    if len(set(allowlist)) != len(allowlist):
        raise Phase7ContractError("post-pin envelope allowlist has duplicates")
    return tuple(allowlist)


def _git_text(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise Phase7ContractError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_blob(revision: str, path: str) -> bytes:
    result = subprocess.run(
        ("git", "show", f"{revision}:{path}"),
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise Phase7ContractError(f"{path} is absent from {revision}")
    return result.stdout


def _verify_envelope_path(path: str, *, head_sha: str) -> str:
    resolved = (REPO_ROOT / path).resolve()
    repo_root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(repo_root):
        raise Phase7ContractError(f"post-pin envelope path escapes REPO_ROOT: {path}")
    if not resolved.is_file():
        raise Phase7ContractError(f"post-pin envelope path is missing: {path}")
    current = resolved.read_bytes()
    if current != _git_blob(head_sha, path):
        raise Phase7ContractError(
            f"post-pin envelope path differs from the execution HEAD blob: {path}"
        )
    return hashlib.sha256(current).hexdigest()


def execution_envelope(
    manifest: Mapping[str, Any],
    *,
    pinned_sha: str,
    pinned_tree: str,
) -> dict[str, Any]:
    """Bind the pinned code commit to the current execution envelope.

    The execution HEAD may advance past the pinned code SHA only by
    post-pin result-envelope commits: the pinned commit must be an ancestor
    of HEAD, the worktree must be clean and every path changed between the
    pin and HEAD must be declared in the manifest allowlist.
    """
    status = _git_text("status", "--porcelain", "--untracked-files=all")
    if status:
        raise Phase7ContractError("execution worktree must be clean before execution")
    head_sha = _git_text("rev-parse", "HEAD")
    head_tree = _git_text("rev-parse", "HEAD^{tree}")
    if _git_text("rev-parse", f"{pinned_sha}^{{commit}}") != pinned_sha:
        raise Phase7ContractError("pinned code SHA does not resolve to itself")
    if _git_text("rev-parse", f"{pinned_sha}^{{tree}}") != pinned_tree:
        raise Phase7ContractError("pinned code tree mismatch")
    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", pinned_sha, head_sha),
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise Phase7ContractError(
            "pinned code SHA is not an ancestor of the execution HEAD"
        )
    allowlist = post_pin_envelope_allowlist(manifest)
    changed = [
        line
        for line in _git_text(
            "diff", "--name-only", f"{pinned_sha}..{head_sha}"
        ).splitlines()
        if line.strip()
    ]
    unexpected = sorted(set(changed) - set(allowlist))
    if unexpected:
        raise Phase7ContractError(
            f"post-pin changes outside the envelope allowlist: {unexpected}"
        )
    envelope_hashes = {
        path: _verify_envelope_path(path, head_sha=head_sha) for path in allowlist
    }
    return {
        "pinned_source_git_sha": pinned_sha,
        "pinned_source_tree_sha": pinned_tree,
        "execution_head_git_sha": head_sha,
        "execution_head_tree_sha": head_tree,
        "pinned_is_ancestor_of_execution_head": True,
        "worktree_clean": True,
        "post_pin_envelope_allowlist": list(allowlist),
        "post_pin_changed_paths": sorted(changed),
        "post_pin_envelope_sha256": envelope_hashes,
    }


def require_envelope_path(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    field: str,
) -> str:
    """Require ``path`` to be an allowlisted, in-repo envelope artifact."""
    resolved = path.resolve()
    repo_root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(repo_root):
        raise Phase7ContractError(f"{field} path is outside REPO_ROOT: {path}")
    relative = str(resolved.relative_to(repo_root))
    if relative not in post_pin_envelope_allowlist(manifest):
        raise Phase7ContractError(
            f"{field} path is not a declared post-pin envelope path: {relative}"
        )
    return relative


def load_execution_context(
    *,
    manifest_path: Path,
    setting_id: str,
    restart_index: int,
    runner_key: str,
    runner_module: str,
    runner_file: Path,
) -> Phase7ExecutionContext:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_envelope(manifest, require_authorized=True)
    setting = select_setting(
        manifest,
        setting_id=setting_id,
        restart_index=restart_index,
        runner_module=runner_module,
    )
    runner_path = str(runner_file.resolve().relative_to(REPO_ROOT))
    current_runner_sha = file_sha256(runner_file)
    implementation = manifest["implementation"]
    pinned_sha = implementation["phase7_pinned_implementation_sha"]
    pinned_tree = implementation["phase7_pinned_tree_sha"]
    envelope = execution_envelope(
        manifest,
        pinned_sha=pinned_sha,
        pinned_tree=pinned_tree,
    )
    require_envelope_path(manifest_path, manifest=manifest, field="manifest")
    pinned_runner_sha = hashlib.sha256(_git_blob(pinned_sha, runner_path)).hexdigest()
    validate_runner_binding(
        manifest,
        runner_key=runner_key,
        runner_module=runner_module,
        runner_path=runner_path,
        current_runner_sha256=current_runner_sha,
        pinned_runner_sha256=pinned_runner_sha,
        observed_pinned_sha=envelope["pinned_source_git_sha"],
        observed_pinned_tree=envelope["pinned_source_tree_sha"],
    )
    source = {
        "source_git_sha": pinned_sha,
        "source_tree_sha": pinned_tree,
        "execution_head_git_sha": envelope["execution_head_git_sha"],
        "execution_head_tree_sha": envelope["execution_head_tree_sha"],
        "source_binding": "pinned_code_sha_with_post_pin_result_envelope",
    }
    return Phase7ExecutionContext(
        manifest_path=manifest_path,
        manifest_file_sha256=file_sha256(manifest_path),
        manifest=manifest,
        setting=setting,
        restart_index=restart_index,
        runner_key=runner_key,
        runner_module=runner_module,
        runner_path=runner_path,
        runner_sha256=current_runner_sha,
        source=source,
        envelope=envelope,
    )


def a8_tokens(
    manifest: Mapping[str, Any],
    *,
    body_tokens: int,
) -> dict[str, Any]:
    expected = build_a8_workload()
    observed = manifest.get("workloads", {}).get("A8")
    if observed != expected:
        raise Phase7ContractError("A8 workload differs from frozen builder")
    matches = [
        row for row in observed["workloads"] if row["body_tokens"] == body_tokens
    ]
    if len(matches) != 1:
        raise Phase7ContractError(f"missing A8 body{body_tokens} workload")
    spec = matches[0]
    source_header = [32_000 + offset for offset in range(64)]
    body = [1_000 + offset for offset in range(body_tokens)]
    if token_list_sha(source_header) != spec["source_header_token_sha256"]:
        raise Phase7ContractError("A8 source-header token hash mismatch")
    if token_list_sha(body) != spec["body_token_sha256"]:
        raise Phase7ContractError("A8 body token hash mismatch")
    targets = []
    for target_spec in spec["targets"]:
        index = int(target_spec["order"])
        header = [36_000 + index * 128 + offset for offset in range(64)]
        suffix = [49_000 + index]
        prompt = header + body + suffix
        checks = {
            "header_token_sha256": token_list_sha(header),
            "body_token_sha256": token_list_sha(body),
            "suffix_token_sha256": token_list_sha(suffix),
            "prompt_token_sha256": token_list_sha(prompt),
        }
        if any(target_spec[key] != value for key, value in checks.items()):
            raise Phase7ContractError(
                f"{target_spec['target_id']}: A8 token hash mismatch"
            )
        if target_spec["prompt_tokens"] != len(prompt):
            raise Phase7ContractError(
                f"{target_spec['target_id']}: A8 prompt length mismatch"
            )
        targets.append(
            {
                "spec": target_spec,
                "header": header,
                "body": body,
                "suffix": suffix,
                "prompt": prompt,
            }
        )
    if [row["spec"]["order"] for row in targets] != list(range(8)):
        raise Phase7ContractError("A8 target order is not 0..7")
    canary = spec["same_context_canary"]
    if token_list_sha(source_header) != canary["header_token_sha256"]:
        raise Phase7ContractError("A8 canary header hash mismatch")
    segment_tokens_max = observed.get("segment_tokens_max")
    if not isinstance(segment_tokens_max, int) or segment_tokens_max <= 0:
        raise Phase7ContractError("A8 workload does not freeze segment_tokens_max")
    source_pin_until_reset = observed.get("source_pin_until_reset")
    if source_pin_until_reset is not True:
        raise Phase7ContractError("A8 workload does not freeze source_pin_until_reset")
    return {
        "spec": spec,
        "source_header": source_header,
        "body": body,
        "targets": targets,
        "canary_suffix": [59_000 + body_tokens],
        "segment_tokens_max": int(segment_tokens_max),
        "source_pin_until_reset": True,
    }


def filler_pool_tokens(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = build_filler_pool()
    observed = manifest.get("workloads", {}).get("filler_pool")
    if observed != expected:
        raise Phase7ContractError("filler pool differs from frozen builder")
    result = []
    for index, item in enumerate(observed["pool"]):
        tokens = [70_000 + index * 512 + offset for offset in range(512)]
        if len(tokens) != item["tokens"]:
            raise Phase7ContractError(f"{item['filler_id']}: filler length mismatch")
        if token_list_sha(tokens) != item["token_sha256"]:
            raise Phase7ContractError(f"{item['filler_id']}: filler hash mismatch")
        result.append({**item, "token_ids": tokens})
    return result


def select_filler_prefix(
    pool: Sequence[Mapping[str, Any]],
    *,
    capacity_tokens: int,
    rho_logical_demand: float,
    setup_resident_tokens: int,
) -> dict[str, Any]:
    if capacity_tokens <= 0 or rho_logical_demand <= 0:
        raise Phase7ContractError("capacity and rho must be positive")
    target = math.ceil(capacity_tokens * rho_logical_demand)
    needed = max(0, target - setup_resident_tokens)
    selected = []
    selected_tokens = 0
    for item in pool:
        if selected_tokens >= needed:
            break
        selected.append(item)
        selected_tokens += int(item["tokens"])
    if selected_tokens < needed:
        raise Phase7ContractError(
            f"filler pool cannot realize rho: need {needed}, have {selected_tokens}"
        )
    return {
        "capacity_tokens": capacity_tokens,
        "rho_logical_demand": rho_logical_demand,
        "logical_target_tokens": target,
        "setup_resident_tokens": setup_resident_tokens,
        "needed_filler_tokens": needed,
        "selected_filler_tokens": selected_tokens,
        "selected_filler_ids": [str(item["filler_id"]) for item in selected],
        "selected": list(selected),
        "realized_logical_tokens": setup_resident_tokens + selected_tokens,
        "realized_logical_rho": (
            (setup_resident_tokens + selected_tokens) / capacity_tokens
        ),
    }


def required_resident_tokens(snapshot: Mapping[str, float]) -> int:
    used = snapshot.get("sglang:kv_used_tokens")
    evictable = snapshot.get("sglang:kv_evictable_tokens")
    if used is None or evictable is None:
        raise Phase7ContractError(
            "rho realization requires kv_used_tokens and kv_evictable_tokens"
        )
    return int(round(float(used) + float(evictable)))


def w_workload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract_path = REPO_ROOT / (
        "benchmark/approx_kv/results/phase6/p6-0-contract.json"
    )
    p6_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = build_w_workload(p6_contract)
    observed = manifest.get("workloads", {}).get("W")
    if observed != expected:
        raise Phase7ContractError("W workload differs from frozen builder")
    segment_tokens_max = observed.get("segment_tokens_max")
    if segment_tokens_max != 512:
        raise Phase7ContractError("W workload must freeze segment_tokens_max at 512")
    request_order = observed["request_order"]
    if len(request_order) != 61:
        raise Phase7ContractError("W request order must contain 61 requests")
    if [row["request_index"] for row in request_order] != list(range(61)):
        raise Phase7ContractError("W request indexes are not contiguous")
    phases = [row["phase"] for row in request_order]
    if phases[:5] != ["workflow"] * 5:
        raise Phase7ContractError("W workflow prefix is not fixed")
    if phases[5:33] != ["replay"] * 28:
        raise Phase7ContractError("W first replay is not fixed")
    if phases[33:] != ["replay-2"] * 28:
        raise Phase7ContractError("W second replay is not fixed")
    objects = {}
    for item in observed["objects"]:
        body = fixed_object_token_ids(int(item["order"]), int(item["logical_tokens"]))
        if payload_sha256(body) != item["token_ids_sha256"]:
            raise Phase7ContractError(
                f"{item['object_id']}: W object token hash mismatch"
            )
        objects[item["object_id"]] = {
            "spec": item,
            "body": body,
            "source_header": w_header(item, source=True),
            "target_header": w_header(item, source=False),
        }
    return {
        **observed,
        "objects_by_id": objects,
        "request_order_sha256": payload_sha256(request_order),
    }


def w_header(item: Mapping[str, Any], *, source: bool) -> list[int]:
    order = int(item["order"])
    role_salt = sum(ord(character) for character in str(item["role"]))
    if source:
        return [31_000 + ((order * 97 + offset * 13) % 8_000) for offset in range(64)]
    return [
        41_000 + ((order * 89 + role_salt * 7 + offset * 17) % 8_000)
        for offset in range(64)
    ]


def store_gauge_snapshot(snapshot: Mapping[str, float]) -> dict[str, float | None]:
    return {name: snapshot.get(name) for name in STORE_GAUGES}


def memory_footprint(
    snapshot: Mapping[str, float],
    *,
    bytes_per_token: int,
) -> dict[str, Any]:
    """Report non-free resident KV without double counting the two stores.

    ``nonfree_resident_*`` is the pool-level occupancy reported by the exact
    cache gauges (``used`` + ``evictable``). Approximate device-owned bytes
    are carved out of the same pool, so they are reported separately and the
    exact-only remainder is an estimate, not an independent measurement.
    """
    used = snapshot.get("sglang:kv_used_tokens")
    evictable = snapshot.get("sglang:kv_evictable_tokens")
    nonfree_tokens = (
        None if used is None or evictable is None else float(used) + float(evictable)
    )
    nonfree_bytes = None if nonfree_tokens is None else nonfree_tokens * bytes_per_token
    approx_device_bytes = snapshot.get("sglang:approx_kv_store_device_bytes")
    exact_only_estimated_bytes = (
        None
        if nonfree_bytes is None or approx_device_bytes is None
        else max(nonfree_bytes - float(approx_device_bytes), 0.0)
    )
    return {
        "nonfree_resident_tokens": nonfree_tokens,
        "nonfree_resident_bytes": nonfree_bytes,
        "approx_device_bytes": approx_device_bytes,
        "approx_host_bytes": snapshot.get("sglang:approx_kv_store_host_bytes"),
        "exact_only_estimated_bytes": exact_only_estimated_bytes,
        "reserved_device_bytes": snapshot.get(
            "sglang:cross_store_reserved_device_bytes"
        ),
        "arm_interval_peak_device_bytes": snapshot.get(
            "sglang:cross_store_peak_device_bytes"
        ),
        "overlap_note": (
            "nonfree_resident_bytes already contains approx_device_bytes; "
            "exact_only_estimated_bytes = max(nonfree - approx_device, 0) and "
            "must not be added back to approx_device_bytes"
        ),
        "peak_semantics": "arm_high_water_since_last_full_reset",
    }


def phase7_reset_invariant(
    snapshot: Mapping[str, float],
    *,
    strict: bool,
    clean_baseline: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Check that a full reset returned every Phase 7 store to empty.

    ``strict=True`` requires every gauge to be exported, including
    ``cross_store_reserved_device_bytes``. ``strict=False`` is only for
    resets that precede any cross-store activity (server startup and the
    exact-only D0/E0 arms), where a not-yet-created series is recorded as
    ``not_yet_exported`` instead of being read as an explicit zero.
    """
    exact = clean_cache_invariant(snapshot)
    baseline = (
        None
        if clean_baseline is None
        else clean_pool_reset_invariant(clean_baseline, snapshot)
    )
    gauges = store_gauge_snapshot(snapshot)
    not_yet_exported = (
        []
        if strict
        else [name for name in LATE_EXPORTED_GAUGES if gauges.get(name) is None]
    )

    def zero(name: str) -> bool:
        if name in not_yet_exported:
            return True
        value = gauges.get(name)
        return value is not None and float(value) == 0.0

    components = {
        "exact": bool(exact.get("passed"))
        and (baseline is None or bool(baseline.get("passed"))),
        "approximate": all(
            zero(name)
            for name in (
                "sglang:approx_kv_store_records",
                "sglang:approx_kv_store_device_bytes",
                "sglang:approx_kv_store_host_bytes",
            )
        ),
        "metadata": zero("sglang:approx_kv_store_records"),
        "reserved": zero("sglang:cross_store_reserved_device_bytes"),
        "provisional": zero("sglang:approx_kv_provisional_tokens"),
        "leases": zero("sglang:approx_kv_store_leases"),
        "orphans": zero("sglang:approx_kv_store_orphans"),
    }
    missing = [
        name
        for name, value in gauges.items()
        if value is None and name not in not_yet_exported
    ]
    return {
        "passed": all(components.values()) and not missing,
        "strict": strict,
        "components": components,
        "gauge_states": {
            name: (
                "not_yet_exported"
                if name in not_yet_exported
                else ("missing" if value is None else "exported")
            )
            for name, value in gauges.items()
        },
        "not_yet_exported": not_yet_exported,
        "missing_gauges": missing,
        "exact_cache": exact,
        "clean_baseline_delta": baseline,
        "store_gauges": gauges,
    }


def parse_labeled_samples(
    text: str,
    name: str,
) -> list[tuple[dict[str, str], float]]:
    samples = []
    for raw_line in text.splitlines():
        match = _SAMPLE_RE.match(raw_line.strip())
        if match is None or match.group("name") != name:
            continue
        labels = {
            key: bytes(value, "utf-8").decode("unicode_escape")
            for key, value in _LABEL_RE.findall(match.group("labels") or "")
        }
        samples.append((labels, float(match.group("value"))))
    return samples


def labeled_counter_observation(
    before: str,
    after: str,
    *,
    name: str,
    required_labels: Mapping[str, str],
    indirect_evidence: str | None = None,
) -> dict[str, Any]:
    samples = parse_labeled_samples(before, name) + parse_labeled_samples(after, name)
    matching_series_available = any(
        all(labels.get(key) == value for key, value in required_labels.items())
        for labels, _ in samples
    )
    if not matching_series_available:
        return {
            "verification": ("indirectly_verified" if indirect_evidence else "unknown"),
            "value": None,
            "indirect_evidence": indirect_evidence,
            "metric": name,
            "labels": dict(required_labels),
        }
    value = labeled_metric_delta(
        before,
        after,
        name,
        dict(required_labels),
    )
    if value < 0:
        raise Phase7ContractError(f"{name} counter decreased")
    return {
        "verification": "direct",
        "value": value,
        "indirect_evidence": None,
        "metric": name,
        "labels": dict(required_labels),
    }


def labeled_breakdown_delta(
    before: str,
    after: str,
    *,
    name: str,
    label_names: Sequence[str],
) -> dict[str, Any]:
    before_samples = parse_labeled_samples(before, name)
    after_samples = parse_labeled_samples(after, name)
    if not before_samples and not after_samples:
        return {"verification": "unknown", "rows": []}
    keys = {
        tuple(labels.get(label, "") for label in label_names)
        for labels, _ in (*before_samples, *after_samples)
    }
    rows = []
    for key in sorted(keys):
        labels = dict(zip(label_names, key))
        value = labeled_metric_delta(before, after, name, labels)
        if value < 0:
            raise Phase7ContractError(f"{name} counter decreased")
        if value:
            rows.append({**labels, "bytes_or_count": value})
    return {"verification": "direct", "rows": rows}


def counter_observation(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    name: str,
    indirect_evidence: str | None = None,
) -> dict[str, Any]:
    value = counter_delta(before, after, name)
    if value is None:
        return {
            "verification": ("indirectly_verified" if indirect_evidence else "unknown"),
            "value": None,
            "indirect_evidence": indirect_evidence,
            "metric": name,
        }
    if value < 0:
        raise Phase7ContractError(f"{name} counter decreased")
    return {
        "verification": "direct",
        "value": value,
        "indirect_evidence": None,
        "metric": name,
    }


def text_counter_observation(
    before: str,
    after: str,
    *,
    name: str,
    indirect_evidence: str | None = None,
) -> dict[str, Any]:
    before_samples = parse_labeled_samples(before, name)
    after_samples = parse_labeled_samples(after, name)
    if not before_samples and not after_samples:
        return {
            "verification": ("indirectly_verified" if indirect_evidence else "unknown"),
            "value": None,
            "indirect_evidence": indirect_evidence,
            "metric": name,
        }
    value = sum(sample for _, sample in after_samples) - sum(
        sample for _, sample in before_samples
    )
    if value < 0:
        raise Phase7ContractError(f"{name} counter decreased")
    return {
        "verification": "direct",
        "value": value,
        "indirect_evidence": None,
        "metric": name,
    }


def request_outcome_observations(
    before: str,
    after: str,
    *,
    operation: str,
) -> dict[str, Any]:
    outcomes = ("success", "dense_fallback", "exact", "exact_host_preferred")
    observations = {
        outcome: labeled_counter_observation(
            before,
            after,
            name="sglang:approx_kv_requests_total",
            required_labels={"operation": operation, "outcome": outcome},
        )
        for outcome in outcomes
    }
    return {
        "verification": (
            "direct"
            if any(row["verification"] == "direct" for row in observations.values())
            else "unknown"
        ),
        "outcomes": observations,
    }


def registration_outcome_observations(
    before: str,
    after: str,
) -> dict[str, Any]:
    rows = {
        outcome: labeled_counter_observation(
            before,
            after,
            name="sglang:approx_kv_requests_total",
            required_labels={"operation": "register", "outcome": outcome},
        )
        for outcome in ("success", "partial", "dense_only", "error")
    }
    positives = {
        name: float(row["value"])
        for name, row in rows.items()
        if row["value"] is not None and float(row["value"]) > 0
    }
    verification = (
        "direct"
        if any(row["verification"] == "direct" for row in rows.values())
        else "unknown"
    )
    return {
        "outcomes": rows,
        "positive_outcomes": positives,
        "registration_failed": (
            verification == "direct"
            and (len(positives) != 1 or "success" not in positives)
        ),
        "verification": verification,
    }


def terminal_reason_observations(before: str, after: str) -> dict[str, Any]:
    """Map raw dense-fallback reasons onto the frozen terminal taxonomy.

    Values are token counts (``sglang:approx_kv_dense_fallback_total`` is
    incremented by the number of tokens that fell back). Only whitelisted raw
    reasons are mapped; anything else is reported verbatim under
    ``unmapped_raw_reasons`` instead of being collapsed into ``unsupported``.
    """
    samples = parse_labeled_samples(
        before, "sglang:approx_kv_dense_fallback_total"
    ) + parse_labeled_samples(after, "sglang:approx_kv_dense_fallback_total")
    if not samples:
        return {
            "verification": "unknown",
            "value_unit": "tokens",
            "mapped": {reason: None for reason in TERMINAL_REASONS},
            "mapped_from": {reason: [] for reason in TERMINAL_REASONS},
            "unmapped_raw_reasons": {},
            "excluded_raw_reasons": {},
            "raw": {},
        }
    raw_reasons = sorted(
        {labels.get("reason", "") for labels, _ in samples if labels.get("reason")}
    )
    raw = {}
    mapped: dict[str, float | None] = {reason: None for reason in TERMINAL_REASONS}
    mapped_from: dict[str, list[str]] = {reason: [] for reason in TERMINAL_REASONS}
    unmapped: dict[str, float] = {}
    excluded: dict[str, float] = {}
    for reason in raw_reasons:
        observation = labeled_counter_observation(
            before,
            after,
            name="sglang:approx_kv_dense_fallback_total",
            required_labels={"reason": reason},
        )
        value = float(observation["value"] or 0.0)
        raw[reason] = value
        if value <= 0:
            continue
        if reason in EXCLUDED_RAW_TERMINAL_REASONS:
            excluded[reason] = value
            continue
        normalized = _RAW_TERMINAL_REASON_MAP.get(reason)
        if normalized is None:
            unmapped[reason] = value
            continue
        mapped[normalized] = float(mapped[normalized] or 0.0) + value
        mapped_from[normalized].append(reason)
    return {
        "verification": "direct",
        "value_unit": "tokens",
        "mapped": mapped,
        "mapped_from": mapped_from,
        "unmapped_raw_reasons": unmapped,
        "excluded_raw_reasons": excluded,
        "raw": raw,
    }


def classify_request_outcome(
    *,
    arm: str,
    cached_tokens: int,
    expected_cached_tokens: int,
    request_observations: Mapping[str, Any] | None,
    terminal_observations: Mapping[str, Any] | None,
    registration_failed: bool = False,
    expected_outcomes: Sequence[str] | None = None,
) -> dict[str, Any]:
    outcome_verification = "direct"
    terminal_reason = None
    terminal_verification = "not_applicable"
    ambiguity = None

    if arm == "D0":
        outcome = "dense_no_reuse_baseline"
    elif arm == "E0":
        outcome = (
            "exact_gpu_hit"
            if cached_tokens >= expected_cached_tokens
            else "ordinary_exact_cache_miss"
        )
        outcome_verification = "indirectly_verified"
    elif arm in {"R0", "R4-like-5x"}:
        direct_values: dict[str, float] = {}
        if request_observations is not None:
            for name, observation in request_observations.get("outcomes", {}).items():
                value = observation.get("value")
                if value is not None and float(value) > 0:
                    direct_values[name] = float(value)
        if len(direct_values) > 1:
            ambiguity = f"multiple request outcomes: {sorted(direct_values)}"
        direct_name = next(iter(direct_values), None)
        if direct_name == "success":
            outcome = "approximate_gpu_recovery"
        elif direct_name == "dense_fallback":
            outcome = "approximate_recovery_failed_dense"
        elif direct_name == "exact":
            outcome = "exact_gpu_hit"
        elif direct_name == "exact_host_preferred":
            outcome = "host_demand_load"
        elif registration_failed:
            outcome = "approximate_recovery_failed_dense"
            outcome_verification = "indirectly_verified"
        else:
            outcome = (
                "approximate_gpu_recovery"
                if cached_tokens >= expected_cached_tokens
                else "approximate_recovery_failed_dense"
            )
            outcome_verification = "indirectly_verified"

        if outcome == "approximate_recovery_failed_dense":
            positives = {}
            unmapped = {}
            if terminal_observations is not None:
                positives = {
                    reason: float(value)
                    for reason, value in terminal_observations.get("mapped", {}).items()
                    if value is not None and float(value) > 0
                }
                unmapped = dict(
                    terminal_observations.get("unmapped_raw_reasons", {}) or {}
                )
            if unmapped:
                ambiguity = f"unmapped terminal reasons: {sorted(unmapped)}"
                terminal_verification = "unknown"
            elif len(positives) == 1:
                terminal_reason = next(iter(positives))
                terminal_verification = "direct"
            elif len(positives) > 1:
                ambiguity = f"multiple terminal reasons: {sorted(positives)}"
                terminal_verification = "unknown"
            elif registration_failed:
                terminal_reason = "registration_failed"
                terminal_verification = "direct"
            else:
                terminal_verification = "unknown"
    else:
        raise Phase7ContractError(f"unsupported arm {arm!r}")

    record = {
        "outcome": outcome,
        "terminal_reason": terminal_reason,
        "outcome_verification": outcome_verification,
        "terminal_reason_verification": terminal_verification,
        "ambiguity": ambiguity,
    }
    record["taxonomy_valid"] = validate_outcome_record(record)
    allowed_expected = (
        tuple(expected_outcomes)
        if expected_outcomes is not None
        else {
            "D0": ("dense_no_reuse_baseline",),
            "E0": ("exact_gpu_hit",),
            "R0": ("approximate_gpu_recovery",),
            "R4-like-5x": (
                "approximate_gpu_recovery",
                "exact_gpu_hit",
                "approximate_recovery_failed_dense",
            ),
        }[arm]
    )
    record["expected_outcome"] = (
        record["taxonomy_valid"] and outcome in allowed_expected
    )
    return record


def validate_outcome_record(record: Mapping[str, Any]) -> bool:
    outcome = record.get("outcome")
    reason = record.get("terminal_reason")
    if outcome not in OUTCOME_TAXONOMY:
        return False
    if reason is not None and reason not in TERMINAL_REASONS:
        return False
    if outcome == "approximate_recovery_failed_dense":
        return reason is not None and record.get("ambiguity") is None
    return reason is None and record.get("ambiguity") is None


def cross_store_metrics(
    *,
    before_text: str,
    after_text: str,
    before_snapshot: Mapping[str, float],
    after_snapshot: Mapping[str, float],
) -> dict[str, Any]:
    inactive_evidence = (
        "host budget and prefetch/async tracks are pinned disabled in the manifest"
    )
    evicted = labeled_breakdown_delta(
        before_text,
        after_text,
        name="sglang:cross_store_evicted_bytes_total",
        label_names=("requester", "provenance", "object_kind"),
    )
    demoted = labeled_breakdown_delta(
        before_text,
        after_text,
        name="sglang:cross_store_demoted_bytes_total",
        label_names=("requester", "provenance", "object_kind"),
    )
    wasted = counter_observation(
        before_snapshot,
        after_snapshot,
        name="sglang:cross_store_wasted_bytes_total",
    )
    unobserved_churn = [
        name
        for name, observation in (
            ("victim_evict_bytes", evicted),
            ("demote_bytes", demoted),
        )
        if observation["verification"] != "direct"
    ]
    return {
        "arm_interval_peak_device_bytes": (
            after_snapshot.get("sglang:cross_store_peak_device_bytes")
        ),
        "peak_semantics": "arm_high_water_since_last_full_reset",
        "victim_evict_bytes": evicted,
        "demote_bytes": demoted,
        "reservation_failures": labeled_breakdown_delta(
            before_text,
            after_text,
            name="sglang:cross_store_reservation_failures_total",
            label_names=("requires_reset",),
        ),
        "wasted_bytes": {
            **wasted,
            "wasted_is_subset_of_evicted": True,
            "definition": (
                "irreversibly destroyed bytes from failed allocations; already "
                "counted inside cross_store_evicted_bytes_total"
            ),
        },
        "churn_bytes": {
            "definition": "evicted_bytes + demoted_bytes (wasted is excluded)",
            "verification": (
                "direct"
                if not unobserved_churn
                else ("unknown" if len(unobserved_churn) == 2 else "partially_direct")
            ),
            "unobserved_components": unobserved_churn,
            "value": (
                None
                if len(unobserved_churn) == 2
                else (
                    (
                        0.0
                        if "victim_evict_bytes" in unobserved_churn
                        else sum(row["bytes_or_count"] for row in evicted["rows"])
                    )
                    + (
                        0.0
                        if "demote_bytes" in unobserved_churn
                        else sum(row["bytes_or_count"] for row in demoted["rows"])
                    )
                )
            ),
        },
        "transfer": {
            "copied_tokens": text_counter_observation(
                before_text,
                after_text,
                name="sglang:approx_kv_copied_tokens_total",
            ),
            "h2d_tokens": text_counter_observation(
                before_text,
                after_text,
                name="sglang:approx_kv_h2d_tokens_total",
                indirect_evidence=inactive_evidence,
            ),
            "h2d_bytes": text_counter_observation(
                before_text,
                after_text,
                name="sglang:approx_kv_h2d_bytes_total",
                indirect_evidence=inactive_evidence,
            ),
            "h2d_duration_seconds": text_counter_observation(
                before_text,
                after_text,
                name="sglang:approx_kv_h2d_duration_seconds_sum",
                indirect_evidence=inactive_evidence,
            ),
            "host_export_bytes": text_counter_observation(
                before_text,
                after_text,
                name="sglang:approx_kv_host_export_bytes_total",
                indirect_evidence=inactive_evidence,
            ),
            "host_export_duration_seconds": text_counter_observation(
                before_text,
                after_text,
                name="sglang:approx_kv_host_export_duration_seconds_sum",
                indirect_evidence=inactive_evidence,
            ),
        },
        "store_gauges_after": store_gauge_snapshot(after_snapshot),
        "inactive_tracks": {
            "host_load": text_counter_observation(
                before_text,
                after_text,
                name="sglang:approx_kv_h2d_tokens_total",
                indirect_evidence=inactive_evidence,
            ),
            "prefetch_request": text_counter_observation(
                before_text,
                after_text,
                name="sglang:workflow_prefetch_requests_total",
                indirect_evidence=inactive_evidence,
            ),
            "prefetch_loaded_tokens": text_counter_observation(
                before_text,
                after_text,
                name="sglang:workflow_prefetch_loaded_tokens_total",
                indirect_evidence=inactive_evidence,
            ),
            "async_load": text_counter_observation(
                before_text,
                after_text,
                name="sglang:approx_kv_h2d_duration_seconds_count",
                indirect_evidence=inactive_evidence,
            ),
        },
    }


def observed_capacity(
    snapshot: Mapping[str, float], bytes_per_token: int
) -> dict[str, Any]:
    tokens = max_total_num_tokens(snapshot)
    return {
        "tokens": tokens,
        "pages": None,
        "bytes": tokens * bytes_per_token,
    }


def rho_payload() -> dict[str, Any]:
    definitions = RhoDefinitions()
    return {
        "definitions": {
            "logical_demand": definitions.logical_demand,
            "physical_demand": definitions.physical_demand,
            "resident": definitions.resident,
            "host": definitions.host,
        }
    }


def finalize_artifact_hash(payload: dict[str, Any]) -> None:
    payload.pop("raw_sha256", None)
    payload["raw_sha256"] = payload_sha256(payload)


def validate_phase7_artifact(payload: dict[str, Any]) -> None:
    validate_phase6_artifact(payload)
    required = (
        "manifest_revision",
        "preregistered_manifest_sha256",
        "manifest_file_sha256",
        "plan",
        "setting_id",
        "restart_index",
        "runner",
        "outcome",
        "reset",
        "provenance",
        "server_log_path",
        "server_log_sha256",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise Phase7ContractError(f"missing Phase 7 artifact fields: {missing}")
    if payload["preregistered_manifest_sha256"] is None:
        raise Phase7ContractError("artifact does not bind the manifest")
    stored_raw_sha = payload.get("raw_sha256")
    canonical = dict(payload)
    canonical.pop("raw_sha256", None)
    if stored_raw_sha != payload_sha256(canonical):
        raise Phase7ContractError("Phase 7 raw artifact hash mismatch")
    if payload["server_log_sha256"] is not None:
        _require_sha(payload["server_log_sha256"], field="server_log_sha256")


def ensure_new_artifact_paths(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("artifact output paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing paths: {existing}")


def ensure_artifact_path_layout(
    *,
    output: Path,
    log: Path,
    central_log: Path,
) -> None:
    resolved = (output.resolve(), log.resolve(), central_log.resolve())
    if len(set(resolved)) != len(resolved):
        raise ValueError("output, server log, and central log must be distinct")
    ensure_new_artifact_paths(output, log)
