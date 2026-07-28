#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("benchmark/approx_kv/results/phase7/phase7-primary-manifest.json")
P6_CONTRACT = Path("benchmark/approx_kv/results/phase6/p6-0-contract.json")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def payload_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("preregistered_manifest_sha256", None)
    return sha256_bytes(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    )


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def token_list_sha(tokens: list[int]) -> str:
    return sha256_bytes(json.dumps(tokens, separators=(",", ":")).encode("utf-8"))


def build_a8_workload() -> dict[str, Any]:
    workloads = []
    for body_tokens in (1024, 2048):
        source_header = [32_000 + offset for offset in range(64)]
        body = [1_000 + offset for offset in range(body_tokens)]
        targets = []
        for target_index in range(8):
            target_header = [
                36_000 + target_index * 128 + offset for offset in range(64)
            ]
            suffix = [49_000 + target_index]
            prompt = target_header + body + suffix
            targets.append(
                {
                    "target_id": f"a8-b{body_tokens}-target-{target_index}",
                    "order": target_index,
                    "extra_key": (f"p7-a8-b{body_tokens}-target-{target_index}"),
                    "header_token_sha256": token_list_sha(target_header),
                    "body_token_sha256": token_list_sha(body),
                    "suffix_token_sha256": token_list_sha(suffix),
                    "prompt_token_sha256": token_list_sha(prompt),
                    "prompt_tokens": len(prompt),
                }
            )
        workloads.append(
            {
                "workload_id": f"A8-body{body_tokens}",
                "body_tokens": body_tokens,
                "source_header_token_sha256": token_list_sha(source_header),
                "body_token_sha256": token_list_sha(body),
                "source_object_pinned_for_sequence": True,
                "dense_source_materialization": "same_source_register_false",
                "targets": targets,
            }
        )
    payload = {
        "schema_version": 1,
        "prompt_family_version": "p7-a8-v1",
        "targets_per_setup": 8,
        "workloads": workloads,
    }
    payload["manifest_sha256"] = payload_sha256(payload)
    return payload


def build_filler_pool() -> dict[str, Any]:
    fillers = []
    for filler_index in range(64):
        tokens = [70_000 + filler_index * 512 + offset for offset in range(512)]
        fillers.append(
            {
                "filler_id": f"p7-filler-{filler_index:02d}",
                "tokens": 512,
                "token_sha256": token_list_sha(tokens),
                "retired": filler_index % 3 == 0,
            }
        )
    payload = {
        "schema_version": 1,
        "selection_rule": (
            "select the shortest deterministic prefix whose logical token "
            "sum reaches the pre-registered rho target after subtracting "
            "setup used+evictable; record the selected IDs at runtime"
        ),
        "pool": fillers,
    }
    payload["manifest_sha256"] = payload_sha256(payload)
    return payload


def build_w_workload(p6_contract: dict[str, Any]) -> dict[str, Any]:
    workload = p6_contract["workload"]
    objects = [dict(item) for item in workload["objects"]]
    active_by_role: dict[str, list[str]] = {}
    for item in objects:
        if item["active"]:
            active_by_role.setdefault(item["role"], []).append(item["object_id"])

    cursor = {role: 0 for role in active_by_role}
    workflow_requests = []
    for request_index, role in enumerate(workload["workflow_sequence"]):
        choices = active_by_role[role]
        object_id = choices[cursor[role] % len(choices)]
        cursor[role] += 1
        workflow_requests.append(
            {
                "request_index": request_index,
                "phase": "workflow",
                "role": role,
                "object_id": object_id,
            }
        )

    replay = [
        {
            "request_index": len(workflow_requests) + index,
            "phase": "replay",
            "role": item["role"],
            "object_id": item["object_id"],
        }
        for index, item in enumerate(objects)
        if item["active"]
    ]
    request_order = workflow_requests + replay
    for index, request in enumerate(request_order):
        later = [
            row["request_index"]
            for row in request_order[index + 1 :]
            if row["object_id"] == request["object_id"]
        ]
        request["next_use_request_index"] = later[0] if later else None

    payload = {
        "schema_version": 1,
        "workload_id": "W-fixed40-v1",
        "source_workload_sha256": workload["manifest_sha256"],
        "objects": objects,
        "workflow_sequence": workload["workflow_sequence"],
        "fill_order": [item["object_id"] for item in objects],
        "request_order": request_order,
    }
    payload["manifest_sha256"] = payload_sha256(payload)
    return payload


def setting(
    setting_id: str,
    *,
    wave: str,
    runner: str,
    workload: str,
    body: int,
    rho: float,
    chunk: int,
    policy: str,
    restarts: list[int],
    arms: list[str],
    conditional: bool = False,
    max_total_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "setting_id": setting_id,
        "wave": wave,
        "runner": runner,
        "workload": workload,
        "body_tokens": body,
        "rho_logical_demand": rho,
        "chunked_prefill_size": chunk,
        "max_prefill_tokens": chunk,
        "policy": policy,
        "restart_indices": restarts,
        "warmup_repeats": 1,
        "formal_repeats": 2,
        "arms": arms,
        "conditional": conditional,
        "max_total_tokens": max_total_tokens,
    }


def build_settings() -> list[dict[str, Any]]:
    settings = [
        setting(
            "p6delta-s4-rho2-chunk4096",
            wave="wave-0",
            runner="benchmark.approx_kv.run_p6_4_capacity_pilot",
            workload="W-fixed40-v1",
            body=2048,
            rho=2.0,
            chunk=4096,
            policy="hierarchical",
            restarts=[0],
            arms=[
                "exact_only",
                "r0_like",
                "r1_like_k32",
                "r2_like",
                "r4_like",
            ],
            max_total_tokens=11392,
        ),
        setting(
            "p6delta-s0-rho2-chunk4096",
            wave="wave-0",
            runner="benchmark.approx_kv.run_p6_4_capacity_pilot",
            workload="W-fixed40-v1",
            body=2048,
            rho=2.0,
            chunk=4096,
            policy="lru",
            restarts=[0],
            arms=[
                "exact_only",
                "r0_like",
                "r1_like_k32",
                "r2_like",
                "r4_like",
            ],
            max_total_tokens=11392,
        ),
        setting(
            "p6delta-s4-rho3-chunk4096",
            wave="conditional",
            runner="benchmark.approx_kv.run_p6_4_capacity_pilot",
            workload="W-fixed40-v1",
            body=2048,
            rho=3.0,
            chunk=4096,
            policy="hierarchical",
            restarts=[0],
            arms=[
                "exact_only",
                "r0_like",
                "r1_like_k32",
                "r2_like",
                "r4_like",
            ],
            conditional=True,
            max_total_tokens=7595,
        ),
    ]
    for body in (1024, 2048):
        for rho in (1.5, 2.0):
            settings.append(
                setting(
                    f"p7-a8-r0-body{body}-rho{rho}",
                    wave="wave-1/wave-2",
                    runner="benchmark.approx_kv.run_p7_ceiling",
                    workload=f"A8-body{body}",
                    body=body,
                    rho=rho,
                    chunk=4096,
                    policy="lru",
                    restarts=[0, 1, 2],
                    arms=["D0", "E0", "R0"],
                )
            )
    settings.append(
        setting(
            "p7-a8-r0-body2048-rho2-chunk1024-sensitivity",
            wave="wave-2",
            runner="benchmark.approx_kv.run_p7_ceiling",
            workload="A8-body2048",
            body=2048,
            rho=2.0,
            chunk=1024,
            policy="lru",
            restarts=[0, 1],
            arms=["D0", "E0", "R0"],
        )
    )
    for rho in (1.5, 2.0):
        for policy in ("lru", "hierarchical"):
            settings.append(
                setting(
                    f"p7-w-r0-{policy}-rho{rho}",
                    wave="wave-2",
                    runner="benchmark.approx_kv.run_p7_scheduler",
                    workload="W-fixed40-v1",
                    body=2048,
                    rho=rho,
                    chunk=4096,
                    policy=policy,
                    restarts=[0, 1, 2],
                    arms=["E0", "R0"],
                )
            )
    for policy in ("lru", "hierarchical"):
        settings.append(
            setting(
                f"p7-w-r4like-{policy}-rho2",
                wave="wave-2",
                runner="benchmark.approx_kv.run_p7_scheduler",
                workload="W-fixed40-v1",
                body=2048,
                rho=2.0,
                chunk=4096,
                policy=policy,
                restarts=[0],
                arms=["R4-like-5x"],
            )
        )
    for rho in (1.5, 2.0):
        settings.append(
            setting(
                f"p7-a8-r2-rho{rho}",
                wave="conditional",
                runner="benchmark.approx_kv.run_p7_ceiling",
                workload="A8-body2048",
                body=2048,
                rho=rho,
                chunk=4096,
                policy="lru",
                restarts=[0],
                arms=["D0", "E0", "R2"],
                conditional=True,
            )
        )
    return settings


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    plan_repo = args.plan_repo.resolve()
    plan_path = args.plan_path.resolve()
    implementation_sha = git("rev-parse", "HEAD")
    implementation_tree = git("rev-parse", "HEAD^{tree}")
    plan_blob = subprocess.run(
        ("git", "show", f"{args.plan_commit}:{args.plan_file_in_repo}"),
        cwd=plan_repo,
        capture_output=True,
        check=True,
    ).stdout
    if sha256_bytes(plan_blob) != sha256_file(plan_path):
        raise RuntimeError("plan path does not match the pinned plan commit")

    p6_contract = json.loads(P6_CONTRACT.read_text())
    settings = build_settings()
    committed = [row for row in settings if not row["conditional"]]
    conditional = [row for row in settings if row["conditional"]]
    committed_starts = sum(len(row["restart_indices"]) for row in committed)
    conditional_starts = sum(len(row["restart_indices"]) for row in conditional)
    runner_paths = {
        "ceiling": "benchmark/approx_kv/run_p7_ceiling.py",
        "scheduler": "benchmark/approx_kv/run_p7_scheduler.py",
        "manifest": "benchmark/approx_kv/build_phase7_manifest.py",
    }
    runner_status = {
        name: {
            "path": path,
            "exists": Path(path).exists(),
            "sha256": sha256_file(Path(path)) if Path(path).exists() else None,
        }
        for name, path in runner_paths.items()
    }
    execution_blockers = [
        name
        for name, status in runner_status.items()
        if name != "manifest" and not status["exists"]
    ]
    execution_blockers.extend(
        [
            "phase7_pinned_implementation_sha_pending_runner_implementation",
            "r2_strategy_pending_runner_decision",
            "explicit_user_authorization_missing",
        ]
    )

    manifest = {
        "schema_version": 1,
        "artifact": "phase7-primary-manifest",
        "status": "preregistered_blocked",
        "phase7_execution_authorized": False,
        "plan": {
            "version": "V5",
            "plan_commit": args.plan_commit,
            "plan_file": args.plan_file_in_repo,
            "plan_sha256": sha256_bytes(plan_blob),
        },
        "implementation": {
            "manifest_generation_sha": implementation_sha,
            "manifest_generation_tree_sha": implementation_tree,
            "phase6_evidence_sha": args.phase6_evidence_sha,
            "phase7_pinned_implementation_sha": None,
            "phase7_pinned_tree_sha": None,
            "atomic_update_required": [
                "phase7-primary-manifest.json",
                "HANDOFF.md",
                "TODO_LOCAL.txt",
            ],
        },
        "runners": runner_status,
        "execution_blockers": execution_blockers,
        "environment": {
            "image_digest": args.image_digest,
            "model": "Qwen/Qwen3-0.6B",
            "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "chat_template_revision": "model-revision-bound",
            "gpu": "NVIDIA GeForce RTX 2080 SUPER, SM75, 8192 MiB",
            "driver": "580.173.02",
        },
        "server_template": {
            "tp_size": 1,
            "mem_fraction_static": 0.35,
            "max_running_requests": 2,
            "attention_backend": "torch_native",
            "sampling_backend": "pytorch",
            "cuda_graph_decode": "disabled",
            "cuda_graph_prefill": "disabled",
            "enable_cache_report": True,
            "enable_metrics": True,
            "restart_seeds": [17, 18, 19],
            "test_only_injection_flags": {
                "SGLANG_APPROX_KV_TEST_ONLY": "0",
                "SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE": "0",
            },
        },
        "workloads": {
            "A8": build_a8_workload(),
            "W": build_w_workload(p6_contract),
            "filler_pool": build_filler_pool(),
        },
        "arms": {
            "D0": "dense_no_reuse_baseline",
            "E0": "exact_cache",
            "R0": "raw_copy_ceiling",
            "R2": "conditional_cross_store_adapter_or_not_comparable",
            "R4-like-5x": "synthetic_footprint_proxy_not_kvcomm",
        },
        "statistics": {
            "process_level_timing_replicates": [0, 1, 2],
            "primary_estimator": "per_restart_paired_target_median",
            "range": "report_all_three_values_and_min_max",
            "p95": "per_restart_then_median_across_restarts",
            "pooled_p95_name": "ratio_of_marginal_p95s",
            "request_level_bootstrap_forbidden": True,
            "mde": {
                "source": "CL2 body768 chunk4096 boundary-free control",
                "speedups": [
                    1.005757470603157,
                    1.0043539446596337,
                    1.0100547151467836,
                    1.0027735443627668,
                ],
                "mean": 1.0057349186930853,
                "sample_sd": 0.003127191348764852,
                "two_sample_sd": 0.006254382697529704,
                "mde_fraction": 0.05,
            },
            "amortization_N": [1, 2, 4, 8],
            "break_even": [
                "full_setup_break_even_observed_N",
                "incremental_setup_break_even_observed_N",
            ],
            "not_observed_value": ">8/not_observed",
        },
        "outcome_taxonomy": [
            "dense_no_reuse_baseline",
            "exact_gpu_hit",
            "ordinary_exact_cache_miss",
            "approximate_gpu_recovery",
            "host_demand_load",
            "approximate_recovery_failed_dense",
        ],
        "exclusive_terminal_reasons": [
            "cross_store_reservation_failed",
            "device_allocation_failed",
            "unsupported",
            "registration_failed",
            "prefix_gap",
        ],
        "settings": settings,
        "early_stops": [
            {
                "rule_id": "ES-ENGINEERING",
                "checkpoint": "immediate",
                "rule": (
                    "unexpected primary OOM, incomplete request, stale handle, "
                    "double free, accounting/reset/orphan failure => INVALID"
                ),
            },
            {
                "rule_id": "ES-CAPACITY",
                "checkpoint": "cell completion",
                "rule": (
                    "R4/P6-4Delta OOM with <=0.05s death-state snapshot and "
                    "empty reclaimable store => DIAGNOSTIC_UNAVAILABLE"
                ),
            },
            {
                "rule_id": "ES-R0-MDE",
                "checkpoint": "complete restart-0 A8 matrix",
                "rule": (
                    "if paired median improvement <5% or does not exceed the "
                    "frozen MDE, record NEGATIVE/INCONCLUSIVE and skip "
                    "restart1-2 supplements"
                ),
            },
            {
                "rule_id": "ES-CHUNK-HEADLINE",
                "checkpoint": "predeclared from CL2",
                "rule": (
                    "do not publish mechanism-intrinsic speedup; all speedup "
                    "reporting is chunk-coupled"
                ),
            },
            {
                "rule_id": "ES-S4",
                "checkpoint": "complete restart-0 W matrix",
                "rule": (
                    "if all-reusable mean gain <5% and "
                    "miss_S4>=miss_S0 and peak_S4>=peak_S0, stop scheduler "
                    "benefit claim"
                ),
            },
            {
                "rule_id": "ES-NATURAL-FALLBACK",
                "checkpoint": "any natural approximate reservation failure",
                "rule": (
                    "failure to complete dense fallback after entering "
                    "approximate recovery => INVALID"
                ),
            },
        ],
        "budget": {
            "committed_gpu_settings": 13,
            "committed_server_starts": committed_starts,
            "conditional_gpu_settings": 3,
            "conditional_server_starts": conditional_starts,
            "all_gpu_settings": 16,
            "all_server_starts": committed_starts + conditional_starts,
            "hard_cap_server_starts": 36,
            "hard_cap_gpu_hours": 6,
            "retries_count_against_cap": True,
        },
        "skipped_tracks": [
            "practical_recovery",
            "practical_scheduler_revalidation",
            "hicache_adapter",
            "host_matrix",
            "prefetch_functionality",
            "prefetch_performance",
            "async_h2d_performance",
        ],
        "required_inactive_counters": [
            "host_load",
            "prefetch_request",
            "prefetch_loaded_tokens",
            "async_load",
        ],
        "scope_caveats": [
            "practical=NONE is scoped to the tested implementation and chunk1024 qualification",
            "R0 is a ceiling, not a practical candidate",
            "R2 is conditional and otherwise not_comparable historical evidence",
            "R4-like is a synthetic footprint proxy, not KVCOMM execution",
            "P6-F verifies fault-injected fallback only; natural pressure reachability is unproven",
            "P6-4 rho1.1/rho1.5/rho3 feasibility remains chunk1024 unless separately revalidated",
            "host/prefetch/async tracks are not generated in V5",
        ],
    }
    manifest["preregistered_manifest_sha256"] = payload_sha256(manifest)
    return manifest


def validate(manifest: dict[str, Any], args: argparse.Namespace) -> list[str]:
    problems: list[str] = []
    if manifest.get("preregistered_manifest_sha256") != payload_sha256(manifest):
        problems.append("manifest self-hash mismatch")
    setting_ids = [row["setting_id"] for row in manifest["settings"]]
    if len(setting_ids) != len(set(setting_ids)):
        problems.append("duplicate setting IDs")
    budget = manifest["budget"]
    committed = [row for row in manifest["settings"] if not row["conditional"]]
    conditional = [row for row in manifest["settings"] if row["conditional"]]
    committed_starts = sum(len(row["restart_indices"]) for row in committed)
    conditional_starts = sum(len(row["restart_indices"]) for row in conditional)
    if committed_starts != budget["committed_server_starts"]:
        problems.append("committed start count mismatch")
    if conditional_starts != budget["conditional_server_starts"]:
        problems.append("conditional start count mismatch")
    if committed_starts + conditional_starts != budget["all_server_starts"]:
        problems.append("total start count mismatch")
    if budget["all_server_starts"] > budget["hard_cap_server_starts"]:
        problems.append("server-start budget exceeds hard cap")
    if manifest["statistics"]["mde"]["mde_fraction"] != 0.05:
        problems.append("MDE is not frozen to 5%")
    if manifest["phase7_execution_authorized"]:
        problems.append("Phase7 must not be authorized by this manifest")
    if not manifest["execution_blockers"]:
        problems.append("pre-implementation manifest must remain blocked")
    for runner_name in ("ceiling", "scheduler"):
        status = manifest["runners"][runner_name]
        if status["exists"] and runner_name in manifest["execution_blockers"]:
            problems.append(f"{runner_name} exists but remains listed as missing")
        if not status["exists"] and runner_name not in manifest["execution_blockers"]:
            problems.append(f"{runner_name} missing without execution blocker")
    if manifest["server_template"]["test_only_injection_flags"] != {
        "SGLANG_APPROX_KV_TEST_ONLY": "0",
        "SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE": "0",
    }:
        problems.append("test-only injection flags are not pinned off")
    if len(manifest["workloads"]["A8"]["workloads"]) != 2:
        problems.append("A8 workload count mismatch")
    for workload in manifest["workloads"]["A8"]["workloads"]:
        if len(workload["targets"]) != 8:
            problems.append(f"{workload['workload_id']} does not have 8 targets")
    if not manifest["workloads"]["W"]["request_order"]:
        problems.append("W request order is empty")
    if args.check_plan:
        plan_repo = args.plan_repo.resolve()
        blob = subprocess.run(
            (
                "git",
                "show",
                f"{manifest['plan']['plan_commit']}:{manifest['plan']['plan_file']}",
            ),
            cwd=plan_repo,
            capture_output=True,
        )
        if blob.returncode != 0:
            problems.append("pinned plan blob cannot be read")
        elif sha256_bytes(blob.stdout) != manifest["plan"]["plan_sha256"]:
            problems.append("pinned plan hash mismatch")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plan-repo",
        type=Path,
        default=Path("/home/chris/Workspaces/code-agent-kvcache"),
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        default=Path(
            "/home/chris/Workspaces/code-agent-kvcache/" "IMPLEMENTATION_PLAN_LATEST.md"
        ),
    )
    parser.add_argument(
        "--plan-file-in-repo",
        default="IMPLEMENTATION_PLAN_LATEST.md",
    )
    parser.add_argument("--plan-commit")
    parser.add_argument(
        "--phase6-evidence-sha",
        default="924c9d1d6c074f304189248f0fc5b15aa6d25adb",
    )
    parser.add_argument(
        "--image-digest",
        default=(
            "sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781"
        ),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-plan", action="store_true", default=True)
    args = parser.parse_args()

    if args.check:
        manifest = json.loads(args.output.read_text())
        problems = validate(manifest, args)
        if problems:
            print(f"FAIL: {len(problems)} problem(s)")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(
            "OK: Phase7 primary manifest is internally consistent, "
            f"blocked by {len(manifest['execution_blockers'])} prerequisite(s)"
        )
        return 0

    if not args.plan_commit:
        parser.error("--plan-commit is required when generating")
    manifest = build_manifest(args)
    problems = validate(manifest, args)
    if problems:
        raise RuntimeError(f"refusing to write invalid manifest: {problems}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {len(manifest['settings'])} settings; "
        f"starts={manifest['budget']['all_server_starts']}; "
        f"blockers={len(manifest['execution_blockers'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
