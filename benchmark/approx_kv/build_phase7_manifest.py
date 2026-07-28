#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("benchmark/approx_kv/results/phase7/phase7-primary-manifest.json")
P6_CONTRACT = Path("benchmark/approx_kv/results/phase6/p6-0-contract.json")
DESIGN_KEYS = (
    "plan",
    "environment",
    "server_template",
    "workloads",
    "arms",
    "footprint_profiles",
    "arm_execution",
    "statistics",
    "outcome_taxonomy",
    "exclusive_terminal_reasons",
    "conditional_rules",
    "settings",
    "early_stops",
    "budget",
    "artifact_templates",
    "skipped_tracks",
    "required_inactive_counters",
    "scope_caveats",
)


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


def design_payload_sha256(payload: dict[str, Any]) -> str:
    design = {key: payload[key] for key in DESIGN_KEYS}
    return sha256_bytes(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode()
    )


def nested_manifest_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    return payload_sha256(canonical)


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
                    "extra_keys_by_arm": {
                        arm: (f"p7-a8-b{body_tokens}-target-{target_index}-{arm}")
                        for arm in ("D0", "E0", "R0", "R2")
                    },
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
                "same_context_canary": {
                    "target_id": f"a8-b{body_tokens}-same-context-canary",
                    "header_token_sha256": token_list_sha(source_header),
                    "body_token_sha256": token_list_sha(body),
                    "extra_key": f"p7-a8-b{body_tokens}-same-context-canary",
                    "max_new_tokens": 8,
                    "placement": "after_target_8_before_reset",
                    "arms": ["R0", "R2"],
                    "included_in_amortization": False,
                    "any_token_mismatch_is_invalid": True,
                },
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

    active_objects = [item for item in objects if item["active"]]
    first_replay = [
        {
            "request_index": len(workflow_requests) + position,
            "phase": "replay",
            "role": item["role"],
            "object_id": item["object_id"],
        }
        for position, item in enumerate(active_objects)
    ]
    second_replay = [
        {
            "request_index": (len(workflow_requests) + len(first_replay) + position),
            "phase": "replay-2",
            "role": item["role"],
            "object_id": item["object_id"],
        }
        for position, item in enumerate(active_objects)
    ]
    request_order = workflow_requests + first_replay + second_replay
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
    mem_fraction_static: float = 0.35,
    rho_realization: str = "filler_pool",
    capacity_ceiling_tokens: int = 13130,
    activation_rule_id: str | None = None,
    supplement_gate: str | None = None,
) -> dict[str, Any]:
    screening = [0] if 0 in restarts else []
    supplements = [restart for restart in restarts if restart != 0]
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
        "screening_restarts": screening,
        "supplement_restarts": supplements,
        "supplement_gate": (supplement_gate if supplements else None),
        "warmup_repeats": 1,
        "formal_repeats": 2,
        "arms": arms,
        "conditional": conditional,
        "activation_rule_id": activation_rule_id,
        "max_total_tokens": max_total_tokens,
        "mem_fraction_static": mem_fraction_static,
        "capacity_mode": (
            "explicit_max_total_tokens"
            if max_total_tokens is not None
            else "natural_capacity"
        ),
        "rho_realization": rho_realization,
        "known_capacity_ceiling_tokens": capacity_ceiling_tokens,
        "arm_order_by_repeat": {
            "0": arms,
            "1": list(reversed(arms)),
        },
        "reset_boundary": (
            "full exact/approx/metadata reset between arms; "
            "A8 sequence has no reset between targets and resets after target 8"
        ),
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
            mem_fraction_static=0.65,
            rho_realization="capacity_pinning",
            capacity_ceiling_tokens=20713,
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
            mem_fraction_static=0.65,
            rho_realization="capacity_pinning",
            capacity_ceiling_tokens=20713,
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
            mem_fraction_static=0.65,
            rho_realization="capacity_pinning",
            capacity_ceiling_tokens=20713,
            activation_rule_id="CR-P6DELTA-RHO3",
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
                    rho_realization="filler_pool",
                    supplement_gate="ES-R0-MDE",
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
            rho_realization="filler_pool",
            supplement_gate="ES-W-UNCONDITIONAL",
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
                    max_total_tokens=(15190 if rho == 1.5 else 11392),
                    mem_fraction_static=0.65,
                    rho_realization="capacity_pinning",
                    capacity_ceiling_tokens=20713,
                    supplement_gate="ES-W-UNCONDITIONAL",
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
                max_total_tokens=11392,
                mem_fraction_static=0.65,
                rho_realization="capacity_pinning",
                capacity_ceiling_tokens=20713,
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
                rho_realization="filler_pool",
                activation_rule_id="CR-R2-ADAPTER",
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
    workloads = {
        "A8": build_a8_workload(),
        "W": build_w_workload(p6_contract),
        "filler_pool": build_filler_pool(),
    }
    per_role_requests = dict(
        Counter(row["role"] for row in workloads["W"]["request_order"])
    )
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
            "required_cpu_test": (
                f"python3 -m pytest -q test/registered/unit/bench/"
                f"test_{Path(path).stem}.py"
                if name != "manifest"
                else "python3 -m benchmark.approx_kv.build_phase7_manifest --check"
            ),
            "cpu_test_status": (
                "passed"
                if name == "manifest" or name in args.runner_ready
                else "pending"
            ),
            "review_status": (
                "reviewed"
                if name == "manifest" or name in args.runner_ready
                else "pending"
            ),
        }
        for name, path in runner_paths.items()
    }
    execution_blockers = []
    for name in ("ceiling", "scheduler"):
        status = runner_status[name]
        if not status["exists"]:
            execution_blockers.append(f"missing_runner:{name}")
        elif name not in args.runner_ready:
            execution_blockers.append(f"runner_not_ready:{name}")
    if not args.phase7_pinned_implementation_sha:
        execution_blockers.append("phase7_pinned_implementation_sha_pending")
    if args.r2_strategy == "pending":
        execution_blockers.append("r2_strategy_pending")
    if args.rho3_resolution == "pending":
        execution_blockers.append("rho3_resolution_pending")
    if not args.authorize:
        execution_blockers.append("explicit_user_authorization_missing")

    manifest = {
        "schema_version": 1,
        "artifact": "phase7-primary-manifest",
        "manifest_revision": args.manifest_revision,
        "supersedes_manifest_sha256": args.supersedes_manifest_sha256,
        "supersedes_design_payload_sha256": (args.supersedes_design_payload_sha256),
        "design_keys": list(DESIGN_KEYS),
        "revision_reason": args.revision_reason,
        "status": args.status,
        "phase7_execution_authorized": args.authorize,
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
            "phase7_pinned_implementation_sha": (args.phase7_pinned_implementation_sha),
            "phase7_pinned_tree_sha": (
                git("rev-parse", f"{args.phase7_pinned_implementation_sha}^{{tree}}")
                if args.phase7_pinned_implementation_sha
                else None
            ),
            "atomic_update_required": [
                "phase7-primary-manifest.json",
                "HANDOFF.md",
                "PROJECT.md",
                "TODO_LOCAL.txt",
            ],
        },
        "runners": runner_status,
        "r2_strategy": args.r2_strategy,
        "conditional_resolution": {
            "CR-P6DELTA-RHO3": args.rho3_resolution,
            "CR-R2-ADAPTER": (
                "enabled"
                if args.r2_strategy == "adapter"
                else (
                    "disabled_not_comparable"
                    if args.r2_strategy == "disabled_not_comparable"
                    else "pending"
                )
            ),
        },
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
            "default_mem_fraction_static": 0.35,
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
            "plugin_env": {
                "SGLANG_APPROX_KV_CORE": "1",
                "SGLANG_APPROX_KV_CROSS_STORE": "1",
                "SGLANG_APPROX_KV_BYTES_PER_TOKEN": "114688",
                "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "0",
                "SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT": "1",
                "SGLANG_APPROX_KV_TEST_ONLY": "0",
                "SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE": "0",
            },
        },
        "workloads": workloads,
        "arms": {
            "D0": "dense_no_reuse_baseline",
            "E0": "exact_cache",
            "R0": "raw_copy_ceiling",
            "R2": "conditional_cross_store_adapter_or_not_comparable",
            "R4-like-5x": "synthetic_footprint_proxy_not_kvcomm",
        },
        "footprint_profiles": {
            "exact_only": "phase6 exact-only footprint profile",
            "r0_like": "one approximate representation",
            "r1_like_k32": "worst-case repair/temporary footprint",
            "r2_like": "2x synthetic representation multiplicity",
            "r4_like": "5x synthetic representation multiplicity",
            "relation_to_R4-like-5x_arm": (
                "same synthetic 5x footprint concept; different runner-specific "
                "identifier; neither executes KVCOMM"
            ),
        },
        "arm_execution": {
            "formal_0": ["D0", "E0", "R0"],
            "formal_1": ["R0", "E0", "D0"],
            "reset_between_arms": True,
            "A8_reset_between_targets": False,
            "A8_reset_after_target_8": True,
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
            "per_role_requests_per_formal": per_role_requests,
            "per_role_ttft_reporting": "descriptive_only_list_all_values",
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
        "conditional_rules": {
            "CR-P6DELTA-RHO3": (
                "enable only if the final report will make a "
                "chunk4096/rho3 claim; otherwise resolve disabled and keep "
                "rho3 feasibility scoped to chunk1024"
            ),
            "CR-R2-ADAPTER": (
                "enable only after the cross-store R2 adapter exists, passes "
                "CPU tests and is pinned in the Phase7 implementation SHA; "
                "otherwise resolve disabled/not_comparable"
            ),
        },
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
                "rule_id": "ES-W-UNCONDITIONAL",
                "checkpoint": "none",
                "rule": (
                    "W and chunk1024 sensitivity supplements are "
                    "unconditional per V5; only ES-ENGINEERING can stop them. "
                    "ES-R0-MDE governs A8 supplements only."
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
                    "only if BOTH rho1.5 and rho2.0 have all-reusable mean "
                    "gain <5% and miss_S4>=miss_S0 and peak_S4>=peak_S0, "
                    "stop scheduler benefit claim"
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
            "committed_gpu_settings": len(committed),
            "committed_server_starts": committed_starts,
            "conditional_gpu_settings": len(conditional),
            "conditional_server_starts": conditional_starts,
            "all_gpu_settings": len(settings),
            "all_server_starts": committed_starts + conditional_starts,
            "expected_gpu_hours": {
                "wave-0": 0.3,
                "wave-1": 0.4,
                "wave-2": 2.9,
                "conditional": 0.4,
            },
            "expected_gpu_hours_total": 4.0,
            "expected_minutes_per_start_by_runner": {
                "run_p6_4_capacity_pilot": 9,
                "run_p7_ceiling_A8": 6,
                "run_p7_ceiling_sensitivity": 6,
                "run_p7_scheduler_W": 8,
                "run_p7_scheduler_R4_like": 8,
                "conditional_R2": 6,
            },
            "hard_cap_server_starts": 36,
            "hard_cap_gpu_hours": 6,
            "gpu_hour_headroom": 2.0,
            "minimum_headroom_fraction": 0.15,
            "retries_count_against_cap": True,
        },
        "artifact_templates": {
            "raw_json": ("benchmark/approx_kv/results/phase7/raw/{run_id}.json"),
            "compact_json": (
                "benchmark/approx_kv/results/phase7/compact/{run_id}.json"
            ),
            "server_log": ("benchmark/approx_kv/results/phase7/logs/{run_id}.log"),
            "hash_timing": "after server stop and file close",
            "result_manifest": (
                "benchmark/approx_kv/results/phase7/RESULT_MANIFEST.json"
            ),
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
    manifest["design_payload_sha256"] = design_payload_sha256(manifest)
    manifest["preregistered_manifest_sha256"] = payload_sha256(manifest)
    return manifest


def validate(manifest: dict[str, Any], args: argparse.Namespace) -> list[str]:
    problems: list[str] = []
    if manifest.get("preregistered_manifest_sha256") != payload_sha256(manifest):
        problems.append("manifest self-hash mismatch")
    if manifest.get("design_payload_sha256") != design_payload_sha256(manifest):
        problems.append("immutable design payload hash mismatch")
    if manifest.get("manifest_revision", 0) <= 0:
        problems.append("manifest revision must be positive")
    if manifest["manifest_revision"] > 1 and not manifest.get(
        "supersedes_manifest_sha256"
    ):
        problems.append("revised manifest must record the superseded hash")
    if manifest["manifest_revision"] > 1 and not manifest.get(
        "supersedes_design_payload_sha256"
    ):
        problems.append("revised manifest must record the superseded design hash")
    if manifest.get("design_keys") != list(DESIGN_KEYS):
        problems.append("design key list does not match the builder")

    expected_settings = build_settings()
    if manifest["settings"] != expected_settings:
        problems.append("settings differ from the frozen builder design")
    p6_contract = json.loads(P6_CONTRACT.read_text())
    expected_workloads = {
        "A8": build_a8_workload(),
        "W": build_w_workload(p6_contract),
        "filler_pool": build_filler_pool(),
    }
    if manifest["workloads"] != expected_workloads:
        problems.append("workloads differ from the frozen builder design")
    for name, workload in manifest["workloads"].items():
        expected_hash = workload.get("manifest_sha256")
        if expected_hash != nested_manifest_sha256(workload):
            problems.append(f"{name} nested manifest hash mismatch")

    setting_ids = [row["setting_id"] for row in manifest["settings"]]
    if len(setting_ids) != len(set(setting_ids)):
        problems.append("duplicate setting IDs")
    allowed_policies = {"lru", "hierarchical"}
    allowed_arms = set(manifest["arms"])
    allowed_footprints = set(manifest["footprint_profiles"])
    for row in manifest["settings"]:
        if row["policy"] not in allowed_policies:
            problems.append(f"{row['setting_id']}: unsupported policy")
        if row["chunked_prefill_size"] not in {1024, 4096}:
            problems.append(f"{row['setting_id']}: unsupported chunk")
        if (
            "sensitivity" not in row["setting_id"]
            and row["chunked_prefill_size"] != 4096
        ):
            problems.append(f"{row['setting_id']}: non-sensitivity chunk is not 4096")
        unknown_arms = set(row["arms"]).difference(allowed_arms | allowed_footprints)
        if unknown_arms:
            problems.append(f"{row['setting_id']}: unknown arms {sorted(unknown_arms)}")
        if row["rho_realization"] == "capacity_pinning":
            if row["max_total_tokens"] is None:
                problems.append(
                    f"{row['setting_id']}: capacity pinning lacks max_total_tokens"
                )
            elif row["max_total_tokens"] > row["known_capacity_ceiling_tokens"]:
                problems.append(
                    f"{row['setting_id']}: pinned capacity exceeds known ceiling"
                )
        elif row["rho_realization"] == "filler_pool":
            if row["max_total_tokens"] is not None:
                problems.append(
                    f"{row['setting_id']}: filler realization pins capacity"
                )
        else:
            problems.append(f"{row['setting_id']}: unknown rho realization")
        if row["restart_indices"] != (
            row["screening_restarts"] + row["supplement_restarts"]
        ):
            problems.append(f"{row['setting_id']}: restart waves do not compose")
        if row["supplement_restarts"] and not row["supplement_gate"]:
            problems.append(f"{row['setting_id']}: supplements lack a gate")
        if row["supplement_gate"] == "ES-R0-MDE":
            if (
                row["runner"] != "benchmark.approx_kv.run_p7_ceiling"
                or not row["workload"].startswith("A8-")
                or "sensitivity" in row["setting_id"]
            ):
                problems.append(
                    f"{row['setting_id']}: ES-R0-MDE used outside A8 primary"
                )
        if row["supplement_restarts"] and row["supplement_gate"] not in {
            "ES-R0-MDE",
            "ES-W-UNCONDITIONAL",
        }:
            problems.append(f"{row['setting_id']}: unknown supplement gate")

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
    if len(committed) != budget["committed_gpu_settings"]:
        problems.append("committed setting count mismatch")
    if len(conditional) != budget["conditional_gpu_settings"]:
        problems.append("conditional setting count mismatch")
    if len(manifest["settings"]) != budget["all_gpu_settings"]:
        problems.append("total setting count mismatch")
    if budget["all_server_starts"] > budget["hard_cap_server_starts"]:
        problems.append("server-start budget exceeds hard cap")
    expected_gpu_hours = sum(budget["expected_gpu_hours"].values())
    if expected_gpu_hours != budget["expected_gpu_hours_total"]:
        problems.append("expected GPU-hour total mismatch")
    if expected_gpu_hours > budget["hard_cap_gpu_hours"] * (
        1 - budget["minimum_headroom_fraction"]
    ):
        problems.append("expected GPU hours do not preserve required headroom")
    if budget["hard_cap_gpu_hours"] - expected_gpu_hours != budget["gpu_hour_headroom"]:
        problems.append("GPU-hour headroom mismatch")
    if manifest["statistics"]["mde"]["mde_fraction"] != 0.05:
        problems.append("MDE is not frozen to 5%")
    status = manifest.get("status")
    authorized = manifest.get("phase7_execution_authorized")
    pinned_sha = manifest["implementation"].get("phase7_pinned_implementation_sha")
    blockers = manifest["execution_blockers"]
    if status == "preregistered_blocked":
        if authorized or pinned_sha is not None or not blockers:
            problems.append("invalid preregistered_blocked state")
    elif status == "pinned_blocked":
        if authorized or pinned_sha is None or not blockers:
            problems.append("invalid pinned_blocked state")
    elif status == "authorized":
        if not authorized or pinned_sha is None or blockers:
            problems.append("invalid authorized state")
    else:
        problems.append(f"unknown manifest status: {status}")

    pinned_tree = manifest["implementation"].get("phase7_pinned_tree_sha")
    if pinned_sha is not None:
        resolved = subprocess.run(
            ("git", "rev-parse", pinned_sha),
            capture_output=True,
            text=True,
        )
        if resolved.returncode != 0 or resolved.stdout.strip() != pinned_sha:
            problems.append("pinned implementation commit does not resolve exactly")
        else:
            expected_tree = git("rev-parse", f"{pinned_sha}^{{tree}}")
            if pinned_tree != expected_tree:
                problems.append("pinned implementation tree mismatch")

    for runner_name in ("ceiling", "scheduler"):
        runner_status_entry = manifest["runners"][runner_name]
        missing_blocker = f"missing_runner:{runner_name}"
        pending_blocker = f"runner_not_ready:{runner_name}"
        if not runner_status_entry["exists"] and missing_blocker not in blockers:
            problems.append(f"{runner_name} missing without blocker")
        if runner_status_entry["exists"] and missing_blocker in blockers:
            problems.append(f"{runner_name} exists but listed as missing")
        if (
            runner_status_entry["exists"]
            and runner_status_entry["cpu_test_status"] != "passed"
        ):
            if pending_blocker not in blockers:
                problems.append(f"{runner_name} untested without blocker")
        if (
            runner_status_entry["exists"]
            and runner_status_entry["review_status"] != "reviewed"
        ):
            if pending_blocker not in blockers:
                problems.append(f"{runner_name} unreviewed without blocker")

    manifest_runner = manifest["runners"]["manifest"]
    manifest_path = Path(manifest_runner["path"])
    if not manifest_path.exists():
        problems.append("manifest builder is missing")
    elif sha256_file(manifest_path) != manifest_runner["sha256"]:
        problems.append("manifest builder hash mismatch")
    generation_sha = manifest["implementation"]["manifest_generation_sha"]
    blob = subprocess.run(
        ("git", "show", f"{generation_sha}:{manifest_runner['path']}"),
        capture_output=True,
    )
    if blob.returncode != 0:
        problems.append("manifest builder is absent from generation commit")
    elif sha256_bytes(blob.stdout) != manifest_runner["sha256"]:
        problems.append("generation commit has a different manifest builder")

    if pinned_sha is not None:
        for runner_name in ("ceiling", "scheduler"):
            runner = manifest["runners"][runner_name]
            if not runner["exists"]:
                continue
            blob = subprocess.run(
                ("git", "show", f"{pinned_sha}:{runner['path']}"),
                capture_output=True,
            )
            if blob.returncode != 0:
                problems.append(
                    f"{runner_name} runner absent from pinned implementation"
                )
            elif sha256_bytes(blob.stdout) != runner["sha256"]:
                problems.append(
                    f"{runner_name} runner hash differs at pinned implementation"
                )

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
        keys = [
            extra_key
            for target in workload["targets"]
            for extra_key in target["extra_keys_by_arm"].values()
        ]
        if len(keys) != len(set(keys)):
            problems.append(f"{workload['workload_id']} has duplicate arm keys")
        canary = workload.get("same_context_canary", {})
        if canary.get("placement") != "after_target_8_before_reset":
            problems.append(f"{workload['workload_id']} canary placement is invalid")
        if canary.get("arms") != ["R0", "R2"]:
            problems.append(f"{workload['workload_id']} canary arms are invalid")
    if not manifest["workloads"]["W"]["request_order"]:
        problems.append("W request order is empty")
    request_order = manifest["workloads"]["W"]["request_order"]
    if [row["request_index"] for row in request_order] != list(
        range(len(request_order))
    ):
        problems.append("W request indexes are not contiguous")
    if len(manifest["workloads"]["W"]["objects"]) != 40:
        problems.append("W object count is not 40")
    if len(manifest["workloads"]["filler_pool"]["pool"]) != 64:
        problems.append("filler pool count is not 64")
    resolutions = manifest["conditional_resolution"]
    if status in {"pinned_blocked", "authorized"}:
        if any(value == "pending" for value in resolutions.values()):
            problems.append("pinned/authorized manifest has pending conditions")
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
        required=True,
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--plan-file-in-repo",
        default="IMPLEMENTATION_PLAN_LATEST.md",
    )
    parser.add_argument("--plan-commit")
    parser.add_argument("--manifest-revision", type=int, default=1)
    parser.add_argument("--supersedes-manifest-sha256")
    parser.add_argument("--supersedes-design-payload-sha256")
    parser.add_argument("--revision-reason", default="initial preregistration")
    parser.add_argument(
        "--status",
        choices=("preregistered_blocked", "pinned_blocked", "authorized"),
        default="preregistered_blocked",
    )
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--phase7-pinned-implementation-sha")
    parser.add_argument(
        "--runner-ready",
        action="append",
        choices=("ceiling", "scheduler"),
        default=[],
    )
    parser.add_argument(
        "--r2-strategy",
        choices=("pending", "adapter", "disabled_not_comparable"),
        default="pending",
    )
    parser.add_argument(
        "--rho3-resolution",
        choices=("pending", "enabled", "disabled_scoped_chunk1024"),
        default="pending",
    )
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
