#!/usr/bin/env python3
"""Same-server paired General/V23 exact-KV mechanism canary.

This is deliberately a mechanical and fidelity experiment, not a task-accuracy
experiment.  One real frozen source request materializes both candidate spans.
The identical next prompt is then issued twice to the same deterministic
server, first as V23 and then as General.  Per-case prefix policy keeps the
General branch dense even though V23 has already populated RadixCache.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    MODEL,
    launch_server,
    load_jsonl,
    stop_server,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    INSTANCE_IDS,
    assistant_request_prefixes,
    coarse_js,
    generate_one,
    make_planner,
    plan_request,
    read_json,
    reset_planner_session,
    sha256,
    token_id,
    top_distribution,
    trajectory_path,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v24_paired_mechanism_canary_20260727"
PRIOR_V24_LOG = DEFAULT_OUTPUT / "run" / "sglang_server.log"
PROJECT = Path(__file__).resolve().parents[2]
GENERAL = "general"
V23 = "coding_post_mutation_target_prefix_v23"
PORT = 32940
MEM_FRACTION_STATIC = 0.80


def _paired_plan() -> dict[str, Any]:
    """Select the first deterministic trajectory transition with unequal spans."""

    for instance_id in INSTANCE_IDS:
        messages = read_json(trajectory_path(instance_id))["messages"]
        prefixes = assistant_request_prefixes(messages)
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for arm in (GENERAL, V23):
            planner = make_planner(
                arm=arm,
                manifest_path=None,
                client_ledger_path=None,
                instance_nonce=f"paired-scan-{arm}",
            )
            reset_planner_session(planner, instance_id=instance_id)
            by_arm[arm] = [
                plan_request(planner, prefix) for prefix in prefixes
            ]
        for target_offset in range(1, len(prefixes)):
            general_source = by_arm[GENERAL][target_offset - 1]["source"]
            v23_source = by_arm[V23][target_offset - 1]["source"]
            general_target = by_arm[GENERAL][target_offset]["target"]
            v23_target = by_arm[V23][target_offset]["target"]
            if not all(
                (general_source, v23_source, general_target, v23_target)
            ):
                continue
            source_hashes = {
                by_arm[arm][target_offset - 1]["prompt_hash"]
                for arm in (GENERAL, V23)
            }
            target_hashes = {
                by_arm[arm][target_offset]["prompt_hash"]
                for arm in (GENERAL, V23)
            }
            if len(source_hashes) != 1 or len(target_hashes) != 1:
                continue
            if int(general_source["length"]) == int(v23_source["length"]):
                continue
            return {
                "instance_id": instance_id,
                "source_request_index": target_offset,
                "target_request_index": target_offset + 1,
                "prefixes": prefixes,
                "source_prompt_hash": source_hashes.pop(),
                "target_prompt_hash": target_hashes.pop(),
                "source_prompt_ids": by_arm[GENERAL][target_offset - 1][
                    "prompt_ids"
                ],
                "target_prompt_ids": by_arm[GENERAL][target_offset][
                    "prompt_ids"
                ],
                "sources": {
                    GENERAL: general_source,
                    V23: v23_source,
                },
                "targets": {
                    GENERAL: general_target,
                    V23: v23_target,
                },
            }
    raise RuntimeError("no paired transition with unequal source spans")


def _public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"prefixes", "source_prompt_ids", "target_prompt_ids"}
    }


def register(output: Path) -> dict[str, Any]:
    path = output / "V24_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = _paired_plan()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_TREATMENT_GPU_RUN",
        "experiment": "V24 same-server paired General/V23 mechanism canary",
        "motivation": (
            "Independent deterministic agent runs diverged before the first "
            "registered KV copy.  Share one real history through the source "
            "request, materialize both spans from it, and branch only at one "
            "identical target prompt."
        ),
        "scope": (
            "Mechanical validity, same-prompt first-token fidelity, and prefix "
            "isolation only.  This canary makes no task-accuracy or speed claim."
        ),
        "prior_failed_attempt": (
            {
                "path": str(PRIOR_V24_LOG),
                "sha256": sha256(PRIOR_V24_LOG),
                "classification": (
                    "Infrastructure failure before target treatment: CUDA OOM "
                    "while materializing the first source span at 0.90 static "
                    "memory fraction; no target-copy ledger event."
                ),
            }
            if PRIOR_V24_LOG.exists()
            else None
        ),
        "protocol": {
            "model": MODEL,
            "temperature": 0,
            "diagnostic_new_tokens": 1,
            "diagnostic_outputs_fed_forward": False,
            "one_real_source_request_materializes_both_spans": True,
            "target_order": [V23, GENERAL],
            "v23_ordinary_prefix_reuse": True,
            "general_ordinary_prefix_reuse": False,
            "same_server": True,
            "same_target_prompt_ids": True,
            "mem_fraction_static": MEM_FRACTION_STATIC,
            "prefetch": False,
        },
        "frozen_gates": {
            "source_materializations": 2,
            "source_ids_exact": True,
            "target_copies": 2,
            "target_fallbacks": 0,
            "target_case_order_exact": [V23, GENERAL],
            "v23_ordinary_prefix_tokens_min": 1,
            "general_ordinary_prefix_tokens": 0,
            "first_token_agreement": True,
        },
        "selected_transition": _public_plan(plan),
        "inputs": {
            "trajectory_path": str(trajectory_path(plan["instance_id"])),
            "trajectory_sha256": sha256(
                trajectory_path(plan["instance_id"])
            ),
            "source_sha256": {
                str(source.relative_to(PROJECT)): sha256(source)
                for source in (
                    PROJECT
                    / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
                    PROJECT
                    / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
                    PROJECT
                    / "python/sglang/srt/mem_cache/kvcomm_exact.py",
                    Path(__file__),
                )
            },
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(path, value)
    return value


def _manifest(output: Path, plan: dict[str, Any]) -> Path:
    run_dir = output / "run"
    path = run_dir / "DYNAMIC_MANIFEST.json"
    v23_target = deepcopy(plan["targets"][V23])
    v23_target["ordinary_prefix_reuse"] = True
    general_target = deepcopy(plan["targets"][GENERAL])
    general_target["ordinary_prefix_reuse"] = False
    write_json(
        path,
        {
            "version": 3,
            "model_id": MODEL,
            "cache_dtype": "bfloat16",
            "lease_ttl_s": 900,
            "ledger_path": str(run_dir / "SERVER_LEDGER.jsonl"),
            "rope": {
                "rotary_dim": 128,
                "base": 10_000_000,
                "is_neox_style": True,
            },
            "sources": [
                plan["sources"][V23],
                plan["sources"][GENERAL],
            ],
            # Ordering is treatment ordering for duplicate target hashes.
            "cases": [v23_target, general_target],
            "release_source_ids": [],
            "arm": "v24_same_server_paired",
            "host_overflow_enabled": True,
            "ordinary_prefix_reuse_enabled": True,
            "ordinary_prefix_repair_tokens": 0,
            "ordinary_prefix_target_only": True,
        },
    )
    return path


def _event_summary(
    ledger: list[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    source_events = [
        row
        for row in ledger
        if row.get("event")
        in {"source_materialized", "source_materialized_host"}
    ]
    copy_events = [
        row for row in ledger if row.get("event") == "target_copied"
    ]
    fallback_events = [
        row for row in ledger if row.get("event") == "target_fallback"
    ]
    prefix_events = {
        str(row["case_id"]): int(row["ordinary_prefix_tokens"])
        for row in ledger
        if row.get("event") == "target_ordinary_prefix_matched"
    }
    expected_sources = {
        str(plan["sources"][arm]["source_id"]) for arm in (V23, GENERAL)
    }
    expected_cases = [
        str(plan["targets"][arm]["case_id"]) for arm in (V23, GENERAL)
    ]
    gates = {
        "source_materializations": len(source_events) == 2,
        "source_ids_exact": {
            str(row["source_id"]) for row in source_events
        }
        == expected_sources,
        "target_copies": len(copy_events) == 2,
        "target_fallbacks": len(fallback_events) == 0,
        "target_case_order_exact": [
            str(row["case_id"]) for row in copy_events
        ]
        == expected_cases,
        "v23_ordinary_prefix_tokens_min": prefix_events.get(
            expected_cases[0], 0
        )
        >= 1,
        "general_ordinary_prefix_tokens": prefix_events.get(
            expected_cases[1], -1
        )
        == 0,
    }
    return {
        "source_events": source_events,
        "copy_events": copy_events,
        "fallback_events": fallback_events,
        "prefix_events": prefix_events,
        "gates": gates,
    }


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    result_path = output / "V24_RESULT.json"
    if result_path.exists():
        return read_json(result_path)
    plan = _paired_plan()
    if _public_plan(plan) != registration["selected_transition"]:
        raise ValueError("paired plan differs from registration")
    run_dir = output / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(output, plan)
    process, log = launch_server(
        run_dir=run_dir,
        arm=V23,
        manifest=manifest,
        port=PORT,
        mem_fraction_static=MEM_FRACTION_STATIC,
    )
    history_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    try:
        base_url = f"http://127.0.0.1:{PORT}"
        planner = make_planner(
            arm="dense",
            manifest_path=None,
            client_ledger_path=None,
            instance_nonce="v24-runtime-dense",
        )
        reset_planner_session(planner, instance_id=plan["instance_id"])
        source_index = int(plan["source_request_index"])
        for request_index, prefix in enumerate(
            plan["prefixes"][:source_index], start=1
        ):
            planned = plan_request(planner, prefix)
            generated = generate_one(
                base_url=base_url,
                input_ids=planned["prompt_ids"],
                key=f"v24-shared-{plan['instance_id']}-q{request_index}",
            )
            history_rows.append(
                {
                    "request_index": request_index,
                    "prompt_hash": planned["prompt_hash"],
                    "prompt_tokens": planned["prompt_tokens"],
                    **generated,
                }
            )
        if history_rows[-1]["prompt_hash"] != plan["source_prompt_hash"]:
            raise ValueError("runtime source prompt differs from registration")
        for arm in (V23, GENERAL):
            case_id = str(plan["targets"][arm]["case_id"])
            generated = generate_one(
                base_url=base_url,
                input_ids=plan["target_prompt_ids"],
                key=case_id,
            )
            target_rows.append(
                {
                    "arm": arm,
                    "case_id": case_id,
                    "prompt_hash": plan["target_prompt_hash"],
                    "prompt_tokens": len(plan["target_prompt_ids"]),
                    **generated,
                }
            )
    finally:
        stop_server(process, log)
    ledger = load_jsonl(run_dir / "SERVER_LEDGER.jsonl")
    events = _event_summary(ledger, plan)
    first_token_agreement = token_id(target_rows[0]) == token_id(
        target_rows[1]
    )
    gates = {
        **events["gates"],
        "first_token_agreement": first_token_agreement,
    }
    result = {
        "completed_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": registration["scope"],
        "selected_transition": _public_plan(plan),
        "shared_history_requests": len(history_rows),
        "shared_source_request": history_rows[-1],
        "target_rows": target_rows,
        "same_prompt_first_token": {
            "agreement": first_token_agreement,
            "v23_token_id": token_id(target_rows[0]),
            "general_token_id": token_id(target_rows[1]),
            "top20_plus_residual_js": coarse_js(
                top_distribution(target_rows[0]),
                top_distribution(target_rows[1]),
            ),
        },
        "events": events,
        "gates": gates,
        "interpretation": (
            "Passing validates the paired branch mechanism.  It does not "
            "validate task accuracy and its ordered TTFT values are not an "
            "unbiased speed comparison."
        ),
    }
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "run", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
    elif args.command == "run":
        value = run(args.output)
    else:
        value = read_json(args.output / "V24_RESULT.json")
    print(
        {
            "status": value.get("status"),
            "output": str(args.output),
            "gates": value.get("gates"),
        }
    )


if __name__ == "__main__":
    main()
