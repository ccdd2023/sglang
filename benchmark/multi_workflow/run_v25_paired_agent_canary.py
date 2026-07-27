#!/usr/bin/env python3
"""Paired SWE-bench agent canary with a shared history and container snapshot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.utils.serialize import recursive_merge

from benchmark.multi_workflow.bridge_reuse_litellm_model import (
    BridgeReuseLitellmModel,
    token_ids_hash,
)
from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    CONFIG,
    DATASET,
    MODEL,
    launch_server,
    load_jsonl,
    stop_server,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v25_paired_agent_canary_20260727"
PRIOR_V25_LEDGER = DEFAULT_OUTPUT / "run" / "SERVER_LEDGER.jsonl"
PROJECT = Path(__file__).resolve().parents[2]
INSTANCE_ID = os.environ.get(
    "IMPACTKV_PAIRED_INSTANCE_ID",
    "scikit-learn__scikit-learn-13779",
)
GENERAL = "general"
V23 = "coding_post_mutation_target_prefix_v23"
ARMS = (V23, GENERAL)
PORT = 32950
MEM_FRACTION_STATIC = 0.80


def _dataset_rows() -> list[dict[str, Any]]:
    path = DATASET / "test.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instance() -> dict[str, Any]:
    return next(
        row for row in _dataset_rows() if row["instance_id"] == INSTANCE_ID
    )


def _base_config() -> dict[str, Any]:
    return recursive_merge(
        get_config_from_spec("swebench.yaml"),
        get_config_from_spec(CONFIG),
    )


def _model(
    config: dict[str, Any],
    *,
    arm: str,
    manifest: Path | None,
    ledger: Path | None,
) -> BridgeReuseLitellmModel:
    value = copy.deepcopy(config["model"])
    value["reuse_arm"] = arm
    value["model_kwargs"]["api_base"] = f"http://127.0.0.1:{PORT}/v1"
    value["reuse_manifest_path"] = manifest
    value["reuse_client_ledger_path"] = ledger
    return get_model(config=value)  # type: ignore[return-value]


def _init_manifest(output: Path) -> Path:
    path = output / "run" / "DYNAMIC_MANIFEST.json"
    write_json(
        path,
        {
            "version": 3,
            "model_id": MODEL,
            "cache_dtype": "bfloat16",
            "lease_ttl_s": 900,
            "ledger_path": str(output / "run" / "SERVER_LEDGER.jsonl"),
            "rope": {
                "rotary_dim": 128,
                "base": 10_000_000,
                "is_neox_style": True,
            },
            "sources": [],
            "cases": [],
            "release_source_ids": [],
            "arm": "v25_paired_agent",
            "host_overflow_enabled": True,
            "ordinary_prefix_reuse_enabled": True,
            "ordinary_prefix_repair_tokens": 0,
            "ordinary_prefix_target_only": True,
        },
    )
    return path


def register(output: Path) -> dict[str, Any]:
    path = output / "V25_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    instance = _instance()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_TREATMENT_GPU_RUN",
        "experiment": "V25 shared-prefix, cloned-repository paired agent canary",
        "motivation": (
            "Free-running General and V23 agents diverged before treatment. "
            "Run one Dense shared agent until the first online point where "
            "both policies select unequal reusable spans, materialize both "
            "from that same request, snapshot the container, and only then "
            "branch."
        ),
        "instance_id": INSTANCE_ID,
        "problem_statement_sha256": hashlib.sha256(
            instance["problem_statement"].encode()
        ).hexdigest(),
        "prior_failed_attempt": (
            {
                "server_ledger": str(PRIOR_V25_LEDGER),
                "server_ledger_sha256": sha256(PRIOR_V25_LEDGER),
                "classification": (
                    "Paired control-loop failure after 22 successful target "
                    "copies and zero fallbacks: FormatError was not converted "
                    "to the standard mini-agent correction message."
                ),
            }
            if PRIOR_V25_LEDGER.exists()
            else None
        ),
        "protocol": {
            "model": MODEL,
            "temperature": 0,
            "step_limit": 20,
            "branch_rule": (
                "first online request where General and V23 both register "
                "a source and their selected lengths differ"
            ),
            "shared_prefix_arm": "dense",
            "target_order_each_paired_step": list(ARMS),
            "container_clone": "docker commit after shared tool observation",
            "same_server": True,
            "mem_fraction_static": MEM_FRACTION_STATIC,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": {
            "branch_found_by_call_max": 12,
            "branch_source_prompt_hash_identical": True,
            "branch_source_lengths_different": True,
            "two_source_materializations_min": 2,
            "target_copies_each_arm_min": 1,
            "target_fallbacks": 0,
            "first_branched_prompt_hash_identical": True,
            "both_branches_produce_submission": True,
            "official_evaluation_required_before_accuracy_claim": True,
        },
        "inputs": {
            "dataset": str(DATASET / "test.jsonl"),
            "dataset_sha256": sha256(DATASET / "test.jsonl"),
            "config": str(CONFIG),
            "config_sha256": sha256(CONFIG),
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


def _initialize_agent(
    model: BridgeReuseLitellmModel,
    env: Any,
    config: dict[str, Any],
    task: str,
) -> DefaultAgent:
    agent = DefaultAgent(model, env, **copy.deepcopy(config["agent"]))
    agent.extra_template_vars = {"task": task}
    agent.add_messages(
        model.format_message(
            role="system",
            content=agent._render_template(agent.config.system_template),
        ),
        model.format_message(
            role="user",
            content=agent._render_template(agent.config.instance_template),
        ),
    )
    return agent


def _clone_agent(
    shared: DefaultAgent,
    model: BridgeReuseLitellmModel,
    env: Any,
    config: dict[str, Any],
) -> DefaultAgent:
    agent = DefaultAgent(model, env, **copy.deepcopy(config["agent"]))
    agent.messages = copy.deepcopy(shared.messages)
    agent.extra_template_vars = copy.deepcopy(shared.extra_template_vars)
    agent.n_calls = shared.n_calls
    agent.cost = shared.cost
    agent._start_time = shared._start_time
    return agent


def _execute_agent_message(agent: DefaultAgent, message: dict[str, Any]) -> None:
    agent.cost += message.get("extra", {}).get("cost", 0.0)
    agent.add_messages(message)
    try:
        agent.execute_actions(message)
    except InterruptAgentFlow as error:
        agent.add_messages(*error.messages)


def _prepare_pair(
    models: dict[str, BridgeReuseLitellmModel],
    messages: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    prepared = {
        arm: models[arm].prepare_reuse_query(
            messages[arm],
            write_sidecar=False,
        )
        for arm in ARMS
    }
    models[V23]._atomic_sidecar_update(
        sources=[
            prepared[arm]["source"]
            for arm in ARMS
            if prepared[arm]["source"] is not None
        ],
        cases=[
            {
                **prepared[arm]["target"],
                "ordinary_prefix_reuse": arm == V23,
            }
            for arm in ARMS
            if prepared[arm]["target"] is not None
        ],
        release_source_ids=[
            source_id
            for arm in ARMS
            for source_id in prepared[arm]["releases"]
        ],
    )
    return prepared


def _prepared_step(
    agent: DefaultAgent,
    prepared: dict[str, Any],
) -> None:
    agent.n_calls += 1
    try:
        message = agent.model.execute_prepared_reuse_query(prepared)
    except InterruptAgentFlow as error:
        agent.add_messages(*error.messages)
        return
    _execute_agent_message(agent, message)


def _normal_step(agent: DefaultAgent) -> None:
    agent.n_calls += 1
    try:
        message = agent.model.query(agent.messages)
    except InterruptAgentFlow as error:
        agent.add_messages(*error.messages)
        return
    _execute_agent_message(agent, message)


def _docker_clone_environments(
    shared_env: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    executable = shared_env.config.executable
    image = f"impactkv-v25-paired-{os.getpid()}:snapshot"
    subprocess.run(
        [executable, "commit", shared_env.container_id, image],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    env_config = shared_env.config.model_dump()
    env_config["image"] = image
    environments = {
        arm: get_environment(
            {"environment_class": "docker", **copy.deepcopy(env_config)}
        )
        for arm in ARMS
    }
    return environments, image


def _submission(agent: DefaultAgent) -> str:
    if not agent.messages:
        return ""
    return str(agent.messages[-1].get("extra", {}).get("submission") or "")


def _save_branch(
    output: Path,
    arm: str,
    agent: DefaultAgent,
) -> None:
    run_dir = output / arm
    instance_dir = run_dir / INSTANCE_ID
    instance_dir.mkdir(parents=True, exist_ok=True)
    agent.save(
        instance_dir / f"{INSTANCE_ID}.traj.json",
        {"instance_id": INSTANCE_ID},
    )
    write_json(
        run_dir / "preds.json",
        {
            INSTANCE_ID: {
                "model_name_or_path": (
                    f"impactkv__v25-paired-{arm}"
                ),
                "instance_id": INSTANCE_ID,
                "model_patch": _submission(agent),
            }
        },
    )


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    result_path = output / "V25_RESULT.json"
    if result_path.exists():
        return read_json(result_path)
    run_dir = output / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _init_manifest(output)
    config = _base_config()
    instance = _instance()
    models = {
        arm: _model(
            config,
            arm=arm,
            manifest=manifest,
            ledger=output / arm / "CLIENT_LEDGER.jsonl",
        )
        for arm in ARMS
    }
    dense_model = _model(
        config,
        arm="dense",
        manifest=None,
        ledger=None,
    )
    process, log = launch_server(
        run_dir=run_dir,
        arm=V23,
        manifest=manifest,
        port=PORT,
        mem_fraction_static=MEM_FRACTION_STATIC,
    )
    shared_env = None
    branch_envs: dict[str, Any] = {}
    snapshot_image = ""
    agents: dict[str, DefaultAgent] = {}
    branch: dict[str, Any] | None = None
    try:
        shared_env = get_sb_environment(copy.deepcopy(config), instance)
        shared = _initialize_agent(
            dense_model,
            shared_env,
            config,
            instance["problem_statement"],
        )
        for call in range(1, 13):
            shadow = {
                arm: models[arm].prepare_reuse_query(
                    shared.messages,
                    write_sidecar=False,
                )
                for arm in ARMS
            }
            hashes = {
                arm: token_ids_hash(shadow[arm]["prompt_ids"])
                for arm in ARMS
            }
            sources = {arm: shadow[arm]["source"] for arm in ARMS}
            eligible = (
                all(sources.values())
                and len({hashes[arm] for arm in ARMS}) == 1
                and int(sources[V23]["length"])
                != int(sources[GENERAL]["length"])
            )
            if eligible:
                models[V23]._atomic_sidecar_update(
                    sources=[sources[arm] for arm in ARMS],
                )
            else:
                for arm in ARMS:
                    models[arm]._pending_source = None
            shared.n_calls += 1
            shared_message = dense_model.query(shared.messages)
            _execute_agent_message(shared, shared_message)
            if shared.messages[-1].get("role") == "exit":
                raise RuntimeError("shared agent exited before branch")
            if eligible:
                branch = {
                    "shared_calls": call,
                    "source_prompt_hash": hashes[V23],
                    "source_lengths": {
                        arm: int(sources[arm]["length"]) for arm in ARMS
                    },
                    "source_ids": {
                        arm: str(sources[arm]["source_id"]) for arm in ARMS
                    },
                }
                break
        if branch is None:
            raise RuntimeError("no unequal paired source by shared call 12")

        branch_envs, snapshot_image = _docker_clone_environments(
            shared_env,
            config,
        )
        agents = {
            arm: _clone_agent(shared, models[arm], branch_envs[arm], config)
            for arm in ARMS
        }
        shared_env.cleanup()
        shared_env = None

        first_branch_hashes: dict[str, str] = {}
        while True:
            active = [
                arm
                for arm in ARMS
                if agents[arm].messages[-1].get("role") != "exit"
                and agents[arm].n_calls < agents[arm].config.step_limit
            ]
            if not active:
                break
            if len(active) == 2:
                prepared = _prepare_pair(
                    models,
                    {arm: agents[arm].messages for arm in ARMS},
                )
                if not first_branch_hashes:
                    first_branch_hashes = {
                        arm: token_ids_hash(prepared[arm]["prompt_ids"])
                        for arm in ARMS
                    }
                for arm in ARMS:
                    _prepared_step(agents[arm], prepared[arm])
            else:
                arm = active[0]
                _normal_step(agents[arm])

        for arm in ARMS:
            if agents[arm].messages[-1].get("role") != "exit":
                agents[arm].add_messages(
                    models[arm].format_message(
                        role="exit",
                        content="LimitsExceeded",
                        extra={
                            "exit_status": "LimitsExceeded",
                            "submission": "",
                        },
                    )
                )
            _save_branch(output, arm, agents[arm])
    finally:
        for env in branch_envs.values():
            env.cleanup()
        if shared_env is not None:
            shared_env.cleanup()
        if snapshot_image:
            executable = (
                next(iter(branch_envs.values())).config.executable
                if branch_envs
                else "docker"
            )
            subprocess.run(
                [executable, "image", "rm", "-f", snapshot_image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        stop_server(process, log)

    server = load_jsonl(run_dir / "SERVER_LEDGER.jsonl")
    copies = [
        row for row in server if row.get("event") == "target_copied"
    ]
    fallbacks = [
        row for row in server if row.get("event") == "target_fallback"
    ]
    materialized = [
        row
        for row in server
        if row.get("event")
        in {"source_materialized", "source_materialized_host"}
    ]
    first_hash_equal = (
        len(set(first_branch_hashes.values())) == 1
        if first_branch_hashes
        else False
    )
    submissions = {arm: _submission(agents[arm]) for arm in ARMS}
    copy_counts = {
        arm: sum(row.get("policy_label") == arm for row in copies)
        for arm in ARMS
    }
    gates = {
        "branch_found_by_call_max": branch is not None
        and int(branch["shared_calls"]) <= 12,
        "branch_source_prompt_hash_identical": branch is not None,
        "branch_source_lengths_different": branch is not None
        and len(set(branch["source_lengths"].values())) == 2,
        "two_source_materializations_min": len(materialized) >= 2,
        "target_copies_each_arm_min": all(
            copy_counts[arm] >= 1 for arm in ARMS
        ),
        "target_fallbacks": len(fallbacks) == 0,
        "first_branched_prompt_hash_identical": first_hash_equal,
        "both_branches_produce_submission": all(submissions.values()),
        "official_evaluation_required_before_accuracy_claim": True,
    }
    result = {
        "completed_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "registration_status": registration["status"],
        "branch": branch,
        "first_branch_prompt_hashes": first_branch_hashes,
        "calls": {arm: agents[arm].n_calls for arm in ARMS},
        "exit_status": {
            arm: agents[arm].messages[-1].get("extra", {}).get(
                "exit_status"
            )
            for arm in ARMS
        },
        "submission_bytes": {
            arm: len(submissions[arm].encode()) for arm in ARMS
        },
        "server": {
            "source_materializations": len(materialized),
            "target_copies": len(copies),
            "copy_counts": copy_counts,
            "target_fallbacks": len(fallbacks),
        },
        "gates": gates,
        "accuracy": (
            "NOT_YET_MEASURED: official SWE-bench evaluation is a separate "
            "required stage."
        ),
    }
    write_json(result_path, result)
    return result


def summarize_official(output: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARMS:
        result = read_json(output / arm / "OFFICIAL_RESULT.json")
        report = result["report"]
        arms[arm] = {
            "resolved": int(report["resolved_instances"]),
            "total": int(report["total_instances"]),
            "empty_patch": int(report["empty_patch_instances"]),
            "report_path": result["report_path"],
        }
    v23_resolved = arms[V23]["resolved"]
    general_resolved = arms[GENERAL]["resolved"]
    if v23_resolved > general_resolved:
        interpretation = (
            "Official paired V23-only resolution: V23 resolved 1/1 while "
            "General produced an empty patch.  Because history, repository "
            "snapshot, first branched prompt, server, and model are shared, "
            "this is causal evidence for the post-branch reuse policy on this "
            "instance, not a population-level accuracy estimate."
        )
    elif v23_resolved == general_resolved:
        interpretation = (
            "The arms have equal official resolution on this instance.  The "
            "result does not rank their accuracy or establish a population "
            "effect."
        )
    else:
        interpretation = (
            "Official paired General-only resolution on this instance.  V23 "
            "must be treated as an accuracy damage until replicated."
        )
    value = {
        "summarized_at_utc": utc_now(),
        "instance_id": INSTANCE_ID,
        "arms": arms,
        "paired_accuracy_interpretation": interpretation,
    }
    write_json(output / "V25_OFFICIAL_RESULT.json", value)
    return value


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
        value = summarize_official(args.output)
    print(
        {
            "status": value.get("status"),
            "output": str(args.output),
            "gates": value.get("gates"),
        }
    )


if __name__ == "__main__":
    main()
