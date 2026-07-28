#!/usr/bin/env python3
"""Paired SWE-bench agent canary with a shared history and container snapshot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
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
    REGISTRATION as DEFAULT_EVAL_REGISTRATION,
    SNAPSHOT as DEFAULT_EVAL_SNAPSHOT,
    MODEL,
    launch_server,
    load_jsonl,
    run_official_evaluation,
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
V23 = os.environ.get(
    "IMPACTKV_PAIRED_CANDIDATE_ARM",
    "coding_post_mutation_target_prefix_v23",
)
ABSTENTION_CANDIDATE = V23 == "coding_critical_event_abstain_v31"
TARGET_VETO_CANDIDATES = {
    "coding_state_transition_target_v33b",
    "coding_critical_current_target_v34",
    "coding_version_validation_target_v35b",
    "coding_patch_lifecycle_target_v37",
}
TARGET_VETO_DENSE_MODES = {
    "state_transition_target_dense_veto",
    "critical_current_target_dense_veto",
    "version_validation_target_dense_veto",
    "patch_lifecycle_target_dense_veto",
}
TARGET_VETO_CANDIDATE = V23 in TARGET_VETO_CANDIDATES
TARGET_VETO_DENSE_MODE = (
    "patch_lifecycle_target_dense_veto"
    if V23 == "coding_patch_lifecycle_target_v37"
    else
    "version_validation_target_dense_veto"
    if V23 == "coding_version_validation_target_v35b"
    else
    "critical_current_target_dense_veto"
    if V23 == "coding_critical_current_target_v34"
    else "state_transition_target_dense_veto"
)
TARGET_PREFIX_CANDIDATES = {
    "coding_post_mutation_target_prefix_v23",
    "coding_post_mutation_payoff_guard_v28",
    "coding_post_mutation_payoff_guard_v29",
}
TARGET_PREFIX_CANDIDATE = V23 in TARGET_PREFIX_CANDIDATES
DENSE = "dense"
REUSE_ARMS = (V23, GENERAL)
INCLUDE_DENSE_CONTROL = (
    os.environ.get("IMPACTKV_PAIRED_DENSE_CONTROL", "0") == "1"
)
ALLOW_EMPTY_SUBMISSION_OUTCOME = (
    os.environ.get("IMPACTKV_ALLOW_EMPTY_SUBMISSION_OUTCOME", "0") == "1"
)
REQUIRE_BRANCH = os.environ.get("IMPACTKV_REQUIRE_BRANCH", "0") == "1"
ARMS = REUSE_ARMS + ((DENSE,) if INCLUDE_DENSE_CONTROL else ())
PORT = 32950
MEM_FRACTION_STATIC = 0.80
REQUEST_TIMEOUT_SECONDS = int(
    os.environ.get("IMPACTKV_REQUEST_TIMEOUT_SECONDS", "900")
)
DATASET_ROOT = Path(os.environ.get("IMPACTKV_DATASET_ROOT", str(DATASET)))
EVAL_REGISTRATION = Path(
    os.environ.get(
        "IMPACTKV_EVAL_REGISTRATION",
        str(DEFAULT_EVAL_REGISTRATION),
    )
)
EVAL_SNAPSHOT = Path(
    os.environ.get(
        "IMPACTKV_EVAL_SNAPSHOT",
        str(DEFAULT_EVAL_SNAPSHOT),
    )
)


def _dataset_rows() -> list[dict[str, Any]]:
    path = DATASET_ROOT / "test.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instance() -> dict[str, Any]:
    return next(
        row for row in _dataset_rows() if row["instance_id"] == INSTANCE_ID
    )


def _policy_mode(record: dict[str, Any]) -> str:
    """Read the policy mode from a prepared query or a client-ledger row."""

    for key in ("policy_decision", "reuse_policy_decision"):
        decision = record.get(key)
        if isinstance(decision, dict):
            return str(decision.get("mode") or "")
    return ""


def _is_target_veto_record(record: dict[str, Any]) -> bool:
    return (
        _policy_mode(record) in TARGET_VETO_DENSE_MODES
        and bool(
            record.get("reuse_policy_decision", {}).get("target_vetoed")
        )
    )


def _branch_kind(
    prepared: dict[str, dict[str, Any]],
) -> str | None:
    """Classify the first same-prompt candidate/General plan difference."""

    hashes = {
        arm: token_ids_hash(prepared[arm]["prompt_ids"])
        for arm in REUSE_ARMS
    }
    if len(set(hashes.values())) != 1:
        return None
    if (
        TARGET_VETO_CANDIDATE
        and prepared[V23]["target"] is None
        and prepared[GENERAL]["target"] is not None
        and _policy_mode(prepared[V23]) in TARGET_VETO_DENSE_MODES
    ):
        return "current_target_veto"
    sources = {arm: prepared[arm]["source"] for arm in REUSE_ARMS}
    if (
        ABSTENTION_CANDIDATE
        and sources[V23] is None
        and sources[GENERAL] is not None
        and _policy_mode(prepared[V23])
        == "critical_event_dense_abstain"
    ):
        return "future_source_plan"
    if (
        not ABSTENTION_CANDIDATE
        and not TARGET_VETO_CANDIDATE
        and all(sources.values())
        and int(sources[V23]["length"])
        != int(sources[GENERAL]["length"])
    ):
        return "future_source_plan"
    return None


def _dense_control_case(
    template: dict[str, Any],
    *,
    policy_label: str,
    suffix: str,
) -> dict[str, Any]:
    return {
        **template,
        "case_id": f"{template['case_id']}-{suffix}",
        "policy_label": policy_label,
        "ordinary_prefix_reuse": False,
        "reuse_enabled": False,
    }


def _paired_cases(
    prepared: dict[str, dict[str, Any]],
    *,
    include_dense_control: bool,
) -> list[dict[str, Any]]:
    """Order cases within each prompt hash, not across diverged trajectories."""

    if include_dense_control and DENSE not in prepared:
        raise ValueError("Dense control must be prepared before case dispatch")
    order = [arm for arm in ARMS if arm in prepared]
    prompt_hashes = {
        arm: token_ids_hash(prepared[arm]["prompt_ids"]) for arm in order
    }
    templates = {
        prompt_hash: next(
            (
                prepared[arm]["target"]
                for arm in order
                if prompt_hashes[arm] == prompt_hash
                and prepared[arm]["target"] is not None
            ),
            None,
        )
        for prompt_hash in set(prompt_hashes.values())
    }
    cases: list[dict[str, Any]] = []
    for arm in order:
        target = prepared[arm]["target"]
        if target is not None:
            cases.append(
                {
                    **target,
                    "ordinary_prefix_reuse": (
                        arm == V23 and TARGET_PREFIX_CANDIDATE
                    ),
                }
            )
        elif templates[prompt_hashes[arm]] is not None:
            # The server dispatches repeated identical prompts by case order.
            # Reserve a Dense slot only within this prompt-hash group.  Once
            # agent trajectories diverge, unrelated prompt hashes must not
            # consume one another's cases.
            cases.append(
                _dense_control_case(
                    templates[prompt_hashes[arm]],
                    policy_label=arm,
                    suffix=f"{arm}-dense-control",
                )
            )
    return cases


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
    value["model_kwargs"]["timeout"] = REQUEST_TIMEOUT_SECONDS
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
            "ordinary_prefix_reuse_enabled": TARGET_PREFIX_CANDIDATE,
            "ordinary_prefix_repair_tokens": 0,
            "ordinary_prefix_target_only": TARGET_PREFIX_CANDIDATE,
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
            (
                f"{V23} must veto the current General target at an online "
                "coding event selected by its frozen policy. Maintain "
                "candidate and General sources from real shared requests, "
                "then clone the repository before that target request."
            )
            if TARGET_VETO_CANDIDATE
            else (
                "Free-running General and the candidate agents diverged "
                "before treatment. Run one Dense shared agent until the first "
                "online point where "
            )
            + (
                "the candidate makes a critical-event Dense abstention while "
                "General registers a source, materialize the General source "
                "from that same request, snapshot the container, and only "
                "then branch."
                if ABSTENTION_CANDIDATE
                else ""
                if TARGET_VETO_CANDIDATE
                else (
                    "both policies select unequal reusable spans, materialize "
                    "both from that same request, snapshot the container, and "
                    "only then branch."
                )
            )
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
                (
                    "first online request where General registers a target "
                    f"and the candidate emits {TARGET_VETO_DENSE_MODE}"
                    if TARGET_VETO_CANDIDATE
                    else
                    "first online request where General registers a source "
                    "and the candidate emits critical_event_dense_abstain"
                    if ABSTENTION_CANDIDATE
                    else (
                        "first online request where General and candidate "
                        "both register a source and selected lengths differ"
                    )
                )
                + "; if no future target remains before the call limit, all "
                "arms inherit the same shared Dense outcome as an "
                "intention-to-treat tie"
            ),
            "shared_prefix_arm": "dense",
            "target_order_each_paired_step": list(ARMS),
            "container_clone": "docker commit after shared tool observation",
            "same_server": True,
            "mem_fraction_static": MEM_FRACTION_STATIC,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": {
            "branch_or_shared_completion": True,
            "branch_source_prompt_hash_identical": True,
            (
                "branch_target_plans_different"
                if TARGET_VETO_CANDIDATE
                else "branch_source_plans_different"
            ): True,
            "source_materializations_if_branched_min": (
                1 if ABSTENTION_CANDIDATE else 2
            ),
            (
                "general_target_copies_if_branched_min"
                if TARGET_VETO_CANDIDATE
                else "target_copies_each_arm_if_branched_min"
            ): 1,
            "candidate_critical_abstentions_if_enabled_min": (
                1 if ABSTENTION_CANDIDATE else 0
            ),
            "candidate_target_vetoes_if_enabled_min": (
                1 if TARGET_VETO_CANDIDATE else 0
            ),
            "dense_control_requests_if_enabled_min": (
                1 if INCLUDE_DENSE_CONTROL else 0
            ),
            "target_fallbacks": 0,
            "first_branched_prompt_hash_identical": True,
            (
                "all_arms_reach_scored_terminal_outcome"
                if ALLOW_EMPTY_SUBMISSION_OUTCOME
                else "both_branches_produce_submission"
            ): True,
            **(
                {"empty_patch_is_official_unresolved_outcome": True}
                if ALLOW_EMPTY_SUBMISSION_OUTCOME
                else {}
            ),
            "official_evaluation_required_before_accuracy_claim": True,
            **({"candidate_branch_required": True} if REQUIRE_BRANCH else {}),
        },
        "inputs": {
            "dataset": str(DATASET_ROOT / "test.jsonl"),
            "dataset_sha256": sha256(DATASET_ROOT / "test.jsonl"),
            "evaluation_registration": str(EVAL_REGISTRATION),
            "evaluation_registration_sha256": sha256(EVAL_REGISTRATION),
            "evaluation_snapshot": str(EVAL_SNAPSHOT),
            "evaluation_snapshot_sha256": sha256(EVAL_SNAPSHOT),
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
    *,
    include_dense_control: bool = False,
    dense_model: BridgeReuseLitellmModel | None = None,
    dense_messages: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    prepared = {
        arm: models[arm].prepare_reuse_query(
            messages[arm],
            write_sidecar=False,
        )
        for arm in REUSE_ARMS
    }
    if include_dense_control:
        if dense_model is None or dense_messages is None:
            raise ValueError("Dense model and messages are required")
        prepared[DENSE] = dense_model.prepare_reuse_query(
            dense_messages,
            write_sidecar=False,
        )
    _install_prepared_pair(
        models,
        prepared,
        include_dense_control=include_dense_control,
    )
    return prepared


def _install_prepared_pair(
    models: dict[str, BridgeReuseLitellmModel],
    prepared: dict[str, dict[str, Any]],
    *,
    include_dense_control: bool,
) -> None:
    """Publish already-prepared sources and ordered target cases atomically."""

    cases = _paired_cases(
        prepared,
        include_dense_control=include_dense_control,
    )
    models[V23]._atomic_sidecar_update(
        sources=[
            prepared[arm]["source"]
            for arm in REUSE_ARMS
            if prepared[arm]["source"] is not None
        ],
        cases=cases,
        release_source_ids=[
            source_id
            for arm in REUSE_ARMS
            for source_id in prepared[arm]["releases"]
        ],
    )


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
        for arm in REUSE_ARMS
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
    initial_branch_prepared: dict[str, dict[str, Any]] | None = None
    branch_agent_elapsed = {arm: 0.0 for arm in ARMS}
    shadow_ledger = run_dir / "SHADOW_LEDGER.jsonl"
    try:
        shared_env = get_sb_environment(copy.deepcopy(config), instance)
        shared = _initialize_agent(
            dense_model,
            shared_env,
            config,
            instance["problem_statement"],
        )
        while shared.n_calls < shared.config.step_limit:
            call = shared.n_calls + 1
            shadow = {
                arm: models[arm].prepare_reuse_query(
                    shared.messages,
                    write_sidecar=False,
                )
                for arm in REUSE_ARMS
            }
            hashes = {
                arm: token_ids_hash(shadow[arm]["prompt_ids"])
                for arm in REUSE_ARMS
            }
            sources = {
                arm: shadow[arm]["source"] for arm in REUSE_ARMS
            }
            branch_kind = _branch_kind(shadow)
            with shadow_ledger.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "call": call,
                            "branch_kind": branch_kind,
                            "policy_modes": {
                                arm: _policy_mode(shadow[arm])
                                for arm in REUSE_ARMS
                            },
                            "prompt_hashes": hashes,
                            "source_registered": {
                                arm: shadow[arm]["source"] is not None
                                for arm in REUSE_ARMS
                            },
                            "target_registered": {
                                arm: shadow[arm]["target"] is not None
                                for arm in REUSE_ARMS
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            target_veto_eligible = branch_kind == "current_target_veto"
            source_plan_eligible = branch_kind == "future_source_plan"
            if (
                target_veto_eligible
                and shared.n_calls < shared.config.step_limit
            ):
                branch = {
                    "kind": "current_target_veto",
                    "shared_calls": shared.n_calls,
                    "branch_request_index": call,
                    "source_prompt_hash": hashes[V23],
                    "source_lengths": {
                        arm: (
                            int(sources[arm]["length"])
                            if sources[arm] is not None
                            else None
                        )
                        for arm in REUSE_ARMS
                    },
                    "source_ids": {
                        arm: (
                            str(sources[arm]["source_id"])
                            if sources[arm] is not None
                            else None
                        )
                        for arm in REUSE_ARMS
                    },
                    "source_decision_modes": {
                        arm: _policy_mode(shadow[arm])
                        for arm in REUSE_ARMS
                    },
                    "target_registered": {
                        arm: shadow[arm]["target"] is not None
                        for arm in REUSE_ARMS
                    },
                }
                initial_branch_prepared = shadow
                break
            if source_plan_eligible and shared.n_calls < shared.config.step_limit:
                models[V23]._atomic_sidecar_update(
                    sources=[
                        sources[arm]
                        for arm in REUSE_ARMS
                        if sources[arm] is not None
                    ],
                )
            elif (
                TARGET_VETO_CANDIDATE
                and shared.n_calls < shared.config.step_limit
            ):
                # Capture sources from this real shared Dense request.  No
                # replay/prefetch request is added.  Planned targets skipped
                # by the shared prefix are retired rather than leaked.
                skipped_targets = [
                    str(shadow[arm]["target"]["source_id"])
                    for arm in REUSE_ARMS
                    if shadow[arm]["target"] is not None
                ]
                models[V23]._atomic_sidecar_update(
                    sources=[
                        sources[arm]
                        for arm in REUSE_ARMS
                        if sources[arm] is not None
                    ],
                    release_source_ids=[
                        *skipped_targets,
                        *[
                            source_id
                            for arm in REUSE_ARMS
                            for source_id in shadow[arm]["releases"]
                        ],
                    ],
                )
            else:
                for arm in REUSE_ARMS:
                    models[arm]._pending_source = None
            _normal_step(shared)
            if shared.messages[-1].get("role") == "exit":
                break
            if source_plan_eligible:
                branch = {
                    "kind": "future_source_plan",
                    "shared_calls": call,
                    "branch_request_index": call + 1,
                    "source_prompt_hash": hashes[V23],
                    "source_lengths": {
                        arm: (
                            int(sources[arm]["length"])
                            if sources[arm] is not None
                            else None
                        )
                        for arm in REUSE_ARMS
                    },
                    "source_ids": {
                        arm: (
                            str(sources[arm]["source_id"])
                            if sources[arm] is not None
                            else None
                        )
                        for arm in REUSE_ARMS
                    },
                    "source_decision_modes": {
                        arm: _policy_mode(shadow[arm])
                        for arm in REUSE_ARMS
                    },
                }
                break

        first_branch_hashes: dict[str, str] = {}
        if branch is None:
            if shared.messages[-1].get("role") != "exit":
                shared.add_messages(
                    dense_model.format_message(
                        role="exit",
                        content="LimitsExceeded",
                        extra={
                            "exit_status": "LimitsExceeded",
                            "submission": "",
                        },
                    )
                )
            agents = {arm: shared for arm in ARMS}
        else:
            branch_envs, snapshot_image = _docker_clone_environments(
                shared_env,
                config,
            )
            agents = {
                arm: _clone_agent(
                    shared,
                    (
                        models[arm]
                        if arm in REUSE_ARMS
                        else _model(
                            config,
                            arm=DENSE,
                            manifest=None,
                            ledger=output
                            / DENSE
                            / "CLIENT_LEDGER.jsonl",
                        )
                    ),
                    branch_envs[arm],
                    config,
                )
                for arm in ARMS
            }
            shared_env.cleanup()
            shared_env = None

            if initial_branch_prepared is not None:
                prepared = dict(initial_branch_prepared)
                if DENSE in agents:
                    prepared[DENSE] = agents[DENSE].model.prepare_reuse_query(
                        agents[DENSE].messages,
                        write_sidecar=False,
                    )
                _install_prepared_pair(
                    models,
                    prepared,
                    include_dense_control=DENSE in agents,
                )
                first_branch_hashes = {
                    arm: token_ids_hash(prepared[arm]["prompt_ids"])
                    for arm in prepared
                }
                for arm in ARMS:
                    started = time.perf_counter()
                    _prepared_step(agents[arm], prepared[arm])
                    branch_agent_elapsed[arm] += (
                        time.perf_counter() - started
                    )

            while True:
                active = [
                    arm
                    for arm in ARMS
                    if agents[arm].messages[-1].get("role") != "exit"
                    and agents[arm].n_calls
                    < agents[arm].config.step_limit
                ]
                if not active:
                    break
                active_reuse = [
                    arm for arm in REUSE_ARMS if arm in active
                ]
                if len(active_reuse) == 2:
                    prepared = _prepare_pair(
                        models,
                        {
                            arm: agents[arm].messages
                            for arm in REUSE_ARMS
                        },
                        include_dense_control=DENSE in active,
                        dense_model=(
                            agents[DENSE].model
                            if DENSE in active
                            else None
                        ),
                        dense_messages=(
                            agents[DENSE].messages
                            if DENSE in active
                            else None
                        ),
                    )
                    if not first_branch_hashes:
                        first_branch_hashes = {
                            arm: token_ids_hash(
                                prepared[arm]["prompt_ids"]
                            )
                            for arm in prepared
                        }
                    for arm in REUSE_ARMS:
                        started = time.perf_counter()
                        _prepared_step(agents[arm], prepared[arm])
                        branch_agent_elapsed[arm] += (
                            time.perf_counter() - started
                        )
                    if DENSE in active:
                        started = time.perf_counter()
                        _prepared_step(agents[DENSE], prepared[DENSE])
                        branch_agent_elapsed[DENSE] += (
                            time.perf_counter() - started
                        )
                else:
                    for arm in active:
                        started = time.perf_counter()
                        _normal_step(agents[arm])
                        branch_agent_elapsed[arm] += (
                            time.perf_counter() - started
                        )

        for arm in ARMS:
            if agents[arm].messages[-1].get("role") != "exit":
                agents[arm].add_messages(
                    agents[arm].model.format_message(
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
    dense_controls = [
        row
        for row in server
        if row.get("event") == "target_dense_control"
    ]
    first_hash_equal = (
        len(set(first_branch_hashes.values())) == 1
        if first_branch_hashes
        else False
    )
    submissions = {arm: _submission(agents[arm]) for arm in ARMS}
    exit_status = {
        arm: agents[arm].messages[-1].get("extra", {}).get("exit_status")
        for arm in ARMS
    }
    copy_counts = {
        arm: sum(row.get("policy_label") == arm for row in copies)
        for arm in ARMS
    }
    candidate_client_path = output / V23 / "CLIENT_LEDGER.jsonl"
    candidate_client = (
        load_jsonl(candidate_client_path)
        if candidate_client_path.exists()
        else []
    )
    candidate_critical_abstentions = (
        int(
            branch is not None
            and branch.get("source_decision_modes", {}).get(V23)
            == "critical_event_dense_abstain"
        )
        + sum(
        _policy_mode(row) == "critical_event_dense_abstain"
        for row in candidate_client
        )
    )
    candidate_target_vetoes = sum(
        _is_target_veto_record(row) for row in candidate_client
    )
    gates = {
        "branch_or_shared_completion": True,
        **(
            {"candidate_branch_required": branch is not None}
            if REQUIRE_BRANCH
            else {}
        ),
        "branch_source_prompt_hash_identical": branch is None
        or branch["source_prompt_hash"] is not None,
        (
            "branch_target_plans_different"
            if TARGET_VETO_CANDIDATE
            else "branch_source_plans_different"
        ): branch is None
        or (
            len(set(branch["target_registered"].values())) == 2
            if TARGET_VETO_CANDIDATE
            else len(set(branch["source_lengths"].values())) == 2
        ),
        "source_materializations_if_branched_min": branch is None
        or len(materialized) >= (1 if ABSTENTION_CANDIDATE else 2),
        (
            "general_target_copies_if_branched_min"
            if TARGET_VETO_CANDIDATE
            else "target_copies_each_arm_if_branched_min"
        ): branch is None
        or (
            copy_counts[GENERAL] >= 1
            and (
                ABSTENTION_CANDIDATE
                or TARGET_VETO_CANDIDATE
                or copy_counts[V23] >= 1
            )
        ),
        "candidate_critical_abstentions_if_enabled_min": (
            not ABSTENTION_CANDIDATE
            or branch is None
            or candidate_critical_abstentions >= 1
        ),
        "candidate_target_vetoes_if_enabled_min": (
            not TARGET_VETO_CANDIDATE
            or branch is None
            or candidate_target_vetoes >= 1
        ),
        "dense_control_requests_if_enabled_min": (
            not INCLUDE_DENSE_CONTROL
            or branch is None
            or len(dense_controls) >= 1
        ),
        "target_fallbacks": len(fallbacks) == 0,
        "first_branched_prompt_hash_identical": branch is None
        or first_hash_equal,
        (
            "all_arms_reach_scored_terminal_outcome"
            if ALLOW_EMPTY_SUBMISSION_OUTCOME
            else "both_branches_produce_submission"
        ): (
            all(
                status in {"Submitted", "LimitsExceeded"}
                for status in exit_status.values()
            )
            if ALLOW_EMPTY_SUBMISSION_OUTCOME
            else all(submissions.values())
        ),
        **(
            {"empty_patch_is_official_unresolved_outcome": True}
            if ALLOW_EMPTY_SUBMISSION_OUTCOME
            else {}
        ),
        "official_evaluation_required_before_accuracy_claim": True,
    }
    result = {
        "completed_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "registration_status": registration["status"],
        "branch": branch,
        "first_branch_prompt_hashes": first_branch_hashes,
        "calls": {arm: agents[arm].n_calls for arm in ARMS},
        "exit_status": exit_status,
        "submission_bytes": {
            arm: len(submissions[arm].encode()) for arm in ARMS
        },
        "branched_agent_elapsed_seconds": branch_agent_elapsed,
        "server": {
            "source_materializations": len(materialized),
            "target_copies": len(copies),
            "copy_counts": copy_counts,
            "dense_control_requests": len(dense_controls),
            "target_fallbacks": len(fallbacks),
            "candidate_critical_abstentions": (
                candidate_critical_abstentions
            ),
            "candidate_target_vetoes": candidate_target_vetoes,
        },
        "gates": gates,
        "accuracy": (
            "NOT_YET_MEASURED: official SWE-bench evaluation is a separate "
            "required stage."
        ),
    }
    write_json(result_path, result)
    return result


def evaluate(output: Path) -> dict[str, Any]:
    runtime = read_json(output / "V25_RESULT.json")
    official: dict[str, Any] = {}
    for arm in ARMS:
        path = output / arm / "OFFICIAL_RESULT.json"
        if path.exists():
            official[arm] = read_json(path)
        else:
            official[arm] = run_official_evaluation(
                output=output,
                run_dir=output / arm,
                arm=f"v25-paired-{arm}",
                instance_ids=[INSTANCE_ID],
                registration=EVAL_REGISTRATION,
                snapshot=EVAL_SNAPSHOT,
            )
    return {
        "runtime_status": runtime["status"],
        "official": official,
        "summary": summarize_official(output),
    }


def summarize_official(output: Path) -> dict[str, Any]:
    runtime = read_json(output / "V25_RESULT.json")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        result = read_json(output / arm / "OFFICIAL_RESULT.json")
        report = result["report"]
        client = [
            row
            for row in load_jsonl(output / arm / "CLIENT_LEDGER.jsonl")
            if row.get("event") == "request_complete"
        ]
        ttfts = [
            1000 * float(row["ttft_seconds"])
            for row in client
            if row.get("ttft_seconds") is not None
        ]
        arms[arm] = {
            "resolved": int(report["resolved_instances"]),
            "total": int(report["total_instances"]),
            "empty_patch": int(report["empty_patch_instances"]),
            "report_path": result["report_path"],
            "branched_model_requests": len(client),
            "branched_model_elapsed_seconds": sum(
                float(row["request_elapsed_seconds"]) for row in client
            ),
            "median_ttft_ms": (
                statistics.median(ttfts) if ttfts else None
            ),
            "branched_agent_elapsed_seconds": runtime.get(
                "branched_agent_elapsed_seconds", {}
            ).get(arm),
        }
    candidate_resolved = arms[V23]["resolved"]
    general_resolved = arms[GENERAL]["resolved"]
    if candidate_resolved > general_resolved:
        interpretation = (
            f"Official paired candidate-only resolution: {V23} resolved 1/1 "
            "while General did not.  Because history, repository snapshot, "
            "first branched prompt, server, and model are shared, this is "
            "causal evidence for the post-branch reuse policy on this "
            "instance, not a population-level accuracy estimate."
        )
    elif candidate_resolved == general_resolved:
        interpretation = (
            "The arms have equal official resolution on this instance.  The "
            "result does not rank their accuracy or establish a population "
            "effect."
        )
    else:
        interpretation = (
            "Official paired General-only resolution on this instance.  "
            f"{V23} must be treated as an accuracy damage until replicated."
        )
    value = {
        "summarized_at_utc": utc_now(),
        "instance_id": INSTANCE_ID,
        "arms": arms,
        "paired_accuracy_interpretation": interpretation,
        "latency_caveat": (
            f"The fixed {'-then-'.join(ARMS)} order makes per-request TTFT an "
            "order-sensitive diagnostic, not an unbiased speed estimate. "
            "Summed model elapsed time includes decode behavior and is useful "
            "for this paired outcome but still requires order replication."
        ),
    }
    write_json(output / "V25_OFFICIAL_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "run", "evaluate", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
    elif args.command == "run":
        value = run(args.output)
    elif args.command == "evaluate":
        value = evaluate(args.output)
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
