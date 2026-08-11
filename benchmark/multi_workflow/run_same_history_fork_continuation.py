#!/usr/bin/env python3
"""Causal same-history fork for natural coding-aware KV reuse.

Independent agent arms can choose different actions before the first KV-copy
request.  Such runs measure an end-to-end arm effect, but they cannot attribute
an accuracy difference to lossy KV reuse.  This runner instead freezes a real
policy trajectory immediately before its first treated request, reconstructs
the same official SWE-bench workspace in two fresh containers, and forks only
the target inference into Dense versus natural-code reuse.

For the reuse arm, earlier frozen prompts are replayed with one discarded
diagnostic token solely to materialize the registered natural KV sources.  The
diagnostic tokens are never added to the agent history, replay time is excluded
from deployment latency claims, and no online prefetch path is enabled.  Both
arms then continue freely from the target response to an official patch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded, Submitted
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name
from minisweagent.utils.serialize import recursive_merge

from benchmark.multi_workflow.bridge_reuse_litellm_model import (
    BridgeReuseLitellmModel,
    token_ids_hash,
)
from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    CHAT_TEMPLATE,
    CONFIG,
    MODEL,
    TOKENIZER_JSON,
    init_manifest,
    launch_server,
    load_jsonl,
    run_official_evaluation,
    stop_server,
    summarize_runtime,
)
from benchmark.multi_workflow.run_swebench_with_limit_patch_capture import (
    fill_empty_submission,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    generate_one,
)


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
SOURCE_CAMPAIGN = (
    ARTIFACTS / "impactkv_natural_code_cost_discordant7_repeat_20260809"
)
SOURCE_TRAJECTORIES = (
    SOURCE_CAMPAIGN / "online/coding_natural_code_cost/full_7"
)
SOURCE_SNAPSHOT = SOURCE_CAMPAIGN / "FROZEN_DISCORDANT7.json"
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_natural_code_cost_same_history_fork_20260809"
)
POLICY_ARM = "coding_natural_code_cost"
ARMS = ("dense", POLICY_ARM)
ARM_PORTS = {"dense": 32340, POLICY_ARM: 32341}
STEP_LIMIT = 32
CAUSAL_SCOPE = (
    "Outcome-selected two-task mechanism canary.  It can test whether KV "
    "treatment changes these continuations, but cannot estimate population "
    "accuracy."
)

# Both are stable repeat rescues and both received a physical treatment.  The
# request index is frozen from TREATMENT_ATTRIBUTION.json before this run.
FORK_TASKS = {
    "sympy__sympy-14711": 6,
    "sympy__sympy-22914": 5,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def trajectory_path(instance_id: str) -> Path:
    return SOURCE_TRAJECTORIES / instance_id / f"{instance_id}.traj.json"


def assistant_request_prefixes(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    return [
        copy.deepcopy(messages[:index])
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]


def prefix_actions(
    messages: list[dict[str, Any]], target_request: int
) -> list[dict[str, Any]]:
    """Return the recorded actions strictly before the fork request."""

    actions: list[dict[str, Any]] = []
    request_index = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        request_index += 1
        if request_index >= target_request:
            break
        for action in message.get("extra", {}).get("actions", []):
            actions.append(copy.deepcopy(action))
    return actions


def frozen_task_rows() -> dict[str, dict[str, Any]]:
    rows = read_json(SOURCE_SNAPSHOT)
    return {str(row["instance_id"]): row for row in rows}


def registration_payload(output: Path) -> dict[str, Any]:
    rows = frozen_task_rows()
    tasks = []
    for instance_id, target_request in FORK_TASKS.items():
        trajectory = read_json(trajectory_path(instance_id))
        messages = trajectory["messages"]
        prefixes = assistant_request_prefixes(messages)
        if target_request > len(prefixes):
            raise AssertionError(f"missing q{target_request}: {instance_id}")
        frozen_prefix = prefixes[target_request - 1]
        actions = prefix_actions(messages, target_request)
        if any(
            any(marker in str(action.get("command") or "") for marker in (
                ">", "python -", "apply_patch", "git checkout", "rm ", "mv ",
            ))
            for action in actions
        ):
            raise AssertionError(
                f"fork prefix is not conservatively read-only: {instance_id}"
            )
        tasks.append(
            {
                "instance_id": instance_id,
                "fork_request_index": target_request,
                "frozen_prefix_messages": len(frozen_prefix),
                "frozen_prefix_sha256": canonical_hash(frozen_prefix),
                "source_trajectory": str(trajectory_path(instance_id)),
                "source_trajectory_sha256": hashlib.sha256(
                    trajectory_path(instance_id).read_bytes()
                ).hexdigest(),
                "prefix_commands": [
                    str(action.get("command") or "") for action in actions
                ],
                "image": get_swebench_docker_image_name(rows[instance_id]),
            }
        )
    return {
        "status": "REGISTERED_BEFORE_CAUSAL_FORK_GPU_OUTCOMES",
        "registered_at_utc": utc_now(),
        "purpose": (
            "Hold messages and official container state fixed through the "
            "first eligible natural-code KV-copy request, then fork only "
            "Dense versus lossy reuse and score the final patches."
        ),
        "selection": {
            "outcome_selected": True,
            "population_claim_allowed": False,
            "reason": (
                "two stable repeat rescues with a physical treatment; this "
                "is a causal-mechanism canary, not an accuracy estimate"
            ),
            "tasks": tasks,
        },
        "protocol": {
            "arms": list(ARMS),
            "arm_execution_order": list(ARMS),
            "model": MODEL,
            "temperature": 0,
            "step_limit_total_including_frozen_prefix": STEP_LIMIT,
            "rolling_history_groups": 6,
            "same_frozen_messages_at_fork": True,
            "fresh_official_container_per_task_and_arm": True,
            "prefix_workspace_commands_replayed": True,
            "prefix_workspace_must_remain_git_clean": True,
            "reuse_source_materialization": (
                "one discarded diagnostic token per earlier frozen request"
            ),
            "diagnostic_tokens_added_to_history": False,
            "prefix_replay_counted_as_online_latency": False,
            "online_prefetch": False,
            "ordinary_radix_prefix_reuse": False,
            "official_metric": "SWE-bench resolved",
        },
        "gates": {
            "target_prompt_hash_identical_across_arms": True,
            "frozen_prefix_hash_identical_across_arms": True,
            "prefix_workspace_git_diff_empty": True,
            "policy_target_registered": True,
            "policy_physical_copy_events_min": 1,
            "dense_target_registered": False,
            "report_accuracy_without_population_generalization": True,
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }


def prepare(output: Path) -> dict[str, Any]:
    path = output / "CAMPAIGN_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    value = registration_payload(output)
    write_json(path, value)
    write_json(
        output / "BRIDGE_REGISTRATION.json",
        {
            "schema_version": 1,
            "registration_id": output.name,
            "registered_at_utc": value["registered_at_utc"],
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [
                {"instance_id": instance_id} for instance_id in FORK_TASKS
            ],
        },
    )
    return value


def merged_config() -> dict[str, Any]:
    # The production launcher passes both configs in this order.  In
    # particular, the built-in file supplies the 10k-character observation
    # renderer that is visible in the frozen histories.
    base = yaml.safe_load(
        (builtin_config_dir / "benchmarks/swebench.yaml").read_text(
            encoding="utf-8"
        )
    )
    override = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return recursive_merge(base, override)


def make_model(
    *, arm: str, port: int, manifest: Path, ledger: Path
) -> BridgeReuseLitellmModel:
    config = merged_config()["model"]
    config = copy.deepcopy(config)
    config.update(
        reuse_arm=arm,
        reuse_client_ledger_path=ledger,
        reuse_manifest_path=(manifest if arm != "dense" else None),
    )
    config["model_kwargs"].update(
        api_base=f"http://127.0.0.1:{port}/v1",
        api_key="EMPTY",
        max_tokens=2048,
        temperature=0.0,
        timeout=900,
    )
    return BridgeReuseLitellmModel(**config)


class ForkContinuationAgent(DefaultAgent):
    """Stock control flow with terminal tracked-diff capture."""

    def query(self) -> dict[str, Any]:
        try:
            return super().query()
        except LimitsExceeded as error:
            fill_empty_submission(self, error)
            raise

    def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return super().execute_actions(message)
        except Submitted as error:
            fill_empty_submission(self, error)
            raise


def replay_workspace_prefix(
    *,
    agent: ForkContinuationAgent,
    frozen_messages: list[dict[str, Any]],
    target_request: int,
) -> dict[str, Any]:
    """Replay frozen tool actions and verify their observation messages."""

    observation_checks = []
    request_index = 0
    for index, message in enumerate(frozen_messages):
        if message.get("role") != "assistant":
            continue
        request_index += 1
        if request_index >= target_request:
            break
        actions = message.get("extra", {}).get("actions", [])
        outputs = [agent.env.execute(action) for action in actions]
        generated = agent.model.format_observation_messages(
            message, outputs, agent.get_template_vars()
        )
        reference = []
        cursor = index + 1
        while cursor < len(frozen_messages):
            if frozen_messages[cursor].get("role") != "tool":
                break
            reference.append(frozen_messages[cursor])
            cursor += 1
        generated_public = [
            {key: value for key, value in row.items() if key != "extra"}
            for row in generated
        ]
        reference_public = [
            {key: value for key, value in row.items() if key != "extra"}
            for row in reference
        ]
        observation_checks.append(
            {
                "request_index": request_index,
                "commands": [str(row.get("command") or "") for row in actions],
                "returncodes": [int(row.get("returncode", -1)) for row in outputs],
                "observation_exact_match": generated_public == reference_public,
                "generated_observation_sha256": canonical_hash(generated_public),
                "reference_observation_sha256": canonical_hash(reference_public),
            }
        )
    status = agent.env.execute(
        {"command": "git status --porcelain=v1 --untracked-files=all"},
        timeout=120,
    )
    diff = agent.env.execute(
        {"command": "git diff --binary --no-ext-diff"}, timeout=120
    )
    if status.get("returncode") != 0 or diff.get("returncode") != 0:
        raise RuntimeError("failed to fingerprint reconstructed workspace")
    return {
        "observation_checks": observation_checks,
        "all_observations_exact": all(
            row["observation_exact_match"] for row in observation_checks
        ),
        "git_status": str(status.get("output") or ""),
        "git_diff": str(diff.get("output") or ""),
        "git_status_sha256": hashlib.sha256(
            str(status.get("output") or "").encode()
        ).hexdigest(),
        "git_diff_sha256": hashlib.sha256(
            str(diff.get("output") or "").encode()
        ).hexdigest(),
    }


def advance_frozen_model_prefix(
    *,
    model: BridgeReuseLitellmModel,
    prefixes: list[list[dict[str, Any]]],
    target_request: int,
    arm: str,
    port: int,
) -> list[dict[str, Any]]:
    """Advance planning state; materialize reuse sources with ignored tokens."""

    rows = []
    for request_index, prefix in enumerate(
        prefixes[: target_request - 1], start=1
    ):
        prepared = model.prepare_reuse_query(prefix)
        row = {
            "request_index": request_index,
            "prompt_hash": token_ids_hash(prepared["prompt_ids"]),
            "prompt_tokens": len(prepared["prompt_ids"]),
            "target_registered": bool(prepared.get("targets")),
            "source_registered": bool(prepared.get("sources")),
            "diagnostic_request_issued": arm != "dense",
        }
        if arm != "dense":
            # Source materialization requires the real earlier prompt to reach
            # SGLang.  One ignored token suffices; it never enters history.
            targets = prepared.get("targets") or []
            request_key = (
                str(targets[0].get("target_group_id") or targets[0]["case_id"])
                if targets
                else f"fork-prefix-{model._instance_nonce}-q{request_index}"
            )
            generate_one(
                base_url=f"http://127.0.0.1:{port}",
                input_ids=prepared["prompt_ids"],
                key=request_key,
            )
        rows.append(row)
    return rows


def run_agent_from_prepared_target(
    *,
    agent: ForkContinuationAgent,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    """Execute the fork request, then resume the stock agent loop."""

    target_message: dict[str, Any] | None = None
    try:
        agent.n_calls += 1
        target_message = agent.model.execute_prepared_reuse_query(prepared)
        agent.cost += target_message.get("extra", {}).get("cost", 0.0)
        agent.add_messages(target_message)
        agent.execute_actions(target_message)
    except InterruptAgentFlow as error:
        agent.add_messages(*error.messages)

    while not agent.messages or agent.messages[-1].get("role") != "exit":
        try:
            agent.step()
        except InterruptAgentFlow as error:
            agent.add_messages(*error.messages)
        finally:
            agent.save(agent.config.output_path)
    return {
        **agent.messages[-1].get("extra", {}),
        "target_message": target_message,
    }


def run_task(
    *,
    output: Path,
    arm: str,
    port: int,
    manifest: Path,
    instance: dict[str, Any],
) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    target_request = FORK_TASKS[instance_id]
    task_dir = output / arm / instance_id
    result_path = task_dir / "FORK_RESULT.json"
    if result_path.exists():
        return read_json(result_path)
    task_dir.mkdir(parents=True, exist_ok=True)

    trajectory = read_json(trajectory_path(instance_id))
    messages = trajectory["messages"]
    prefixes = assistant_request_prefixes(messages)
    frozen_prefix = prefixes[target_request - 1]
    model = make_model(
        arm=arm,
        port=port,
        manifest=manifest,
        ledger=output / arm / "CLIENT_LEDGER.jsonl",
    )
    environment_config = copy.deepcopy(merged_config()["environment"])
    environment_config.update(
        image=get_swebench_docker_image_name(instance),
        cwd="/testbed",
        timeout=120,
        container_timeout="30m",
    )
    env = DockerEnvironment(**environment_config)
    agent_config = copy.deepcopy(merged_config()["agent"])
    agent_config.update(
        step_limit=STEP_LIMIT,
        output_path=task_dir / f"{instance_id}.traj.json",
    )
    agent = ForkContinuationAgent(model, env, **agent_config)
    agent.extra_template_vars = {"task": instance["problem_statement"]}
    agent.messages = copy.deepcopy(frozen_prefix)
    agent.n_calls = target_request - 1
    agent._start_time = time.time()
    try:
        workspace = replay_workspace_prefix(
            agent=agent,
            frozen_messages=messages,
            target_request=target_request,
        )
        if workspace["git_status"] or workspace["git_diff"]:
            raise AssertionError(
                f"non-clean reconstructed prefix for {instance_id}/{arm}"
            )
        replay = advance_frozen_model_prefix(
            model=model,
            prefixes=prefixes,
            target_request=target_request,
            arm=arm,
            port=port,
        )
        prepared = model.prepare_reuse_query(frozen_prefix)
        target = {
            "prompt_hash": token_ids_hash(prepared["prompt_ids"]),
            "prompt_tokens": len(prepared["prompt_ids"]),
            "target_registered": bool(prepared.get("targets")),
            "target_islands": len(prepared.get("targets") or []),
            "copied_tokens_planned": sum(
                int(row["length"]) for row in prepared.get("targets") or []
            ),
            "source_registered": bool(prepared.get("sources")),
            "frozen_prefix_sha256": canonical_hash(frozen_prefix),
        }
        if arm == "dense" and target["target_registered"]:
            raise AssertionError("Dense unexpectedly registered treatment")
        if arm != "dense" and not target["target_registered"]:
            raise AssertionError(
                f"policy target missing at fork: {instance_id} q{target_request}"
            )
        outcome = run_agent_from_prepared_target(agent=agent, prepared=prepared)
        submission = str(outcome.get("submission") or "")
        final = {
            "arm": arm,
            "instance_id": instance_id,
            "fork_request_index": target_request,
            "completed_at_utc": utc_now(),
            "workspace": workspace,
            "prefix_model_replay": replay,
            "target": target,
            "exit_status": outcome.get("exit_status"),
            "submission": submission,
            "submission_sha256": hashlib.sha256(submission.encode()).hexdigest(),
            "submission_characters": len(submission),
            "model_calls_total": agent.n_calls,
        }
        write_json(result_path, final)
        preds_path = output / arm / "preds.json"
        preds = read_json(preds_path) if preds_path.exists() else {}
        preds[instance_id] = {
            "model_name_or_path": f"same-history-fork-{arm}",
            "instance_id": instance_id,
            "model_patch": submission,
        }
        write_json(preds_path, preds)
        return final
    finally:
        env.cleanup()


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    prepare(output)
    arm_dir = output / arm
    result_path = arm_dir / "ARM_RESULT.json"
    if result_path.exists():
        return read_json(result_path)
    arm_dir.mkdir(parents=True, exist_ok=True)
    manifest = init_manifest(arm_dir, arm)
    process, log = launch_server(
        run_dir=arm_dir,
        arm=arm,
        manifest=manifest,
        port=port,
    )
    rows = []
    by_id = frozen_task_rows()
    try:
        for instance_id in FORK_TASKS:
            rows.append(
                run_task(
                    output=output,
                    arm=arm,
                    port=port,
                    manifest=manifest,
                    instance=by_id[instance_id],
                )
            )
    finally:
        stop_server(process, log)
    runtime = summarize_runtime(arm_dir, arm)
    value = {
        "arm": arm,
        "completed_at_utc": utc_now(),
        "tasks": rows,
        "runtime": runtime,
    }
    write_json(result_path, value)
    return value


def evaluate_arm(output: Path, arm: str) -> dict[str, Any]:
    result_path = output / arm / "OFFICIAL_RESULT.json"
    if result_path.exists():
        return read_json(result_path)
    return run_official_evaluation(
        output=output,
        run_dir=output / arm,
        arm=f"same_history_fork_{arm}",
        instance_ids=list(FORK_TASKS),
        registration=output / "BRIDGE_REGISTRATION.json",
        snapshot=SOURCE_SNAPSHOT,
    )


def resolved_ids(official: dict[str, Any]) -> set[str]:
    report = official.get("report") or {}
    return set(report.get("resolved_ids") or [])


def assistant_message_at_request(
    trajectory: dict[str, Any], request_index: int
) -> dict[str, Any]:
    assistants = [
        row
        for row in trajectory.get("messages", [])
        if row.get("role") == "assistant"
    ]
    if not 1 <= request_index <= len(assistants):
        raise IndexError(request_index)
    return assistants[request_index - 1]


def public_response_signature(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": message.get("content"),
        # Tool-call IDs are process-local bookkeeping, not model behavior.
        "actions": [
            {
                key: value
                for key, value in action.items()
                if key != "tool_call_id"
            }
            for action in message.get("extra", {}).get("actions", [])
        ],
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = prepare(output)
    arms = {arm: read_json(output / arm / "ARM_RESULT.json") for arm in ARMS}
    official = {arm: evaluate_arm(output, arm) for arm in ARMS}
    dense_ids = resolved_ids(official["dense"])
    policy_ids = resolved_ids(official[POLICY_ARM])
    server = load_jsonl(output / POLICY_ARM / "SERVER_LEDGER.jsonl")
    copies = [row for row in server if row.get("event") == "target_copied"]
    fallbacks = [row for row in server if row.get("event") == "target_fallback"]
    manifest = read_json(
        output / POLICY_ARM / "DYNAMIC_MANIFEST.json"
    )

    task_rows = []
    for instance_id in FORK_TASKS:
        dense = next(
            row for row in arms["dense"]["tasks"]
            if row["instance_id"] == instance_id
        )
        policy = next(
            row for row in arms[POLICY_ARM]["tasks"]
            if row["instance_id"] == instance_id
        )
        prompt_equal = (
            dense["target"]["prompt_hash"] == policy["target"]["prompt_hash"]
        )
        prefix_equal = (
            dense["target"]["frozen_prefix_sha256"]
            == policy["target"]["frozen_prefix_sha256"]
        )
        if not prompt_equal or not prefix_equal:
            raise AssertionError(f"fork identity mismatch: {instance_id}")
        request_index = FORK_TASKS[instance_id]
        dense_target_message = assistant_message_at_request(
            read_json(
                output / "dense" / instance_id / f"{instance_id}.traj.json"
            ),
            request_index,
        )
        policy_target_message = assistant_message_at_request(
            read_json(
                output
                / POLICY_ARM
                / instance_id
                / f"{instance_id}.traj.json"
            ),
            request_index,
        )
        dense_treatment = dense_target_message.get("extra", {}).get(
            "reuse_treatment", {}
        )
        policy_treatment = policy_target_message.get("extra", {}).get(
            "reuse_treatment", {}
        )
        dense_ttft_ms = 1000 * float(dense_treatment["ttft_seconds"])
        policy_ttft_ms = 1000 * float(policy_treatment["ttft_seconds"])
        target_case_ids = {
            str(case["case_id"])
            for case in manifest.get("cases", [])
            if case.get("target_prompt_hash")
            == policy["target"]["prompt_hash"]
        }
        fork_copy_rows = [
            row for row in copies if str(row.get("case_id")) in target_case_ids
        ]
        fork_fallback_rows = [
            row
            for row in fallbacks
            if str(row.get("case_id")) in target_case_ids
        ]
        if not fork_copy_rows:
            raise AssertionError(f"no physical fork copy: {instance_id}")
        task_rows.append(
            {
                "instance_id": instance_id,
                "fork_request_index": FORK_TASKS[instance_id],
                "target_prompt_hash_equal": prompt_equal,
                "frozen_prefix_hash_equal": prefix_equal,
                "dense_resolved": instance_id in dense_ids,
                "policy_resolved": instance_id in policy_ids,
                "policy_target_islands": policy["target"]["target_islands"],
                "policy_copied_tokens_planned": policy["target"][
                    "copied_tokens_planned"
                ],
                "policy_fork_copy_events": len(fork_copy_rows),
                "policy_fork_fallback_events": len(fork_fallback_rows),
                "policy_fork_copied_k_tokens": sum(
                    int(row.get("copied_k_tokens", 0))
                    for row in fork_copy_rows
                ),
                "target_response_equal": (
                    public_response_signature(dense_target_message)
                    == public_response_signature(policy_target_message)
                ),
                "dense_target_action": public_response_signature(
                    dense_target_message
                )["actions"],
                "policy_target_action": public_response_signature(
                    policy_target_message
                )["actions"],
                "dense_target_ttft_ms": dense_ttft_ms,
                "policy_target_ttft_ms": policy_ttft_ms,
                "target_ttft_saving_percent": (
                    100 * (dense_ttft_ms - policy_ttft_ms) / dense_ttft_ms
                ),
            }
        )

    if not copies:
        raise AssertionError("policy arm did not execute a physical KV copy")
    value = {
        "completed_at_utc": utc_now(),
        "registration": registration,
        "causal_scope": CAUSAL_SCOPE,
        "accuracy": {
            "dense_resolved": len(dense_ids),
            "policy_resolved": len(policy_ids),
            "tasks": len(FORK_TASKS),
            "rescues": sorted(policy_ids - dense_ids),
            "damages": sorted(dense_ids - policy_ids),
            "both_resolved": sorted(policy_ids & dense_ids),
            "both_unresolved": sorted(
                set(FORK_TASKS) - policy_ids - dense_ids
            ),
        },
        "fork_identity": {
            "all_target_prompt_hashes_equal": all(
                row["target_prompt_hash_equal"] for row in task_rows
            ),
            "all_frozen_prefix_hashes_equal": all(
                row["frozen_prefix_hash_equal"] for row in task_rows
            ),
            "all_prefix_workspaces_clean": all(
                not task["workspace"]["git_status"]
                and not task["workspace"]["git_diff"]
                for arm in arms.values() for task in arm["tasks"]
            ),
            "all_prefix_observations_exact": all(
                task["workspace"]["all_observations_exact"]
                for arm in arms.values()
                for task in arm["tasks"]
            ),
        },
        "physical_reuse": {
            "target_copy_events": len(copies),
            "target_fallback_events": len(fallbacks),
            "copied_k_tokens": sum(
                int(row.get("copied_k_tokens", 0)) for row in copies
            ),
            "rotated_k_tokens": sum(
                int(row.get("rotated_k_tokens", 0)) for row in copies
            ),
            "fork_target_copy_events": sum(
                row["policy_fork_copy_events"] for row in task_rows
            ),
            "fork_target_fallback_events": sum(
                row["policy_fork_fallback_events"] for row in task_rows
            ),
            "fork_target_copied_k_tokens": sum(
                row["policy_fork_copied_k_tokens"] for row in task_rows
            ),
        },
        "fork_target_latency": {
            "scope": (
                "same target prompt, cache-ready TTFT; frozen-prefix replay "
                "and source build are experimental setup and excluded"
            ),
            "median_ttft_saving_percent": statistics.median(
                row["target_ttft_saving_percent"] for row in task_rows
            ),
            "median_paired_speedup": statistics.median(
                row["dense_target_ttft_ms"] / row["policy_target_ttft_ms"]
                for row in task_rows
            ),
            "policy_wins": sum(
                row["policy_target_ttft_ms"] < row["dense_target_ttft_ms"]
                for row in task_rows
            ),
            "pairs": len(task_rows),
        },
        "per_task": task_rows,
        "official": official,
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int)
    evaluate = sub.add_parser("evaluate-arm")
    evaluate.add_argument("--arm", choices=ARMS, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()

    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "run-arm":
        value = run_arm(output, args.arm, args.port or ARM_PORTS[args.arm])
    elif args.command == "evaluate-arm":
        value = evaluate_arm(output, args.arm)
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
