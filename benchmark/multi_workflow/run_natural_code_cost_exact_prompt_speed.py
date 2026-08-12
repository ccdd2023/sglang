#!/usr/bin/env python3
"""Replay every treated fresh9 prompt for exact-token Dense/reuse TTFT.

The online agent campaign establishes official task accuracy but its arms take
different trajectories.  This post-accuracy experiment reconstructs the exact
prompt token IDs recorded on the policy trajectory, keeps every physically
treated target group, and compares Dense with the same K/V-copy plan.  Source
prompts are replayed only to materialize the registered snapshots and their
cost is reported separately; they are not counted as agent prefetch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jinja2 import StrictUndefined, Template
import requests
from tokenizers import Tokenizer

from benchmark.multi_workflow.bridge_reuse_litellm_model import (
    BridgeReuseLitellmModel,
    token_ids_hash,
)
from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    CHAT_TEMPLATE,
    MODEL,
    TOKENIZER_JSON,
    launch_server,
    stop_server,
)
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = RuntimePaths.from_project(PROJECT).artifacts
CAMPAIGN = ARTIFACTS / "impactkv_natural_code_cost_agent_20260808"
POLICY_RUN = CAMPAIGN / "online/coding_natural_code_cost/full_9"
DEFAULT_OUTPUT = CAMPAIGN / "exact_prompt_speed"
ARM = "coding_natural_code_cost"
WARMUPS = 1
MEASURED_ROUNDS = 3
TOTAL_ROUNDS = WARMUPS + MEASURED_ROUNDS


def generate_detailed(
    *, base_url: str, input_ids: list[int], key: str
) -> dict[str, Any]:
    """Issue one streamed 1-token request and measure client-observed TTFT."""
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "extra_key": key,
            "input_ids": input_ids,
            "return_logprob": False,
            "sampling_params": {
                "ignore_eos": False,
                "max_new_tokens": 1,
                "temperature": 0,
            },
            "stream": True,
        },
        stream=True,
        timeout=900,
    )
    response.raise_for_status()
    value = None
    ttft_ms = math.inf
    for chunk in response.iter_lines(decode_unicode=True):
        if not chunk or not chunk.startswith("data:"):
            continue
        payload = chunk[5:].strip()
        if payload == "[DONE]":
            break
        value = json.loads(payload)
        if "error" in value:
            raise RuntimeError(value["error"])
        completion_tokens = int(
            value.get("meta_info", {}).get("completion_tokens", 0)
        )
        if math.isinf(ttft_ms) and completion_tokens:
            ttft_ms = 1000 * (time.perf_counter() - started)
    if value is None or math.isinf(ttft_ms):
        raise RuntimeError("empty generation stream")
    meta = value.get("meta_info", {})
    return {
        "cached_tokens": int(meta.get("cached_tokens", 0)),
        "completion_tokens": int(meta.get("completion_tokens", 0)),
        "elapsed_ms": 1000 * (time.perf_counter() - started),
        "finish_reason": meta.get("finish_reason"),
        "output_text": str(value.get("text") or ""),
        "ttft_ms": ttft_ms,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _renderer() -> BridgeReuseLitellmModel:
    model = object.__new__(BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm=ARM,
        rolling_history_groups=6,
        prompt_token_limit=28_000,
        max_tool_observation_chars=6_000,
        max_assistant_reasoning_chars=3_000,
        emergency_message_chars=1_500,
    )
    model._tokenizer = Tokenizer.from_file(TOKENIZER_JSON)
    model._chat_template = Template(
        CHAT_TEMPLATE.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    return model


def request_prompt_cutoffs(messages: list[dict[str, Any]]) -> list[int]:
    """Locate every model request represented by a saved trajectory.

    A successful response is stored as an assistant message, so its request
    prompt is ``messages[:index]``.  mini-swe-agent stores an unparseable
    response only as the following user-side FormatError.  Its failed request
    therefore has the same cutoff, immediately before that interrupt.  The
    latter matters when a reuse source remains live across a format loop.
    """

    return [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        or (
            message.get("role") == "user"
            and (message.get("extra") or {}).get("interrupt_type")
            == "FormatError"
        )
    ]


def reconstruct_prompt_index() -> dict[str, list[int]]:
    model = _renderer()
    prompts: dict[str, list[int]] = {}
    for trajectory_path in sorted(POLICY_RUN.rglob("*.traj.json")):
        messages = read_json(trajectory_path)["messages"]
        for index in request_prompt_cutoffs(messages):
            rolling, _, _ = model._rolling_messages(messages[:index])
            compacted, _ = model.compact_messages(rolling)
            ids = model._render_prompt_ids(compacted)
            digest = token_ids_hash(ids)
            if digest in prompts and prompts[digest] != ids:
                raise ValueError("prompt hash collision")
            prompts[digest] = ids
    return prompts


def build_plan() -> dict[str, Any]:
    prompt_index = reconstruct_prompt_index()
    manifest = read_json(POLICY_RUN / "DYNAMIC_MANIFEST.json")
    sources = {str(row["source_id"]): row for row in manifest["sources"]}
    groups: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for case in manifest["cases"]:
        group_id = str(case["target_group_id"])
        if group_id not in groups:
            group_order.append(group_id)
            groups[group_id] = []
        groups[group_id].append(case)

    plan = []
    for index, group_id in enumerate(group_order):
        cases = sorted(groups[group_id], key=lambda row: int(row["target_start"]))
        target_hashes = {str(row["target_prompt_hash"]) for row in cases}
        if len(target_hashes) != 1:
            raise ValueError(f"{group_id}: target prompt hashes differ")
        target_hash = next(iter(target_hashes))
        if target_hash not in prompt_index:
            raise ValueError(f"{group_id}: target prompt was not reconstructed")
        source_ids = list(dict.fromkeys(str(row["source_id"]) for row in cases))
        source_rows = [copy.deepcopy(sources[value]) for value in source_ids]
        source_hashes = list(
            dict.fromkeys(str(row["source_prompt_hash"]) for row in source_rows)
        )
        missing = [value for value in source_hashes if value not in prompt_index]
        if missing:
            raise ValueError(f"{group_id}: missing source prompts {missing}")
        if target_hash in source_hashes:
            raise ValueError(f"{group_id}: static source/target role collision")
        replay_cases = []
        for island_index, row in enumerate(cases):
            replay_cases.append(
                {
                    **copy.deepcopy(row),
                    "case_id": f"replay-g{index:03d}-i{island_index}",
                    "target_group_id": f"replay-g{index:03d}",
                    "target_uses": TOTAL_ROUNDS,
                }
            )
        plan.append(
            {
                "group_index": index,
                "original_target_group_id": group_id,
                "target_prompt_hash": target_hash,
                "target_input_ids": prompt_index[target_hash],
                "source_prompt_hashes": source_hashes,
                "source_input_ids": [prompt_index[value] for value in source_hashes],
                "sources": source_rows,
                "cases": replay_cases,
                "islands": len(replay_cases),
                "copied_tokens": sum(int(row["length"]) for row in replay_cases),
            }
        )
    return {"groups": plan}


def prepare(output: Path) -> dict[str, Any]:
    registration_path = output / "REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    plan = build_plan()
    output.mkdir(parents=True)
    plan_path = output / "PLAN.json"
    write_json(plan_path, plan)
    registration = {
        "status": "REGISTERED_BEFORE_EXACT_PROMPT_SPEED_GPU",
        "classification": "post-accuracy exact-token cache-ready speed validation",
        "selection": (
            "all online fresh9 target groups with a physical registered copy; "
            "no accuracy, output, or TTFT used for selection"
        ),
        "capacity": {
            "target_groups": len(plan["groups"]),
            "islands": sum(row["islands"] for row in plan["groups"]),
            "copied_tokens_per_round": sum(
                row["copied_tokens"] for row in plan["groups"]
            ),
        },
        "protocol": {
            "model": MODEL,
            "arms": ["dense", ARM],
            "decode_tokens": 1,
            "warmups": WARMUPS,
            "measured_rounds": MEASURED_ROUNDS,
            "exact_target_prompt_tokens": True,
            "ordinary_radix_prefix_reuse": False,
            "source_build_reported_separately": True,
            "synthetic_source_replay_for_measurement": True,
            "agent_prefetch": False,
        },
        "metrics": {
            "primary": "paired cache-ready target TTFT",
            "secondary": "N=4 including one source materialization per target group",
            "accuracy": "not measured here; use fresh9 official agent result",
        },
        "inputs": {
            "online_manifest": str(POLICY_RUN / "DYNAMIC_MANIFEST.json"),
            "online_manifest_sha256": sha256(
                POLICY_RUN / "DYNAMIC_MANIFEST.json"
            ),
            "plan_sha256": sha256(plan_path),
            "trajectory_sha256": {
                str(path.relative_to(POLICY_RUN)): sha256(path)
                for path in sorted(POLICY_RUN.rglob("*.traj.json"))
            },
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
    }
    write_json(registration_path, registration)
    return registration


def _manifest(output: Path, group: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 3,
        "model_id": MODEL,
        "cache_dtype": "bfloat16",
        "lease_ttl_s": 900,
        "ledger_path": str(output / "server/reuse/SERVER_LEDGER.jsonl"),
        "rope": {"rotary_dim": 128, "base": 10_000_000, "is_neox_style": True},
        "sources": group["sources"],
        "cases": group["cases"],
        "release_source_ids": [],
        "arm": ARM,
        "host_overflow_enabled": True,
        "ordinary_prefix_reuse_enabled": False,
        "ordinary_prefix_repair_tokens": 0,
        "ordinary_prefix_target_only": False,
    }


def _empty_manifest(output: Path) -> dict[str, Any]:
    """Create the version-3 sidecar before dynamically appending any group."""
    value = _manifest(output, {"sources": [], "cases": []})
    return value


def _atomic_manifest(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    if arm not in {"dense", "reuse"}:
        raise ValueError(arm)
    prepare(output)
    result_path = output / f"{arm}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    plan = read_json(output / "PLAN.json")["groups"]
    run_dir = output / f"server/{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "DYNAMIC_MANIFEST.json"
    # A version-3 controller starts empty.  Loading a populated manifest at
    # construction time would derive sources from cases and also read the
    # explicit persistent source rows, duplicating their source IDs.  Online
    # operation uses this same empty-then-atomic-append lifecycle.
    write_json(manifest_path, _empty_manifest(output))
    server_arm = ARM if arm == "reuse" else "dense"
    process, stream = launch_server(
        run_dir=run_dir,
        arm=server_arm,
        manifest=manifest_path,
        port=port,
        mem_fraction_static=0.90,
    )
    base_url = f"http://127.0.0.1:{port}"
    sources = []
    targets = []
    try:
        generate_detailed(base_url=base_url, input_ids=[100] * 128, key=f"warm-{arm}")
        for group in plan:
            if arm == "reuse":
                _atomic_manifest(manifest_path, _manifest(output, group))
                for source_index, source_ids in enumerate(group["source_input_ids"]):
                    row = generate_detailed(
                        base_url=base_url,
                        input_ids=source_ids,
                        key=f"source-g{group['group_index']}-{source_index}",
                    )
                    sources.append(
                        {**row, "group_index": group["group_index"], "source_index": source_index}
                    )
            for round_index in range(TOTAL_ROUNDS):
                row = generate_detailed(
                    base_url=base_url,
                    input_ids=group["target_input_ids"],
                    key=f"target-{arm}-g{group['group_index']}-r{round_index}",
                )
                targets.append(
                    {
                        **row,
                        "group_index": group["group_index"],
                        "round_index": max(0, round_index - WARMUPS),
                        "warmup": round_index < WARMUPS,
                        "target_prompt_hash": group["target_prompt_hash"],
                    }
                )
    finally:
        stop_server(process, stream)
    ledger = read_jsonl(run_dir / "SERVER_LEDGER.jsonl")
    value = {"arm": arm, "sources": sources, "targets": targets, "ledger_rows": ledger}
    write_json(result_path, value)
    return {
        "arm": arm,
        "sources": len(sources),
        "targets": len(targets),
        "copy_events": sum(row.get("event") == "target_copied" for row in ledger),
        "fallback_events": sum(row.get("event") == "target_fallback" for row in ledger),
    }


def summarize(output: Path) -> dict[str, Any]:
    plan = read_json(output / "PLAN.json")["groups"]
    dense = read_json(output / "dense.json")
    reuse = read_json(output / "reuse.json")
    dense_rows = {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in dense["targets"] if not row["warmup"]
    }
    reuse_rows = {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in reuse["targets"] if not row["warmup"]
    }
    if set(dense_rows) != set(reuse_rows):
        raise ValueError("paired targets differ")
    savings = [
        1 - float(reuse_rows[key]["ttft_ms"]) / float(dense_rows[key]["ttft_ms"])
        for key in dense_rows
    ]
    group_savings = []
    source_build = {
        index: sum(
            float(row["elapsed_ms"])
            for row in reuse["sources"]
            if int(row["group_index"]) == index
        )
        for index in range(len(plan))
    }
    n4_dense = 0.0
    n4_reuse = 0.0
    for group in plan:
        index = int(group["group_index"])
        dense_mean = statistics.fmean(
            float(row["ttft_ms"])
            for key, row in dense_rows.items() if key[0] == index
        )
        reuse_mean = statistics.fmean(
            float(row["ttft_ms"])
            for key, row in reuse_rows.items() if key[0] == index
        )
        saving = 1 - reuse_mean / dense_mean
        group_savings.append(saving)
        n4_dense += dense_mean * 4
        n4_reuse += reuse_mean * 4 + source_build[index]
    ledger = reuse["ledger_rows"]
    result = {
        "status": "COMPLETE",
        "classification": "exact-target-prompt cache-ready speed validation",
        "coverage": {
            "target_groups": len(plan),
            "islands": sum(row["islands"] for row in plan),
            "measured_pairs": len(savings),
        },
        "latency": {
            "cache_ready_speedup_ratio_of_means": (
                statistics.fmean(float(row["ttft_ms"]) for row in dense_rows.values())
                / statistics.fmean(float(row["ttft_ms"]) for row in reuse_rows.values())
            ),
            "paired_ttft_saving_median": statistics.median(savings),
            "paired_ttft_win_rate": sum(value > 0 for value in savings) / len(savings),
            "target_group_saving_median": statistics.median(group_savings),
            "target_group_win_rate": sum(value > 0 for value in group_savings) / len(group_savings),
            "n4_including_one_source_build_speedup": n4_dense / n4_reuse,
            "mean_source_build_ms_per_target_group": statistics.fmean(source_build.values()),
        },
        "mechanism": {
            "copy_events": sum(row.get("event") == "target_copied" for row in ledger),
            "expected_copy_events": sum(row["islands"] for row in plan) * TOTAL_ROUNDS,
            "fallback_events": sum(row.get("event") == "target_fallback" for row in ledger),
        },
        "one_token_output_agreement": {
            "fraction": sum(
                dense_rows[key].get("output_text") == reuse_rows[key].get("output_text")
                for key in dense_rows
            ) / len(dense_rows),
            "not_accuracy": True,
        },
    }
    write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=("dense", "reuse"), required=True)
    run.add_argument("--port", type=int, default=30000)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "run-arm":
        value = run_arm(output, args.arm, args.port)
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
