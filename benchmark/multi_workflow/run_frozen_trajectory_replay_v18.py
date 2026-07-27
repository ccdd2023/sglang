#!/usr/bin/env python3
"""Replay frozen agent prompts for paired Dense/General/V17 measurements.

The generated diagnostic token is never appended to the next request.  Every
request is rebuilt from the already-frozen Dense trajectory, so all arms see
the same prompt token IDs even when their diagnostic next token differs.
This experiment measures native copy validity, cache-ready TTFT, amortized
build cost, and same-prompt first-token fidelity.  It is deliberately not a
task-accuracy experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests
from jinja2 import StrictUndefined, Template
from tokenizers import Tokenizer

from benchmark.multi_workflow.bridge_reuse_litellm_model import (
    BridgeReuseLitellmModel,
    token_ids_hash,
)
from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    CHAT_TEMPLATE,
    MODEL,
    init_manifest,
    launch_server,
    load_jsonl,
    stop_server,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v18_frozen_replay_20260727"
TRAJECTORY_ROOT = (
    ARTIFACTS
    / "swebench_verified_bridge_v1_20260724"
    / "agent_dense_contextbound_v1"
    / "full_18"
)
INSTANCE_IDS = (
    "astropy__astropy-14995",
    "psf__requests-1142",
    "sphinx-doc__sphinx-9230",
)
ARMS = ("dense", "general", "coding_version_graph_v17")
PORTS = {
    "dense": 32100,
    "general": 32101,
    "coding_version_graph_v17": 32102,
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_path(instance_id: str) -> Path:
    return TRAJECTORY_ROOT / instance_id / f"{instance_id}.traj.json"


def assistant_request_prefixes(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    return [
        messages[:index]
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]


def make_planner(
    *,
    arm: str,
    manifest_path: Path | None,
    client_ledger_path: Path | None,
    instance_nonce: str,
) -> BridgeReuseLitellmModel:
    """Construct only the pure prompt/planning portion of the model wrapper."""

    planner = object.__new__(BridgeReuseLitellmModel)
    planner.config = SimpleNamespace(
        reuse_arm=arm,
        rolling_history_groups=6,
        reuse_copy_cap=4096,
        reuse_min_tokens=128,
        reuse_manifest_path=manifest_path,
        reuse_client_ledger_path=client_ledger_path,
        prompt_token_limit=28_000,
        max_tool_observation_chars=6_000,
        max_assistant_reasoning_chars=3_000,
        emergency_message_chars=1_500,
    )
    planner._tokenizer = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    planner._chat_template = Template(
        CHAT_TEMPLATE.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    planner._instance_nonce = instance_nonce
    planner._request_index = 0
    planner._session_index = 1
    planner._pending_source = None
    return planner


def reset_planner_session(
    planner: BridgeReuseLitellmModel,
    *,
    instance_id: str,
) -> None:
    if planner._pending_source is not None:
        planner._atomic_sidecar_update(
            release_source_ids=[str(planner._pending_source["source_id"])]
        )
    planner._pending_source = None
    planner._request_index = 0
    planner._session_index += 1
    planner._instance_nonce = (
        f"frozen-{planner.config.reuse_arm}-{instance_id}"
    )


def plan_request(
    planner: BridgeReuseLitellmModel,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    planner._request_index += 1
    rolling_messages, selected_groups, rolling = planner._rolling_messages(
        messages
    )
    compacted_messages, compaction = planner.compact_messages(rolling_messages)
    prompt_ids = planner._render_prompt_ids(compacted_messages)
    target = None
    source = None
    next_pending = None
    decision: dict[str, Any] = {
        "arm": planner.config.reuse_arm,
        "mode": "dense",
    }
    releases: list[str] = []
    if planner.config.reuse_arm != "dense":
        target, releases = planner._target_case(prompt_ids)
        source, next_pending, decision = planner._future_source(
            prompt_ids=prompt_ids,
            selected_groups=selected_groups,
        )
        planner._atomic_sidecar_update(
            sources=[source] if source else [],
            cases=[target] if target else [],
            release_source_ids=releases,
        )
        planner._pending_source = next_pending
    return {
        "prompt_ids": prompt_ids,
        "prompt_hash": token_ids_hash(prompt_ids),
        "prompt_tokens": len(prompt_ids),
        "target": target,
        "source": source,
        "decision": decision,
        "rolling": rolling,
        "compaction": compaction,
    }


def simulate_arm(arm: str) -> list[dict[str, Any]]:
    planner = make_planner(
        arm=arm,
        manifest_path=None,
        client_ledger_path=None,
        instance_nonce=f"registration-{arm}",
    )
    rows = []
    for instance_id in INSTANCE_IDS:
        reset_planner_session(planner, instance_id=instance_id)
        messages = read_json(trajectory_path(instance_id))["messages"]
        for request_index, prefix in enumerate(
            assistant_request_prefixes(messages), start=1
        ):
            planned = plan_request(planner, prefix)
            rows.append(
                {
                    "instance_id": instance_id,
                    "request_index": request_index,
                    "prompt_hash": planned["prompt_hash"],
                    "prompt_tokens": planned["prompt_tokens"],
                    "target_registered": planned["target"] is not None,
                    "source_registered": planned["source"] is not None,
                    "copied_tokens_planned": (
                        int(planned["target"]["length"])
                        if planned["target"]
                        else 0
                    ),
                    "source_tokens_planned": (
                        int(planned["source"]["length"])
                        if planned["source"]
                        else 0
                    ),
                    "decision": planned["decision"],
                }
            )
    return rows


def prepare(
    output: Path,
    *,
    arm_order: tuple[str, ...] = ARMS,
    replication_of: str | None = None,
) -> dict[str, Any]:
    path = output / "REPLAY_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    plans = {arm: simulate_arm(arm) for arm in ARMS}
    identities = [
        (row["instance_id"], row["request_index"], row["prompt_hash"])
        for row in plans["dense"]
    ]
    for arm in ARMS[1:]:
        assert identities == [
            (row["instance_id"], row["request_index"], row["prompt_hash"])
            for row in plans[arm]
        ]
    value = {
        "registration_id": output.name,
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_TREATMENT_GPU_RUN",
        "purpose": (
            "Remove free-running agent branch divergence by replaying identical "
            "frozen prompt IDs; measure mechanism, speed, and first-token "
            "fidelity, never task accuracy."
        ),
        "arms": list(ARMS),
        "arm_order": list(arm_order),
        "replication_of": replication_of,
        "instances": list(INSTANCE_IDS),
        "trajectory_sha256": {
            instance_id: sha256(trajectory_path(instance_id))
            for instance_id in INSTANCE_IDS
        },
        "protocol": {
            "model": MODEL,
            "temperature": 0,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_is_never_added_to_future_prompts": True,
            "return_logprob": True,
            "top_logprobs_num": 20,
            "rolling_history_groups": 6,
            "prompt_token_limit": 28_000,
            "copy_cap_tokens": 4_096,
            "prefetch": False,
            "dense_radix_cache": False,
            "reuse_ordinary_radix_prefix": False,
            "n4_including_build_formula": (
                "cache_ready_target_ttft_ms + source_materialize_ms / 4"
            ),
        },
        "gates": {
            "prompt_hashes_identical_across_arms": True,
            "version_graph_target_fallbacks_max": 0,
            "version_graph_physical_copies_min": 1,
            "version_graph_first_token_agreement_with_dense_min": 0.95,
            "report_cache_ready_and_n4_build_inclusive_latency": True,
            "do_not_claim_task_accuracy_from_replay": True,
        },
        "plans": plans,
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    write_json(path, value)
    return value


def generate_one(
    *,
    base_url: str,
    input_ids: list[int],
    key: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "extra_key": key,
            "input_ids": input_ids,
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
            "top_logprobs_num": 20,
            "sampling_params": {
                "ignore_eos": True,
                "max_new_tokens": 1,
                "temperature": 0,
            },
            "stream": True,
        },
        stream=True,
        timeout=900,
    )
    response.raise_for_status()
    last: dict[str, Any] | None = None
    ttft_ms = math.inf
    for chunk in response.iter_lines(decode_unicode=True):
        if not chunk or not chunk.startswith("data:"):
            continue
        payload = chunk[5:].strip()
        if payload == "[DONE]":
            break
        last = json.loads(payload)
        if "error" in last:
            raise RuntimeError(str(last["error"]))
        if (
            math.isinf(ttft_ms)
            and int(last.get("meta_info", {}).get("completion_tokens", 0)) > 0
        ):
            ttft_ms = 1000 * (time.perf_counter() - started)
    if last is None or math.isinf(ttft_ms):
        raise RuntimeError("empty frozen replay response")
    meta = last.get("meta_info", {})
    output_logprobs = meta.get("output_token_logprobs") or []
    output_top = meta.get("output_top_logprobs") or []
    return {
        "ttft_ms": ttft_ms,
        "elapsed_ms": 1000 * (time.perf_counter() - started),
        "output_text": str(last.get("text") or ""),
        "output_token_logprobs": output_logprobs,
        "output_top_logprobs": output_top,
        "completion_tokens": int(meta.get("completion_tokens", 0)),
    }


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    prepare(output)
    run_dir = output / arm
    result_path = run_dir / "REPLAY_RESULTS.json"
    if result_path.exists():
        return read_json(result_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = init_manifest(run_dir, arm)
    planner = make_planner(
        arm=arm,
        manifest_path=manifest if arm != "dense" else None,
        client_ledger_path=run_dir / "PLANNER_LEDGER.jsonl",
        instance_nonce=f"runtime-{arm}",
    )
    process, log = launch_server(
        run_dir=run_dir,
        arm=arm,
        manifest=manifest,
        port=port,
    )
    rows: list[dict[str, Any]] = []
    try:
        base_url = f"http://127.0.0.1:{port}"
        for instance_id in INSTANCE_IDS:
            reset_planner_session(planner, instance_id=instance_id)
            messages = read_json(trajectory_path(instance_id))["messages"]
            for request_index, prefix in enumerate(
                assistant_request_prefixes(messages), start=1
            ):
                planned = plan_request(planner, prefix)
                target = planned["target"]
                key = (
                    str(target["case_id"])
                    if target
                    else f"frozen-{arm}-{instance_id}-q{request_index}"
                )
                generated = generate_one(
                    base_url=base_url,
                    input_ids=planned["prompt_ids"],
                    key=key,
                )
                row = {
                    "arm": arm,
                    "instance_id": instance_id,
                    "request_index": request_index,
                    "request_key": key,
                    "prompt_hash": planned["prompt_hash"],
                    "prompt_tokens": planned["prompt_tokens"],
                    "target_registered": target is not None,
                    "target_source_id": (
                        str(target["source_id"]) if target else None
                    ),
                    "target_length": int(target["length"]) if target else 0,
                    "source_registered": planned["source"] is not None,
                    "source_id": (
                        str(planned["source"]["source_id"])
                        if planned["source"]
                        else None
                    ),
                    "source_length": (
                        int(planned["source"]["length"])
                        if planned["source"]
                        else 0
                    ),
                    "decision": planned["decision"],
                    **generated,
                }
                rows.append(row)
                with (run_dir / "REPLAY_RESULTS.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
        if planner._pending_source is not None and arm != "dense":
            planner._atomic_sidecar_update(
                release_source_ids=[
                    str(planner._pending_source["source_id"])
                ]
            )
            planner._pending_source = None
    finally:
        stop_server(process, log)
    value = {
        "arm": arm,
        "completed_at_utc": utc_now(),
        "requests": len(rows),
        "rows": rows,
    }
    write_json(result_path, value)
    return value


def token_id(row: dict[str, Any]) -> int | None:
    values = row.get("output_token_logprobs") or []
    if not values or len(values[0]) < 2:
        return None
    return int(values[0][1])


def top_distribution(row: dict[str, Any]) -> dict[int, float]:
    values = row.get("output_top_logprobs") or []
    if not values:
        return {}
    distribution: dict[int, float] = {}
    for item in values[0] or []:
        if len(item) >= 2:
            distribution[int(item[1])] = math.exp(float(item[0]))
    return distribution


def coarse_js(a: dict[int, float], b: dict[int, float]) -> float | None:
    """Top-20 Jensen-Shannon divergence with one aggregate residual bucket."""

    if not a or not b:
        return None
    keys = set(a) | set(b)
    pa = [a.get(key, 0.0) for key in keys]
    pb = [b.get(key, 0.0) for key in keys]
    pa.append(max(0.0, 1.0 - sum(pa)))
    pb.append(max(0.0, 1.0 - sum(pb)))
    midpoint = [(left + right) / 2 for left, right in zip(pa, pb)]

    def kl(left: list[float], right: list[float]) -> float:
        return sum(
            p * math.log(p / q)
            for p, q in zip(left, right)
            if p > 0 and q > 0
        )

    return (kl(pa, midpoint) + kl(pb, midpoint)) / 2


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def summarize(output: Path) -> dict[str, Any]:
    results = {
        arm: read_json(output / arm / "REPLAY_RESULTS.json")["rows"]
        for arm in ARMS
    }
    indexed = {
        arm: {
            (row["instance_id"], int(row["request_index"])): row
            for row in rows
        }
        for arm, rows in results.items()
    }
    keys = list(indexed["dense"])
    prompt_identity_ok = all(
        set(indexed[arm]) == set(keys)
        and all(
            indexed[arm][key]["prompt_hash"]
            == indexed["dense"][key]["prompt_hash"]
            for key in keys
        )
        for arm in ARMS
    )
    ledgers = {
        arm: load_jsonl(output / arm / "SERVER_LEDGER.jsonl")
        for arm in ARMS
    }
    summaries: dict[str, Any] = {}
    for arm in ARMS:
        rows = results[arm]
        copies = [
            row for row in ledgers[arm] if row.get("event") == "target_copied"
        ]
        fallbacks = [
            row for row in ledgers[arm] if row.get("event") == "target_fallback"
        ]
        builds = {
            str(row["source_id"]): float(row["materialize_ms"])
            for row in ledgers[arm]
            if row.get("event")
            in ("source_materialized", "source_materialized_host")
        }
        target_rows = [row for row in rows if row["target_registered"]]
        n4 = [
            float(row["ttft_ms"])
            + builds.get(str(row["target_source_id"]), 0.0) / 4
            for row in target_rows
            if str(row["target_source_id"]) in builds
        ]
        summaries[arm] = {
            "requests": len(rows),
            "median_all_ttft_ms": statistics.median(
                float(row["ttft_ms"]) for row in rows
            ),
            "p95_all_ttft_ms": percentile95(
                [float(row["ttft_ms"]) for row in rows]
            ),
            "registered_targets": len(target_rows),
            "physical_copies": len(copies),
            "target_fallbacks": len(fallbacks),
            "copied_tokens": sum(
                int(row.get("copied_k_tokens", 0)) for row in copies
            ),
            "source_builds": len(builds),
            "median_cache_ready_target_ttft_ms": (
                statistics.median(
                    float(row["ttft_ms"]) for row in target_rows
                )
                if target_rows
                else None
            ),
            "median_n4_including_build_ms": (
                statistics.median(n4) if n4 else None
            ),
        }
    fidelity: dict[str, Any] = {}
    for arm in ARMS[1:]:
        agreements: list[bool] = []
        divergences: list[float] = []
        target_agreements: list[bool] = []
        target_divergences: list[float] = []
        nontarget_agreements: list[bool] = []
        for key in keys:
            dense_row = indexed["dense"][key]
            arm_row = indexed[arm][key]
            dense_token = token_id(dense_row)
            arm_token = token_id(arm_row)
            if dense_token is not None and arm_token is not None:
                agreement = dense_token == arm_token
                agreements.append(agreement)
                (
                    target_agreements
                    if arm_row["target_registered"]
                    else nontarget_agreements
                ).append(agreement)
            value = coarse_js(
                top_distribution(dense_row), top_distribution(arm_row)
            )
            if value is not None:
                divergences.append(value)
                if arm_row["target_registered"]:
                    target_divergences.append(value)
        fidelity[arm] = {
            "first_token_comparable_requests": len(agreements),
            "first_token_agreement": (
                sum(agreements) / len(agreements) if agreements else None
            ),
            "mean_top20_plus_residual_js": (
                statistics.fmean(divergences) if divergences else None
            ),
            "target_first_token_agreement": (
                sum(target_agreements) / len(target_agreements)
                if target_agreements
                else None
            ),
            "nontarget_first_token_agreement": (
                sum(nontarget_agreements) / len(nontarget_agreements)
                if nontarget_agreements
                else None
            ),
            "target_mean_top20_plus_residual_js": (
                statistics.fmean(target_divergences)
                if target_divergences
                else None
            ),
            "distribution_metric_scope": (
                "top-20 probabilities plus one aggregate residual bucket; "
                "not full-vocabulary KL"
            ),
        }
    version_copy_keys = {
        (row["instance_id"], int(row["request_index"]))
        for row in results["coding_version_graph_v17"]
        if row["target_registered"]
    }
    paired_speed = {}
    for arm in ARMS:
        values = [
            float(indexed[arm][key]["ttft_ms"])
            for key in version_copy_keys
            if key in indexed[arm]
        ]
        paired_speed[arm] = {
            "requests": len(values),
            "median_ttft_ms": statistics.median(values) if values else None,
            "p95_ttft_ms": percentile95(values),
        }
    coding_active_keys = {
        key
        for key in version_copy_keys
        if int(indexed["coding_version_graph_v17"][key]["target_length"])
        < int(indexed["general"][key]["target_length"])
    }
    coding_active = {}
    for arm in ARMS[1:]:
        agreements = [
            token_id(indexed[arm][key]) == token_id(indexed["dense"][key])
            for key in coding_active_keys
        ]
        divergences = [
            coarse_js(
                top_distribution(indexed["dense"][key]),
                top_distribution(indexed[arm][key]),
            )
            for key in coding_active_keys
        ]
        valid_divergences = [value for value in divergences if value is not None]
        ttfts = [float(indexed[arm][key]["ttft_ms"]) for key in coding_active_keys]
        coding_active[arm] = {
            "requests": len(coding_active_keys),
            "first_token_agreement": (
                sum(agreements) / len(agreements) if agreements else None
            ),
            "mean_top20_plus_residual_js": (
                statistics.fmean(valid_divergences)
                if valid_divergences
                else None
            ),
            "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
            "median_copied_tokens": (
                statistics.median(
                    int(indexed[arm][key]["target_length"])
                    for key in coding_active_keys
                )
                if coding_active_keys
                else None
            ),
        }
    version_fidelity = fidelity["coding_version_graph_v17"][
        "first_token_agreement"
    ]
    version_summary = summaries["coding_version_graph_v17"]
    value = {
        "completed_at_utc": utc_now(),
        "experiment_scope": (
            "mechanism/speed/same-prompt first-token fidelity only; no task "
            "accuracy claim"
        ),
        "prompt_hashes_identical_across_arms": prompt_identity_ok,
        "arm_summaries": summaries,
        "dense_reference_fidelity": fidelity,
        "paired_on_version_graph_target_keys": paired_speed,
        "coding_active_shortened_span_cohort": {
            "definition": (
                "requests where V17's online repository-version rule copied "
                "fewer tokens than General"
            ),
            "arms": coding_active,
        },
        "gate_outcomes": {
            "prompt_hashes_identical_across_arms": prompt_identity_ok,
            "version_graph_target_fallbacks_max_0": (
                version_summary["target_fallbacks"] == 0
            ),
            "version_graph_physical_copies_min_1": (
                version_summary["physical_copies"] >= 1
            ),
            "version_graph_first_token_agreement_min_0_95": (
                version_fidelity is not None and version_fidelity >= 0.95
            ),
            "promoted": (
                prompt_identity_ok
                and version_summary["target_fallbacks"] == 0
                and version_summary["physical_copies"] >= 1
                and version_fidelity is not None
                and version_fidelity >= 0.95
            ),
        },
    }
    write_json(output / "REPLAY_SUMMARY.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "summarize"),
        default="run",
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument(
        "--arm-order",
        default=",".join(ARMS),
        help="comma-separated preregistered execution order",
    )
    parser.add_argument("--replication-of")
    args = parser.parse_args()
    arm_order = tuple(args.arm_order.split(","))
    if sorted(arm_order) != sorted(ARMS) or len(arm_order) != len(ARMS):
        raise ValueError(f"--arm-order must contain each arm once: {ARMS}")
    if args.stage == "prepare":
        print(
            json.dumps(
                prepare(
                    args.output,
                    arm_order=arm_order,
                    replication_of=args.replication_of,
                ),
                indent=2,
            )
        )
        return
    if args.stage == "summarize":
        print(json.dumps(summarize(args.output), indent=2))
        return
    prepare(
        args.output,
        arm_order=arm_order,
        replication_of=args.replication_of,
    )
    arms = (args.arm,) if args.arm else arm_order
    for arm in arms:
        run_arm(args.output, arm, PORTS[arm])
    if not args.arm or all(
        (args.output / arm / "REPLAY_RESULTS.json").exists() for arm in ARMS
    ):
        print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
