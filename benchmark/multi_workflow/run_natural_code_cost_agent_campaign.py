#!/usr/bin/env python3
"""Freeze and run the fresh natural-code cost-policy agent campaign.

The campaign deliberately separates two evidence levels.  Dense and the
natural-code policy can be compared with official SWE-bench accuracy under the
same rolling mini-SWE-agent backend.  Existing CacheBlend and KVCOMM native
bridges accept frozen verifier prompts, not the multi-turn tool history, so
their controlled same-token results remain external references until a true
rolling-history adapter exists; they are never relabeled as agent accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
PATHS = RuntimePaths.from_project(PROJECT)
ARTIFACTS = PATHS.artifacts
NATURAL = ARTIFACTS / "impactkv_natural_module_attention_20260808"
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_natural_code_cost_agent_20260808"
BRIDGE_RUNNER = (
    PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py"
)
MINI_PYTHON = PATHS.mini_python
STAGE = (
    NATURAL
    / "attention_initial20_r1/physical_splice_minimal_reliable/"
    "stage_overhead_code_only_r2"
)
FRESH_IDS = (
    "django__django-13343",
    "sympy__sympy-22914",
    "sphinx-doc__sphinx-7757",
    "pytest-dev__pytest-6202",
    "matplotlib__matplotlib-25287",
    "scikit-learn__scikit-learn-13142",
    "pydata__xarray-7229",
    "astropy__astropy-14309",
    "pylint-dev__pylint-6903",
)
ONLINE_ARMS = ("dense", "coding_natural_code_cost")
DEFAULT_CANARY = "pytest-dev__pytest-6202"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare(output: Path) -> dict[str, Any]:
    registration_path = output / "CAMPAIGN_REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to register over non-empty output: {output}"
        )

    cohort = read_json(NATURAL / "COHORT_REGISTRATION.json")
    initial_ids = {
        row["instance_id"] for row in cohort["selection"]["initial"]
    }
    maximum_ids = [
        row["instance_id"]
        for row in cohort["selection"]["capacity_ceiling"]
    ]
    unused = tuple(value for value in maximum_ids if value not in initial_ids)
    if unused != FRESH_IDS:
        raise RuntimeError(
            "the frozen capacity-ceiling remainder no longer matches FRESH_IDS"
        )

    maximum_dataset = NATURAL / "dataset_max29/test.jsonl"
    rows = [
        json.loads(line)
        for line in maximum_dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["instance_id"]): row for row in rows}
    dataset_rows = [by_id[value] for value in FRESH_IDS]

    maximum_snapshot = read_json(NATURAL / "FROZEN_MAX29.json")
    snapshot_by_id = {
        str(row["instance_id"]): row for row in maximum_snapshot
    }
    snapshot_rows = [snapshot_by_id[value] for value in FRESH_IDS]

    output.mkdir(parents=True)
    dataset_path = output / "dataset/test.jsonl"
    snapshot_path = output / "FROZEN_FRESH9.json"
    bridge_registration_path = output / "BRIDGE_FRESH9_REGISTRATION.json"
    _write_jsonl(dataset_path, dataset_rows)
    write_json(snapshot_path, snapshot_rows)
    write_json(
        bridge_registration_path,
        {
            "registration_id": "impactkv-natural-code-cost-agent-fresh9",
            "schema_version": 1,
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [
                {"instance_id": value} for value in FRESH_IDS
            ],
        },
    )

    policy_source = PROJECT / "benchmark/multi_workflow/coding_reuse_policy.py"
    bridge_source = (
        PROJECT / "benchmark/multi_workflow/bridge_reuse_litellm_model.py"
    )
    runner_source = Path(__file__).resolve()
    value = {
        "status": "REGISTERED_BEFORE_FRESH_AGENT_TREATMENT",
        "registered_at_utc": utc_now(),
        "purpose": (
            "test official task accuracy and online TTFT after replacing the "
            "fixed island-length heuristic with natural repository-code "
            "modules and a measured cache-ready cost gate"
        ),
        "selection": {
            "instance_ids": list(FRESH_IDS),
            "count": len(FRESH_IDS),
            "rule": (
                "all nine preselected capacity-ceiling tasks not used by the "
                "initial20 natural-module trajectories; order frozen before "
                "opening any task outcome in this campaign"
            ),
            "outcome_used_for_selection": False,
            "canary": DEFAULT_CANARY,
        },
        "online_same_mas_protocol": {
            "arms": list(ONLINE_ARMS),
            "backend": "mini-SWE-agent rolling6 + SGLang",
            "model": "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "temperature": 0,
            "step_limit": 32,
            "same_system_and_agent_templates": True,
            "official_accuracy": "SWE-bench resolved",
            "speed": "paired-within-arm request TTFT normalized to Dense",
            "prefetch": False,
        },
        "policy": {
            "eligible_module": (
                "successful, version-valid, single-file direct repository "
                "read tool result"
            ),
            "always_dense": [
                "repository search",
                "assistant interpretation",
                "tool command",
                "test or execution feedback",
                "diff or mutation feedback",
                "ambiguous multi-file output",
            ],
            "pool": (
                "at most three persistent sources, keyed by source prompt, "
                "group identity, path, and segment token identity"
            ),
            "cost_formula": (
                "saving_ms = 0.13169242 * "
                "(island_tokens * target_prompt_tokens / 10000) - 14.66811245"
            ),
            "admission": "strictly positive predicted cache-ready saving",
            "fit": {
                "cases": 24,
                "r2": 0.8750207389619562,
                "classification": (
                    "post-hoc exploratory engineering fit; accuracy-blind"
                ),
            },
        },
        "external_baseline_boundary": {
            "CacheBlend": (
                "native fixed-prompt verifier bridge exists; it does not "
                "currently execute rolling tool-call agent histories"
            ),
            "KVCOMM": (
                "native fixed-prompt verifier bridge exists; it requires a "
                "two-message template and is not official agent accuracy"
            ),
            "reporting_rule": (
                "do not rank their fixed-prompt accuracy against this fresh9 "
                "agent accuracy; compare only after a prompt/token-preserving "
                "rolling-history adapter is implemented"
            ),
        },
        "gates": {
            "canary_terminal_output": True,
            "canary_target_copy_events_min": 1,
            "target_fallback_events": 0,
            "full_official_evaluation_required": True,
            "advantage_required_for_continuation": (
                "positive cache-ready TTFT point estimate with no official "
                "resolved-task loss versus same-campaign Dense"
            ),
        },
        "inputs": {
            "cohort_registration": str(NATURAL / "COHORT_REGISTRATION.json"),
            "cohort_registration_sha256": sha256(
                NATURAL / "COHORT_REGISTRATION.json"
            ),
            "stage_result": str(STAGE / "RESULT.json"),
            "stage_result_sha256": sha256(STAGE / "RESULT.json"),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256(dataset_path),
            "snapshot": str(snapshot_path),
            "snapshot_sha256": sha256(snapshot_path),
            "bridge_registration": str(bridge_registration_path),
            "bridge_registration_sha256": sha256(bridge_registration_path),
            "source_sha256": {
                str(path.relative_to(PROJECT)): sha256(path)
                for path in (policy_source, bridge_source, runner_source)
            },
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
    }
    write_json(registration_path, value)
    return value


def run_bridge(
    output: Path,
    *,
    arm: str,
    scope: str,
    canary: str,
    official: bool,
    port: int,
) -> None:
    if arm not in ONLINE_ARMS:
        raise ValueError(f"unsupported online arm: {arm}")
    prepare(output)
    env = os.environ.copy()
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(output / "dataset"),
            "IMPACTKV_EVAL_SNAPSHOT": str(output / "FROZEN_FRESH9.json"),
            "IMPACTKV_EVAL_REGISTRATION": str(
                output / "BRIDGE_FRESH9_REGISTRATION.json"
            ),
            "IMPACTKV_AGENT_STEP_LIMIT": "32",
            "PYTHONPATH": (
                str(PROJECT)
                + (
                    os.pathsep + env["PYTHONPATH"]
                    if env.get("PYTHONPATH")
                    else ""
                )
            ),
        }
    )
    command = [
        str(MINI_PYTHON),
        str(BRIDGE_RUNNER),
        "--output",
        str(output / "online"),
        "run-arm",
        "--arm",
        arm,
        "--scope",
        scope,
        "--port",
        str(port),
    ]
    if scope == "canary":
        if canary not in FRESH_IDS:
            raise ValueError(f"canary is not in the frozen cohort: {canary}")
        command.extend(["--instance-filter", canary])
    if official:
        command.append("--official")
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=ONLINE_ARMS, required=True)
    run.add_argument("--scope", choices=("canary", "full"), required=True)
    run.add_argument("--canary", default=DEFAULT_CANARY)
    run.add_argument("--official", action="store_true")
    run.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    output = args.output.resolve()
    if args.command == "prepare":
        print(json.dumps(prepare(output), ensure_ascii=False, indent=2))
    else:
        run_bridge(
            output,
            arm=args.arm,
            scope=args.scope,
            canary=args.canary,
            official=args.official,
            port=args.port,
        )


if __name__ == "__main__":
    main()
