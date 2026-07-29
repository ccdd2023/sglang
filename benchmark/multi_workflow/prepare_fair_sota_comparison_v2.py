#!/usr/bin/env python3
"""Materialize immutable workloads and commands for the fair SOTA comparison."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.cacheblend_coding_matrix import prepare_workload
from benchmark.multi_workflow.fair_sota_comparison_v2 import (
    ARTIFACT_ROOT,
    MODEL_SNAPSHOT,
    STATIC_SOURCE,
    materialize_registration,
    validate_workload,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
REPO = ROOT / "sglang-kvflow-worktrees/fair-comparison-v2"
CACHEBLEND = (
    ROOT / "kvflow-reproductions/worktrees/cacheblend-fair-v2"
)
KVCOMM = ROOT / "kvflow-reproductions/worktrees/kvcomm-fair-v2"
SGLANG_PYTHON = Path("/home/gfy/.conda/envs/sglang-kvflow/bin/python")
CACHEBLEND_PYTHON = Path(
    "/home/gfy/.conda/envs/cacheblend-repro-20260719/bin/python"
)
KVCOMM_PYTHON = Path(
    "/home/gfy/.conda/envs/kvcomm-repro-20260719/bin/python"
)


def _write_once(path: Path, value: Any) -> None:
    serialized = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to replace different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def select_length_canaries(workload: dict[str, Any]) -> dict[str, Any]:
    """Select short/median/long prompts without consulting model output."""

    cases = sorted(
        workload["cases"],
        key=lambda case: (
            sum(len(str(message["content"])) for message in case["messages"]),
            str(case["case_id"]),
        ),
    )
    indices = [0, len(cases) // 2, len(cases) - 1]
    selected = []
    seen: set[str] = set()
    for index in indices:
        case = copy.deepcopy(cases[index])
        case_id = str(case["case_id"])
        if case_id in seen:
            continue
        seen.add(case_id)
        case["split"] = "calibration"
        selected.append(case)
    if len(selected) != 3:
        raise ValueError("static workload needs at least three distinct cases")
    return {
        **{key: value for key, value in workload.items() if key != "cases"},
        "adapter": "fair-sota-v2-length-canary",
        "cases": selected,
        "protocol": {
            **workload.get("protocol", {}),
            "selection": "shortest, median and longest rendered-message text",
            "selection_uses_method_output": False,
        },
    }


def _command(
    *,
    command_id: str,
    layer: str,
    method: str,
    mode: str,
    workdir: Path,
    argv: list[str],
    env: dict[str, str],
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "comparison_layer": layer,
        "method": method,
        "mode": mode,
        "workdir": str(workdir),
        "argv": argv,
        "env": env,
    }


def canary_commands(output: Path, dataset: str) -> list[dict[str, Any]]:
    workload = output / f"static/{dataset}/CANARY_WORKLOAD.json"
    model = str(MODEL_SNAPSHOT)
    cacheblend_runner = CACHEBLEND / "example/repro_common.py"
    kvcomm_runner = KVCOMM / "experiments/repro_common.py"
    v40_runner = REPO / "benchmark/multi_workflow/run_v40_repobench_control.py"
    commands = []

    for mode, ratio in (("dense", 0.5), ("reuse", 0.5)):
        metrics = output / f"canary/{dataset}/cacheblend/{mode}.jsonl"
        commands.append(
            _command(
                command_id=f"{dataset}-cacheblend-{mode}",
                layer="controlled_candidate",
                method="cacheblend",
                mode=mode,
                workdir=CACHEBLEND,
                argv=[
                    str(CACHEBLEND_PYTHON),
                    str(cacheblend_runner),
                    "--workload",
                    str(workload),
                    "--metrics",
                    str(metrics),
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--phase",
                    "canary",
                    "--split",
                    "calibration",
                    "--limit",
                    "0",
                    "--recompute-ratio",
                    str(ratio),
                    "--run-id",
                    f"fair-v2-{dataset}-cacheblend-{mode}",
                ],
                env={
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": str(CACHEBLEND / "vllm_blend"),
                    "KVFLOW_ENGINE_COMMIT": (
                        "a798011319c1bdb59ff6b8a9da06fa5028a3292b"
                    ),
                },
            )
        )

    for mode, threshold in (("dense", 0.5), ("reuse", 0.5)):
        metrics = output / f"canary/{dataset}/kvcomm/{mode}.jsonl"
        commands.append(
            _command(
                command_id=f"{dataset}-kvcomm-{mode}",
                layer="native",
                method="kvcomm",
                mode=mode,
                workdir=KVCOMM,
                argv=[
                    str(KVCOMM_PYTHON),
                    str(kvcomm_runner),
                    "--workload",
                    str(workload),
                    "--metrics",
                    str(metrics),
                    "--output-dir",
                    str(output / f"canary/{dataset}/kvcomm/{mode}-outputs"),
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--phase",
                    "canary",
                    "--split",
                    "calibration",
                    "--limit",
                    "0",
                    "--threshold",
                    str(threshold),
                    "--max-anchor-num",
                    "20",
                    "--window-size",
                    "5",
                    "--run-id",
                    f"fair-v2-{dataset}-kvcomm-{mode}",
                ],
                env={
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTHONNOUSERSITE": "1",
                    "KVFLOW_ENGINE_COMMIT": (
                        "3bf7410ca3fd63930241f9332e0c396c91fc05ed"
                    ),
                },
            )
        )

    v40_output = output / f"canary/{dataset}/v40/cap-4096"
    commands.append(
        _command(
            command_id=f"{dataset}-v40-prepare",
            layer="controlled_candidate",
            method="v40",
            mode="prepare",
            workdir=REPO,
            argv=[
                str(SGLANG_PYTHON),
                str(v40_runner),
                "prepare",
                "--workload",
                str(workload),
                "--output",
                str(v40_output),
                "--copy-cap",
                "4096",
            ],
            env={"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": ".:python"},
        )
    )
    for mode, port in (("dense", 31310), ("coding_grounded_observation_island_v40", 31311)):
        commands.append(
            _command(
                command_id=f"{dataset}-v40-{mode}",
                layer="controlled_candidate",
                method="v40",
                mode=mode,
                workdir=REPO,
                argv=[
                    str(SGLANG_PYTHON),
                    str(v40_runner),
                    "run",
                    "--output",
                    str(v40_output),
                    "--arm",
                    mode,
                    "--port",
                    str(port),
                ],
                env={"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": ".:python"},
            )
        )
    return commands


def full_static_commands(output: Path, dataset: str) -> list[dict[str, Any]]:
    """Build the frozen 200-case parameter sweep for one static dataset."""

    workload = output / f"static/{dataset}/WORKLOAD.json"
    model = str(MODEL_SNAPSHOT)
    commands: list[dict[str, Any]] = []
    cacheblend_runner = CACHEBLEND / "example/repro_common.py"
    kvcomm_runner = KVCOMM / "experiments/repro_common.py"
    v40_runner = (
        REPO / "benchmark/multi_workflow/run_v40_repobench_control.py"
    )
    cacheblend_configs = [("dense", 0.5)] + [
        ("reuse", ratio) for ratio in (0.25, 0.5, 0.75)
    ]
    for mode, ratio in cacheblend_configs:
        config = "dense" if mode == "dense" else f"recompute-{ratio:.2f}"
        commands.append(
            _command(
                command_id=f"{dataset}-cacheblend-{config}-full",
                layer="controlled_candidate",
                method="cacheblend",
                mode=mode,
                workdir=CACHEBLEND,
                argv=[
                    str(CACHEBLEND_PYTHON),
                    str(cacheblend_runner),
                    "--workload",
                    str(workload),
                    "--metrics",
                    str(
                        output
                        / f"static/{dataset}/cacheblend/{config}.jsonl"
                    ),
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--phase",
                    "accuracy",
                    "--split",
                    "formal",
                    "--limit",
                    "0",
                    "--recompute-ratio",
                    str(ratio),
                    "--run-id",
                    f"fair-v2-{dataset}-cacheblend-{config}",
                ],
                env={
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": str(CACHEBLEND / "vllm_blend"),
                    "KVFLOW_ENGINE_COMMIT": (
                        "a798011319c1bdb59ff6b8a9da06fa5028a3292b"
                    ),
                },
            )
        )
    kvcomm_configs = [("dense", 0.5)] + [
        ("reuse", threshold) for threshold in (0.3, 0.5, 0.7)
    ]
    for mode, threshold in kvcomm_configs:
        config = "dense" if mode == "dense" else f"threshold-{threshold:.1f}"
        commands.append(
            _command(
                command_id=f"{dataset}-kvcomm-{config}-full",
                layer="native",
                method="kvcomm",
                mode=mode,
                workdir=KVCOMM,
                argv=[
                    str(KVCOMM_PYTHON),
                    str(kvcomm_runner),
                    "--workload",
                    str(workload),
                    "--metrics",
                    str(output / f"static/{dataset}/kvcomm/{config}.jsonl"),
                    "--output-dir",
                    str(
                        output / f"static/{dataset}/kvcomm/{config}-outputs"
                    ),
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--phase",
                    "accuracy",
                    "--split",
                    "formal",
                    "--limit",
                    "0",
                    "--threshold",
                    str(threshold),
                    "--max-anchor-num",
                    "20",
                    "--window-size",
                    "5",
                    "--run-id",
                    f"fair-v2-{dataset}-kvcomm-{config}",
                ],
                env={
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTHONNOUSERSITE": "1",
                    "KVFLOW_ENGINE_COMMIT": (
                        "3bf7410ca3fd63930241f9332e0c396c91fc05ed"
                    ),
                    "KVFLOW_ADAPTER_COMMIT": (
                        "66f89fb6b5f64e3d7eff2511d8c5922ab641acde"
                    ),
                },
            )
        )
    for cap_index, copy_cap in enumerate((2048, 4096, 8192)):
        config = f"cap-{copy_cap}"
        v40_output = output / f"static/{dataset}/v40/{config}"
        commands.append(
            _command(
                command_id=f"{dataset}-v40-{config}-prepare-full",
                layer="controlled_candidate",
                method="v40",
                mode="prepare",
                workdir=REPO,
                argv=[
                    str(SGLANG_PYTHON),
                    str(v40_runner),
                    "prepare",
                    "--workload",
                    str(workload),
                    "--output",
                    str(v40_output),
                    "--copy-cap",
                    str(copy_cap),
                ],
                env={"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": ".:python"},
            )
        )
        for arm_index, arm in enumerate(
            ("dense", "coding_grounded_observation_island_v40")
        ):
            port = 31400 + cap_index * 10 + arm_index
            commands.append(
                _command(
                    command_id=f"{dataset}-v40-{config}-{arm}-full",
                    layer="controlled_candidate",
                    method="v40",
                    mode=arm,
                    workdir=REPO,
                    argv=[
                        str(SGLANG_PYTHON),
                        str(v40_runner),
                        "run",
                        "--output",
                        str(v40_output),
                        "--arm",
                        arm,
                        "--port",
                        str(port),
                    ],
                    env={
                        "CUDA_VISIBLE_DEVICES": "0",
                        "PYTHONPATH": ".:python",
                    },
                )
            )
    return commands


def prepare(output: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    registration = materialize_registration(output=output)
    workload_summaries = {}
    for dataset in ("repobench-p", "lcc"):
        source = STATIC_SOURCE / f"{dataset}.jsonl"
        workload = prepare_workload(source, dataset, limit=0)
        formal = output / f"static/{dataset}/WORKLOAD.json"
        canary = output / f"static/{dataset}/CANARY_WORKLOAD.json"
        _write_once(formal, workload)
        canary_value = select_length_canaries(workload)
        _write_once(canary, canary_value)
        workload_summaries[dataset] = {
            "formal": validate_workload(workload),
            "canary": validate_workload(canary_value),
        }
    commands = []
    for dataset in ("repobench-p", "lcc"):
        commands.extend(canary_commands(output, dataset))
    plan = {
        "schema_version": 1,
        "registration_id": registration["registration_id"],
        "status": "PREPARED_FOR_PROTOCOL_CANARY",
        "commands": commands,
        "execution_rule": (
            "run one command at a time in listed order; refuse an existing "
            "ledger rather than append; validate all ledgers before expansion"
        ),
        "canary_coverage": {
            "native_lengths": "short, median and long static prompts",
            "version_invalidation": (
                "covered by coding_reuse_policy and coding_aware policy tests"
            ),
            "prefetch": False,
        },
    }
    _write_once(output / "CANARY_COMMAND_PLAN.json", plan)
    static_commands = []
    for dataset in ("repobench-p", "lcc"):
        static_commands.extend(full_static_commands(output, dataset))
    static_plan = {
        "schema_version": 1,
        "registration_id": registration["registration_id"],
        "status": "READY_AFTER_CANARY_PASS",
        "commands": static_commands,
        "execution_rule": (
            "run sequentially on GPU 0; preserve every failed ledger; "
            "validate one method/config before starting the next"
        ),
        "prefetch": False,
    }
    _write_once(output / "STATIC_COMMAND_PLAN.json", static_plan)
    return {
        "registration": str(output / "COMPARISON_REGISTRATION.json"),
        "command_plan": str(output / "CANARY_COMMAND_PLAN.json"),
        "workloads": workload_summaries,
        "commands": len(commands),
        "static_command_plan": str(output / "STATIC_COMMAND_PLAN.json"),
        "static_commands": len(static_commands),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
