from benchmark.multi_workflow.cacheblend_coding_matrix import prepare_case
from benchmark.multi_workflow.prepare_fair_sota_comparison_v2 import (
    canary_commands,
    full_static_commands,
    select_length_canaries,
)


def _row(case_id, context):
    return {
        "_id": case_id,
        "answers": ["pass"],
        "context": [context],
        "input": "query",
        "language": "python",
    }


def test_length_canary_selection_is_output_independent():
    workload = {
        "schema_version": 1,
        "dataset": "repobench-p",
        "cases": [
            prepare_case(_row("long", "x" * 300), "repobench-p", 0),
            prepare_case(_row("short", "x"), "repobench-p", 1),
            prepare_case(_row("middle", "x" * 100), "repobench-p", 2),
            prepare_case(_row("middle-2", "x" * 150), "repobench-p", 3),
        ],
    }

    selected = select_length_canaries(workload)

    assert [case["case_id"] for case in selected["cases"]] == [
        "short",
        "middle-2",
        "long",
    ]
    assert {case["split"] for case in selected["cases"]} == {"calibration"}
    assert selected["protocol"]["selection_uses_method_output"] is False


def test_canary_command_plan_keeps_kvcomm_native_only(tmp_path):
    commands = canary_commands(tmp_path, "repobench-p")
    by_id = {command["command_id"]: command for command in commands}

    assert by_id["repobench-p-kvcomm-reuse"]["comparison_layer"] == "native"
    assert (
        by_id["repobench-p-cacheblend-reuse"]["comparison_layer"]
        == "controlled_candidate"
    )
    assert (
        by_id[
            "repobench-p-v40-coding_grounded_observation_island_v40"
        ]["env"]["PYTHONPATH"]
        == ".:python"
    )
    assert all(command["env"].get("CUDA_VISIBLE_DEVICES") == "0" for command in commands)


def test_full_plan_contains_all_frozen_parameter_points(tmp_path):
    commands = full_static_commands(tmp_path, "lcc")
    identifiers = {command["command_id"] for command in commands}

    assert "lcc-cacheblend-recompute-0.25-full" in identifiers
    assert "lcc-cacheblend-recompute-0.75-full" in identifiers
    assert "lcc-kvcomm-threshold-0.3-full" in identifiers
    assert "lcc-kvcomm-threshold-0.7-full" in identifiers
    assert "lcc-v40-cap-2048-prepare-full" in identifiers
    assert (
        "lcc-v40-cap-8192-coding_grounded_observation_island_v40-full"
        in identifiers
    )
    assert len(commands) == 17
