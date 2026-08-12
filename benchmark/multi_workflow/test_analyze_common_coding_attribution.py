import json
from pathlib import Path

from benchmark.multi_workflow.analyze_common_coding_attribution import analyze


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in value), encoding="utf-8"
        )
    else:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _official(resolved: list[str], empty: list[str]) -> dict:
    return {
        "report": {
            "submitted_instances": len(resolved) + len(empty),
            "resolved_instances": len(resolved),
            "resolved_ids": resolved,
            "empty_patch_ids": empty,
            "instances": [],
        }
    }


def test_common_attribution_separates_copy_exposed_outcomes(tmp_path: Path) -> None:
    dense = tmp_path / "runs/sglang_formal/dense/full_24"
    coding = (
        tmp_path
        / "runs/sglang_formal/coding_dependency_graph_cold_lcb/full_24"
    )
    _write(dense / "OFFICIAL_RESULT.json", _official([], ["task-a", "task-b"]))
    _write(coding / "OFFICIAL_RESULT.json", _official(["task-a"], ["task-b"]))
    for task, nonce in (("task-a", "p10-m1"), ("task-b", "p10-m2")):
        _write(
            coding / task / f"{task}.traj.json",
            {
                "instance_id": task,
                "messages": [{"role": "assistant", "content": f"call_{nonce}_q1"}],
            },
        )
    _write(
        coding / "CLIENT_LEDGER.jsonl",
        [
            {
                "event": "request_complete",
                "model_instance_nonce": "p10-m1",
                "source_registered": True,
                "target_registered": True,
                "reuse_policy_decision": {
                    "eligible_observations": 1,
                    "dependency_cold_observations": 1,
                },
                "native_backend_metrics": {
                    "physical_reuse": True,
                    "reused_k_tokens": 128,
                    "reused_v_tokens": 128,
                },
            },
            {
                "event": "request_complete",
                "model_instance_nonce": "p10-m2",
                "source_registered": False,
                "target_registered": False,
                "reuse_policy_decision": {
                    "excluded_repository_searches": 1,
                    "source_skip_reasons": {"source_below_minimum_tokens": 1},
                },
                "native_backend_metrics": {"physical_reuse": False},
            },
        ],
    )
    _write(
        coding / "SERVER_LEDGER.jsonl",
        [{"event": "source_materialized"}, {"event": "target_copied"}],
    )
    _write(
        coding / "RUNTIME_SUMMARY.json",
        {"target_copy_events": 1, "copied_tokens": 128},
    )

    result = analyze(tmp_path)

    assert result["status"] == "ACCURACY_COMPLETE_SPEED_PENDING"
    assert result["copy_exposure"]["tasks"] == 1
    assert result["copy_exposure"]["exposed_outcomes"] == {"coding_rescue": 1}
    assert result["copy_exposure"]["unexposed_outcomes"] == {
        "both_unresolved": 1
    }
    assert result["selector_flow"]["source_skip_reasons"] == {
        "source_below_minimum_tokens": 1
    }
