from pathlib import Path

from benchmark.multi_workflow.monitor_common_sglang_campaign import (
    archive_retry_artifacts,
    coding_copy_gate,
)


def test_archive_retry_artifacts_preserves_relative_layout(tmp_path: Path) -> None:
    canary = tmp_path / "runs/sglang_canary/dense/full_4"
    exact = tmp_path / "exact_prompt_replay/canary4/sglang_coding"
    canary.mkdir(parents=True)
    exact.mkdir(parents=True)
    (canary / "result.json").write_text("{}\n", encoding="utf-8")
    (exact / "RESULT.json").write_text("{}\n", encoding="utf-8")

    archive_value, moved = archive_retry_artifacts(
        tmp_path, {"dense": "100", "coding": "101"}
    )

    assert archive_value is not None
    archive = Path(archive_value)
    assert moved == [
        "runs/sglang_canary",
        "exact_prompt_replay/canary4/sglang_coding",
    ]
    assert (archive / "runs/sglang_canary/dense/full_4/result.json").is_file()
    assert (
        archive / "exact_prompt_replay/canary4/sglang_coding/RESULT.json"
    ).is_file()
    assert not (tmp_path / "runs/sglang_canary").exists()
    assert not (
        tmp_path / "exact_prompt_replay/canary4/sglang_coding"
    ).exists()


def test_coding_copy_gate_distinguishes_zero_capacity_from_copy_failure() -> None:
    zero_capacity = {
        "source_materialized_device_events": 1,
        "target_registered_requests": 0,
        "target_copy_events": 0,
        "target_fallback_events": 0,
    }
    assert coding_copy_gate(zero_capacity, allow_zero_target=True) == (
        "zero_target_opportunity"
    )
    assert coding_copy_gate(zero_capacity, allow_zero_target=False) == (
        "failed_physical_copy_gate"
    )
    assert coding_copy_gate(
        {**zero_capacity, "target_copy_events": 1}, allow_zero_target=False
    ) == "physical_target_copy"
    assert coding_copy_gate(
        {**zero_capacity, "target_registered_requests": 1},
        allow_zero_target=True,
    ) == "failed_physical_copy_gate"
