from pathlib import Path

from benchmark.multi_workflow.monitor_common_sglang_campaign import (
    archive_retry_artifacts,
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
