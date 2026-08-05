from __future__ import annotations

from benchmark.multi_workflow import run_m56_v40_same_prompt_replay as m56


def test_percentile_is_nearest_rank() -> None:
    assert m56._percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert m56._percentile([4.0, 1.0, 3.0, 2.0], 0.50) == 2.0


def test_replay_arms_are_only_dense_and_v40() -> None:
    assert m56.ARMS == ("dense", "coding_grounded_observation_island_v40")
    assert len(m56.TASKS) == 13


def test_row_key_is_task_and_request() -> None:
    assert m56._key({"instance_id": "task", "request_index": 7}) == ("task", 7)


def test_campaign_completion_uses_coverage_not_scientific_verdict() -> None:
    assert m56._task_campaign_complete(
        {
            "status": "NOT_SUPPORTED_V40_RATIONALE",
            "aggregate": {"complete_tasks": 13},
        }
    )
    assert not m56._task_campaign_complete(
        {
            "status": "SUPPORTED_V40_RATIONALE",
            "aggregate": {"complete_tasks": 12},
        }
    )


def test_recorded_prompt_tokens_are_read_from_assistant_audit() -> None:
    trajectory = {
        "messages": [
            {"role": "user", "content": "issue"},
            {
                "role": "assistant",
                "extra": {"reuse_treatment": {"prompt_tokens": 123}},
            },
            {"role": "tool", "content": "result"},
            {
                "role": "assistant",
                "extra": {"reuse_treatment": {"prompt_tokens": 456}},
            },
        ]
    }
    assert m56._recorded_prompt_tokens(trajectory) == [123, 456]


def test_partial_run_is_preserved_before_single_retry(tmp_path) -> None:
    run = tmp_path / "dense"
    run.mkdir()
    (run / "REPLAY_RESULTS.jsonl").write_text("partial\n")
    failed = m56._preserve_partial_run(run)
    assert failed == tmp_path / "dense.failed_run_1"
    assert not run.exists()
    assert (failed / "REPLAY_RESULTS.jsonl").read_text() == "partial\n"
