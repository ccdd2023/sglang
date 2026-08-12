from __future__ import annotations

import json

import pytest

from benchmark.multi_workflow import run_common_sglang_exact_prompt_replay as replay
from benchmark.multi_workflow import run_natural_code_cost_exact_prompt_speed as base


def dump(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def arm_value(ttft: float, *, reuse: bool, build: float) -> dict:
    targets = []
    for index in range(replay.TOTAL_ROUNDS):
        targets.append(
            {
                "group_index": 0,
                "round_index": max(0, index - replay.WARMUPS),
                "warmup": index < replay.WARMUPS,
                "ttft_ms": ttft,
            }
        )
    return {
        "targets": targets,
        "sources": (
            [{"group_index": 0, "elapsed_ms": build}] if reuse else []
        ),
        "ledger_rows": (
            [{"event": "target_copied"} for _ in range(replay.TOTAL_ROUNDS)]
            if reuse
            else []
        ),
    }


def test_configure_uses_frozen_canary_cardinality(tmp_path, monkeypatch) -> None:
    campaign = tmp_path / "campaign"
    dump(
        campaign / "CANARY4.json",
        [{"instance_id": f"task-{index}"} for index in range(3)],
    )
    monkeypatch.setattr(replay, "CAMPAIGN", campaign)

    policy_run, output = replay.configure("canary4")

    assert policy_run == (
        campaign / "runs/sglang_canary" / replay.ARM / "full_3"
    )
    assert output == campaign / "exact_prompt_replay/canary4/sglang_coding"


def test_request_prompt_cutoffs_include_failed_format_requests() -> None:
    messages = [
        {"role": "system"},
        {"role": "user"},
        {"role": "assistant"},
        {"role": "tool"},
        {"role": "user", "extra": {"interrupt_type": "FormatError"}},
        {"role": "user", "extra": {"interrupt_type": "FormatError"}},
        {"role": "exit"},
    ]

    assert base.request_prompt_cutoffs(messages) == [2, 4, 5]


def test_summarize_reports_cache_ready_and_amortized_speedups(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "exact"
    output.mkdir()
    dump(
        output / "PLAN.json",
        {
            "groups": [
                {
                    "group_index": 0,
                    "target_prompt_hash": "frozen-hash",
                    "target_input_ids": list(range(100)),
                    "islands": 1,
                    "copied_tokens": 40,
                }
            ]
        },
    )
    for sequence, build in (("ab", 40.0), ("ba", 60.0)):
        dump(output / sequence / "dense.json", arm_value(100.0, reuse=False, build=0))
        dump(output / sequence / "reuse.json", arm_value(50.0, reuse=True, build=build))
    monkeypatch.setattr(replay, "prepare", lambda label, root: {"label": label})

    result = replay.summarize("canary4", output)

    assert result["status"] == "PASS"
    target = result["targets"][0]
    assert target["cache_ready_speedup"] == pytest.approx(2.0)
    assert target["n1_including_build_speedup"] == pytest.approx(1.0)
    assert target["n4_including_build_speedup"] == pytest.approx(1.6)
    assert target["n16_including_build_speedup"] == pytest.approx(100 / 53.125)
    assert result["summary"]["physical_copy_events"] == 14
    assert result["summary"]["fallback_events"] == 0
