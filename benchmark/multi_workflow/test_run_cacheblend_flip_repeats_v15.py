from __future__ import annotations

from pathlib import Path

from benchmark.multi_workflow.run_cacheblend_flip_repeats_v15 import (
    build_command,
)


def test_build_command_is_frozen_and_selects_all_flips(
    monkeypatch,
    tmp_path: Path,
) -> None:
    registration = {
        "registered_before_repeat_gpu": True,
        "protocol": {"case_ids": [f"case-{index}" for index in range(20)]},
    }
    output = tmp_path / "audit"
    output.mkdir()
    (output / "V15_REPEAT_REGISTRATION.json").write_text(
        __import__("json").dumps(registration)
    )

    command = build_command(output=output, start=2, arm="reuse")

    assert command[0].endswith("cacheblend-repro-20260719/bin/python")
    assert command[command.index("--mode") + 1] == "reuse"
    assert command[command.index("--recompute-ratio") + 1] == "0.05"
    assert command.count("--case-id") == 20
    assert command[command.index("--limit") + 1] == "0"
