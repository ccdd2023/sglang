"""Launcher proof: comma-valued prefetch modes never reach sbatch --export."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.multi_workflow.slurm_export_guard import (
    MODES_EXPORT_KEY,
    build_sbatch_argv,
    export_carries_comma_modes,
    sbatch_export_argv,
    slurm_split_export,
    write_modes_file,
)


def test_historical_114807_export_is_truncated_by_slurm() -> None:
    bad = "--export=ALL,IMPACTKV_PREFETCH_MODES=prefetch_only,combined"
    parts = slurm_split_export(bad)
    assert "IMPACTKV_PREFETCH_MODES=prefetch_only" in parts
    assert "combined" in parts
    assert "IMPACTKV_PREFETCH_MODES=prefetch_only,combined" not in parts
    assert export_carries_comma_modes(bad)


def test_export_builder_rejects_comma_modes() -> None:
    with pytest.raises(ValueError, match="must not be passed via sbatch --export"):
        sbatch_export_argv({MODES_EXPORT_KEY: "prefetch_only,combined"})
    with pytest.raises(ValueError, match="comma"):
        sbatch_export_argv({"OTHER": "a,b"})


def test_builder_never_emits_114807_shape(tmp_path: Path) -> None:
    sbatch = tmp_path / "swebench_template_prefetch.sbatch"
    sbatch.write_text("#SBATCH --job-name=x\n", encoding="utf-8")
    modes_file = tmp_path / "prefetch_modes.txt"
    argv = build_sbatch_argv(
        modes="prefetch_only,combined",
        sbatch=sbatch,
        modes_file=modes_file,
    )
    joined = " ".join(argv)
    assert "IMPACTKV_PREFETCH_MODES=" not in joined
    assert modes_file.read_text(encoding="utf-8").strip() == "prefetch_only,combined"
    export_flags = [item for item in argv if item.startswith("--export=")]
    assert export_flags == ["--export=ALL"]
    for flag in export_flags:
        assert not export_carries_comma_modes(flag)


def test_single_token_combined_still_uses_modes_file(tmp_path: Path) -> None:
    sbatch = tmp_path / "job.sbatch"
    sbatch.write_text("", encoding="utf-8")
    modes_file = tmp_path / "prefetch_modes.txt"
    argv = build_sbatch_argv(
        modes="combined", sbatch=sbatch, modes_file=modes_file
    )
    assert "--export=ALL" in argv
    assert MODES_EXPORT_KEY not in " ".join(argv)
    assert modes_file.read_text(encoding="utf-8").strip() == "combined"


def test_write_modes_file_strips_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "modes.txt"
    write_modes_file(path, " combined , ")
    assert path.read_text(encoding="utf-8") == "combined\n"


def test_submit_shell_hardcodes_export_all() -> None:
    text = (
        Path(__file__).resolve().parent
        / "slurm"
        / "submit_swebench_template_prefetch.sh"
    ).read_text(encoding="utf-8")
    assert 'EXPORT_ARG="--export=ALL"' in text
    assert "sbatch --export=ALL,IMPACTKV" not in text
    assert 'sbatch "$EXPORT_ARG" "$SBATCH"' in text


def test_parallel_submit_shell_hardcodes_export_all_and_per_mode_files() -> None:
    text = (
        Path(__file__).resolve().parent
        / "slurm"
        / "submit_swebench_template_prefetch_parallel.sh"
    ).read_text(encoding="utf-8")
    assert 'EXPORT_ARG="--export=ALL"' in text
    assert "sbatch --export=ALL,IMPACTKV" not in text
    assert "prefetch_mode_${mode}.txt" in text
    assert "prefix_only lossy_only dual combined" in text or "dense prefix_only lossy_only dual combined" in text
    assert "IMPACTKV_PREFETCH_MODES=" not in text or "IMPACTKV_PREFETCH_MODES_FILE" in text
    assert "--export=ALL,IMPACTKV_PREFETCH_MODES=" not in text


def test_sbatch_exclusive_avoids_dual_30b_on_one_node() -> None:
    text = (
        Path(__file__).resolve().parent
        / "slurm"
        / "swebench_template_prefetch.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --exclusive" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "memory.used" in text
    assert "Qwen2.5-Coder-7B-Instruct" in text
    assert "IMPACTKV_MEM_FRACTION_STATIC" in text


def test_sbatch_script_reads_modes_file_not_export_cli() -> None:
    text = (
        Path(__file__).resolve().parent
        / "slurm"
        / "swebench_template_prefetch.sbatch"
    ).read_text(encoding="utf-8")
    assert "prefetch_modes.txt" in text
    assert "Never put IMPACTKV_PREFETCH_MODES=a,b on sbatch --export" in text
    assert "--export=ALL,IMPACTKV_PREFETCH_MODES" not in text


def test_parallel_arm_submits_use_export_all_and_modes_file(tmp_path: Path) -> None:
    sbatch = tmp_path / "swebench_template_prefetch.sbatch"
    sbatch.write_text("#SBATCH --job-name=x\n", encoding="utf-8")
    for mode in ("prefix_only", "lossy_only", "dual", "combined"):
        modes_file = tmp_path / f"prefetch_mode_{mode}.txt"
        argv = build_sbatch_argv(
            modes=mode, sbatch=sbatch, modes_file=modes_file
        )
        assert argv[1] == "--export=ALL", mode
        joined = " ".join(argv)
        assert "IMPACTKV_PREFETCH_MODES=" not in joined, mode
        assert "," not in argv[1] or argv[1] == "--export=ALL"
        assert modes_file.read_text(encoding="utf-8").strip() == mode
        for flag in argv:
            if flag.startswith("--export="):
                assert not export_carries_comma_modes(flag)


def test_parse_modes_is_not_substring_match() -> None:
    from benchmark.multi_workflow.template_prefetch_modes import parse_modes

    assert parse_modes("combined") == ["combined"]
    assert parse_modes("prefix_only,lossy_only") == ["prefix_only", "lossy_only"]
    with pytest.raises(ValueError, match="unknown"):
        parse_modes("prefix_only,combined,extra")
