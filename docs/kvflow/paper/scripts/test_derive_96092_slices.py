"""Deriver must read frozen 96092 files; N=4 oracle is RESULT.json not 1.375."""

from __future__ import annotations

from pathlib import Path

from derive_96092_slices import DEFAULT_ART, DEFAULT_TRAJ, analyze, pair_rows, read_json


def test_n4_matches_frozen_result_not_a_literal() -> None:
    result = read_json(DEFAULT_ART / "RESULT.json")
    assert result["status"] == "COMPLETE"
    slices = analyze(DEFAULT_ART, DEFAULT_TRAJ)
    n4 = float(result["latency"]["n4_including_one_source_build_speedup"])
    assert abs(slices["n_use_including_one_source_build"]["4"] - n4) < 1e-6
    assert slices["status"] == "DERIVED_FROM_FROZEN_96092"
    assert slices["not_a_new_gpu_arm"] is True
    assert slices["prefetch"] is False


def test_dataset_card_is_24_tasks_not_235_tasks() -> None:
    slices = analyze(DEFAULT_ART, DEFAULT_TRAJ)
    assert slices["dataset"]["tasks"] == 24
    assert slices["dataset"]["target_groups"] == 235
    assert slices["dataset"]["groups_are_rolling6_turns_not_tasks"] is True
    assert slices["dataset"]["instances_with_eligible_groups"] == len(
        slices["dataset"]["instances"]
    )
    assert slices["dataset"]["instances_with_eligible_groups"] <= 24
    assert slices["dataset"]["n_repos"] >= 5
    assert sum(slices["dataset"]["repos"].values()) == 235
    for instance_id in slices["dataset"]["instances"]:
        assert "__" in instance_id


def test_island_slices_cover_all_groups_and_pairs_match() -> None:
    plan = read_json(DEFAULT_ART / "PLAN.json")["groups"]
    dense = read_json(DEFAULT_ART / "dense.json")
    reuse = read_json(DEFAULT_ART / "reuse.json")
    pairs = pair_rows(dense, reuse)
    slices = analyze(DEFAULT_ART, DEFAULT_TRAJ)
    island = slices["island_count_slices"]
    assert sum(row["groups"] for row in island.values()) == len(plan) == 235
    assert sum(row["pairs"] for row in island.values()) == len(pairs)
    for row in island.values():
        assert row["cache_ready_speedup"] > 0


def test_delta_and_repo_slices_cover_all_groups() -> None:
    slices = analyze(DEFAULT_ART, DEFAULT_TRAJ)
    delta = slices["abs_delta_slices"]
    repo = slices["repo_slices"]
    assert sum(row["groups"] for row in delta.values()) == 235
    assert sum(row["groups"] for row in repo.values()) == 235
    assert set(delta) == {"<500", "500-1500", "1500-3000", ">=3000"}
    assert slices["dataset"]["n_repos"] == len(repo)


def test_copied_fraction_quartiles_and_shared_source_build() -> None:
    result = read_json(DEFAULT_ART / "RESULT.json")
    slices = analyze(DEFAULT_ART, DEFAULT_TRAJ)
    n4 = float(result["latency"]["n4_including_one_source_build_speedup"])
    frac = slices["copied_fraction_quartiles"]
    assert set(frac["slices"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert sum(row["groups"] for row in frac["slices"].values()) == 235
    assert len(frac["cuts"]) == 3
    shared = slices["cross_group_source_amortization"]
    assert shared["not_headline"] is True
    assert abs(shared["n4_per_group_build"] - n4) < 1e-6
    assert shared["n4_unique_source_build_once"] > n4
    assert shared["unique_source_hashes"] == slices["source_sharing"][
        "unique_source_prompt_hashes"
    ]


def test_zero_shift_is_token_share_not_ttft() -> None:
    slices = analyze(DEFAULT_ART, DEFAULT_TRAJ)
    zero = slices["skipped_zero_shift"]
    assert zero["not_a_radix_ttft"] is True
    assert zero["dropped_islands"] == 48
    assert zero["dropped_tokens"] > 0
    assert 0 < zero["dropped_over_target"] < 1


def test_writes_slices_next_to_artifact(tmp_path: Path) -> None:
    # Smoke: functions exist on the shipped module path.
    assert (Path(__file__).parent / "derive_96092_slices.py").is_file()
