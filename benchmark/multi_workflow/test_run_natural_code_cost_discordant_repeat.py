from benchmark.multi_workflow.run_natural_code_cost_discordant_repeat import (
    paired_label,
    stability_summary,
)


def test_paired_label_covers_four_outcomes() -> None:
    dense = {"damage", "both"}
    policy = {"rescue", "both"}
    assert paired_label("rescue", dense, policy) == "rescue"
    assert paired_label("damage", dense, policy) == "damage"
    assert paired_label("both", dense, policy) == "both_resolved"
    assert paired_label("neither", dense, policy) == "both_unresolved"


def test_stability_summary_tracks_transitions_without_relabeling() -> None:
    result = stability_summary(
        original_rescues={"r1", "r2", "r3"},
        original_damages={"d1", "d2"},
        repeat_dense_ids={"r2", "d1", "d2"},
        repeat_policy_ids={"r1", "r3", "d2"},
    )
    assert result["stable_rescues"] == ["r1", "r3"]
    assert result["stable_damages"] == ["d1"]
    assert result["repeat"]["rescues"] == ["r1", "r3"]
    assert result["repeat"]["damages"] == ["d1", "r2"]
    assert result["transition_counts"] == {
        "damage->both_resolved": 1,
        "damage->damage": 1,
        "rescue->damage": 1,
        "rescue->rescue": 2,
    }


def test_stability_summary_preserves_posthoc_warning() -> None:
    result = stability_summary(
        original_rescues={"r"},
        original_damages={"d"},
        repeat_dense_ids=set(),
        repeat_policy_ids=set(),
    )
    assert "not population-level confirmatory" in result["selection_warning"]
    assert result["repeat"]["both_unresolved"] == ["d", "r"]
