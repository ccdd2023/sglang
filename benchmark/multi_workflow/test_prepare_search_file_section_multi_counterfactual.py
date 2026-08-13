from benchmark.multi_workflow import prepare_search_file_section_multi_counterfactual as prep


def test_multi_counterfactual_changes_only_target_island_cap() -> None:
    assert prep.ARM == "coding_search_file_section_multi_mean"
    assert prep.SOURCE_ARM == "coding_search_file_section_mean"
    assert prep.BASELINE_N1_MAX > 1
