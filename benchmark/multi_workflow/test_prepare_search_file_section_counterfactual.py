from benchmark.multi_workflow import prepare_search_file_section_counterfactual as prep


def test_search_file_section_campaign_is_lossy_and_prefetch_free() -> None:
    assert prep.ARM == "coding_search_file_section_mean"
    assert "search_file_section" in prep.TARGET.name
