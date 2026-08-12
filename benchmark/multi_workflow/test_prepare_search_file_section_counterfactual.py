from benchmark.multi_workflow import prepare_search_file_section_counterfactual as prep


def test_search_file_section_campaign_is_lossy_and_prefetch_free() -> None:
    assert prep.ARM == "coding_search_file_section_mean"
    assert "search_file_section" in prep.TARGET.name


def test_search_monitor_contains_canary_speed_and_accuracy_gate() -> None:
    source = (
        prep.PROJECT
        / "benchmark/multi_workflow/monitor_search_file_section_counterfactual.py"
    ).read_text(encoding="utf-8")
    assert 'state["canary4_gate"]' in source
    assert "if not (speed_pass and accuracy_pass)" in source
