from benchmark.multi_workflow import prepare_dependency_graph_mean_counterfactual as prep


def test_graph_mean_canary_contains_preoutcome_mechanism_task() -> None:
    assert "django__django-15957" in prep.CANARY_IDS
    assert len(prep.CANARY_IDS) == 4
    assert prep.ARM == "coding_dependency_graph_cold_mean"
    assert prep.SOURCE_ARM == "coding_dependency_graph_cold_lcb"
