from benchmark.multi_workflow import audit_search_file_section_multi_capacity as audit


def test_multi_capacity_audit_is_outcome_blind_by_construction() -> None:
    source = audit.Path(audit.__file__).read_text(encoding="utf-8")
    assert "OFFICIAL_RESULT" not in source
    assert "RESULT.json" in source
    assert audit.ARM == "coding_search_file_section_mean"
