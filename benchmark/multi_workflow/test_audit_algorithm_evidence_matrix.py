from __future__ import annotations

from benchmark.multi_workflow import audit_algorithm_evidence_matrix as audit


def test_evidence_matrix_is_complete_and_hash_backed() -> None:
    value = audit.build()
    assert value["status"] == "COMPLETE"
    assert len(value["rows"]) >= 14
    assert all(source["sha256"] for source in value["sources"].values())


def test_invalid_m51_is_explicitly_excluded() -> None:
    value = audit.build()
    assert value["protected"]["invalid_m51_excluded"] is True
    assert value["protected"]["invalid_m51_tombstone"]["sha256"]


def test_current_core_claims_are_narrowly_scoped() -> None:
    rows = {row["family"]: row for row in audit.build()["rows"]}
    assert rows["V40 grounded single observation"]["status"] == "RESEARCH_BASELINE"
    assert rows["V46 bounded multi-observation pool"]["status"] == "NOT_PROMOTED"
    assert rows["M52/M53 path dependency"]["status"] == "SUPPORTED_COMPONENT_ONLY"
    assert rows["M54 path-weighted drift"]["status"] == "FALSIFIED_MECHANISM"


def test_all_zero_fresh_accuracy_is_inconclusive() -> None:
    result = {
        "status": "SUPPORTED_V40_RATIONALE",
        "aggregate": {
            "complete_tasks": 13,
            "resolved": {"dense": 0, "general": 0, "v40": 0},
        },
    }
    assert audit._fresh_accuracy_evidence_status(result) == "INCONCLUSIVE_ZERO_POWER"
