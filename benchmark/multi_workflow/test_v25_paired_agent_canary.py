from benchmark.multi_workflow import run_v25_paired_agent_canary as canary


def test_policy_mode_accepts_prepared_and_client_ledger_schema() -> None:
    assert (
        canary._policy_mode(
            {"policy_decision": {"mode": "critical_event_dense_abstain"}}
        )
        == "critical_event_dense_abstain"
    )
    assert (
        canary._policy_mode(
            {
                "reuse_policy_decision": {
                    "mode": "critical_event_dense_abstain"
                }
            }
        )
        == "critical_event_dense_abstain"
    )
    assert canary._policy_mode({}) == ""


def test_abstention_cases_reserve_general_target_for_second_request(
    monkeypatch,
) -> None:
    candidate = "coding_critical_event_abstain_v31"
    monkeypatch.setattr(canary, "V23", candidate)
    monkeypatch.setattr(canary, "REUSE_ARMS", (candidate, canary.GENERAL))
    monkeypatch.setattr(canary, "ABSTENTION_CANDIDATE", True)
    target = {
        "case_id": "q9-general",
        "policy_label": canary.GENERAL,
        "reuse_enabled": True,
    }
    prepared = {
        candidate: {"target": None},
        canary.GENERAL: {"target": target},
    }

    cases = canary._paired_cases(
        prepared,
        include_dense_control=True,
    )

    assert [case["policy_label"] for case in cases] == [
        candidate,
        canary.GENERAL,
        canary.DENSE,
    ]
    assert [case["reuse_enabled"] for case in cases] == [
        False,
        True,
        False,
    ]
    assert len({case["case_id"] for case in cases}) == 3
