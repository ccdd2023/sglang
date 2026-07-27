from benchmark.multi_workflow.run_v25_paired_agent_canary import _policy_mode


def test_policy_mode_accepts_prepared_and_client_ledger_schema() -> None:
    assert (
        _policy_mode(
            {"policy_decision": {"mode": "critical_event_dense_abstain"}}
        )
        == "critical_event_dense_abstain"
    )
    assert (
        _policy_mode(
            {
                "reuse_policy_decision": {
                    "mode": "critical_event_dense_abstain"
                }
            }
        )
        == "critical_event_dense_abstain"
    )
    assert _policy_mode({}) == ""
