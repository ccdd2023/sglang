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
    monkeypatch.setattr(
        canary,
        "ARMS",
        (candidate, canary.GENERAL, canary.DENSE),
    )
    monkeypatch.setattr(canary, "ABSTENTION_CANDIDATE", True)
    target = {
        "case_id": "q9-general",
        "policy_label": canary.GENERAL,
        "reuse_enabled": True,
    }
    prepared = {
        candidate: {"prompt_ids": [1, 2, 3], "target": None},
        canary.GENERAL: {
            "prompt_ids": [1, 2, 3],
            "target": target,
        },
        canary.DENSE: {"prompt_ids": [1, 2, 3], "target": None},
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


def test_diverged_prompts_do_not_consume_another_arms_cases(
    monkeypatch,
) -> None:
    candidate = "coding_critical_event_abstain_v31"
    monkeypatch.setattr(canary, "V23", candidate)
    monkeypatch.setattr(canary, "REUSE_ARMS", (candidate, canary.GENERAL))
    monkeypatch.setattr(
        canary,
        "ARMS",
        (candidate, canary.GENERAL, canary.DENSE),
    )
    target = {
        "case_id": "general-target",
        "policy_label": canary.GENERAL,
        "reuse_enabled": True,
    }
    prepared = {
        candidate: {"prompt_ids": [1], "target": None},
        canary.GENERAL: {"prompt_ids": [2], "target": target},
        canary.DENSE: {"prompt_ids": [3], "target": None},
    }

    cases = canary._paired_cases(
        prepared,
        include_dense_control=True,
    )

    assert cases == [{**target, "ordinary_prefix_reuse": False}]
