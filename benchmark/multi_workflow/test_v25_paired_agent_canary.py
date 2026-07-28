from benchmark.multi_workflow import run_v25_paired_agent_canary as canary
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
)


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


def test_paired_accuracy_manifest_prefers_host_sources(tmp_path) -> None:
    path = canary._init_manifest(tmp_path)
    manifest = read_json(path)

    assert manifest["host_overflow_enabled"] is True
    assert manifest["prefer_host_sources"] is True


def test_target_veto_counter_accepts_v33b_and_v34_modes() -> None:
    for mode in (
        "state_transition_target_dense_veto",
        "critical_current_target_dense_veto",
        "version_validation_target_dense_veto",
        "patch_lifecycle_target_dense_veto",
        "commit_phase_target_dense_veto",
    ):
        assert canary._is_target_veto_record(
            {
                "reuse_policy_decision": {
                    "mode": mode,
                    "target_vetoed": True,
                }
            }
        )
    assert not canary._is_target_veto_record(
        {
            "reuse_policy_decision": {
                "mode": "critical_current_target_general_reuse",
                "target_vetoed": False,
            }
        }
    )


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


def test_v33b_branches_before_current_target_request(monkeypatch) -> None:
    candidate = "coding_state_transition_target_v33b"
    monkeypatch.setattr(canary, "V23", candidate)
    monkeypatch.setattr(canary, "REUSE_ARMS", (candidate, canary.GENERAL))
    monkeypatch.setattr(canary, "TARGET_VETO_CANDIDATE", True)
    monkeypatch.setattr(canary, "ABSTENTION_CANDIDATE", False)
    general_target = {
        "case_id": "general-target",
        "source_id": "source-1",
        "length": 1024,
    }
    prepared = {
        candidate: {
            "prompt_ids": [1, 2, 3],
            "source": {"length": 1024},
            "target": None,
            "policy_decision": {
                "mode": "state_transition_target_dense_veto"
            },
        },
        canary.GENERAL: {
            "prompt_ids": [1, 2, 3],
            "source": {"length": 1024},
            "target": general_target,
            "policy_decision": {"mode": "general_contiguous"},
        },
    }

    assert canary._branch_kind(prepared) == "current_target_veto"


def test_v34_branches_on_critical_current_target_veto(monkeypatch) -> None:
    candidate = "coding_critical_current_target_v34"
    monkeypatch.setattr(canary, "V23", candidate)
    monkeypatch.setattr(canary, "REUSE_ARMS", (candidate, canary.GENERAL))
    monkeypatch.setattr(canary, "TARGET_VETO_CANDIDATE", True)
    monkeypatch.setattr(canary, "ABSTENTION_CANDIDATE", False)
    prepared = {
        candidate: {
            "prompt_ids": [4, 5, 6],
            "source": {"length": 2048},
            "target": None,
            "policy_decision": {
                "mode": "critical_current_target_dense_veto"
            },
        },
        canary.GENERAL: {
            "prompt_ids": [4, 5, 6],
            "source": {"length": 2048},
            "target": {
                "case_id": "general-target",
                "source_id": "source-2",
                "length": 2048,
            },
            "policy_decision": {"mode": "general_contiguous"},
        },
    }

    assert canary._branch_kind(prepared) == "current_target_veto"


def test_v35b_branches_on_version_validation_target_veto(
    monkeypatch,
) -> None:
    candidate = "coding_version_validation_target_v35b"
    monkeypatch.setattr(canary, "V23", candidate)
    monkeypatch.setattr(canary, "REUSE_ARMS", (candidate, canary.GENERAL))
    monkeypatch.setattr(canary, "TARGET_VETO_CANDIDATE", True)
    monkeypatch.setattr(canary, "ABSTENTION_CANDIDATE", False)
    prepared = {
        candidate: {
            "prompt_ids": [7, 8, 9],
            "source": {"length": 1536},
            "target": None,
            "policy_decision": {
                "mode": "version_validation_target_dense_veto"
            },
        },
        canary.GENERAL: {
            "prompt_ids": [7, 8, 9],
            "source": {"length": 1536},
            "target": {
                "case_id": "general-target",
                "source_id": "source-3",
                "length": 1536,
            },
            "policy_decision": {"mode": "general_contiguous"},
        },
    }

    assert canary._branch_kind(prepared) == "current_target_veto"


def test_v37_branches_on_patch_lifecycle_target_veto(
    monkeypatch,
) -> None:
    candidate = "coding_patch_lifecycle_target_v37"
    monkeypatch.setattr(canary, "V23", candidate)
    monkeypatch.setattr(canary, "REUSE_ARMS", (candidate, canary.GENERAL))
    monkeypatch.setattr(canary, "TARGET_VETO_CANDIDATE", True)
    monkeypatch.setattr(canary, "ABSTENTION_CANDIDATE", False)
    prepared = {
        candidate: {
            "prompt_ids": [10, 11, 12],
            "source": {"length": 986},
            "target": None,
            "policy_decision": {
                "mode": "patch_lifecycle_target_dense_veto"
            },
        },
        canary.GENERAL: {
            "prompt_ids": [10, 11, 12],
            "source": {"length": 986},
            "target": {
                "case_id": "general-target",
                "source_id": "source-4",
                "length": 986,
            },
            "policy_decision": {"mode": "general_contiguous"},
        },
    }

    assert canary._branch_kind(prepared) == "current_target_veto"


def test_v38_branches_when_commit_phase_latches(
    monkeypatch,
) -> None:
    candidate = "coding_commit_phase_dense_v38"
    monkeypatch.setattr(canary, "V23", candidate)
    monkeypatch.setattr(canary, "REUSE_ARMS", (candidate, canary.GENERAL))
    monkeypatch.setattr(canary, "TARGET_VETO_CANDIDATE", True)
    monkeypatch.setattr(canary, "ABSTENTION_CANDIDATE", False)
    prepared = {
        candidate: {
            "prompt_ids": [13, 14, 15],
            "source": None,
            "target": None,
            "policy_decision": {
                "mode": "commit_phase_target_dense_veto"
            },
        },
        canary.GENERAL: {
            "prompt_ids": [13, 14, 15],
            "source": {"length": 1024},
            "target": {
                "case_id": "general-target",
                "source_id": "source-v38",
                "length": 1024,
            },
            "policy_decision": {"mode": "general_contiguous"},
        },
    }

    assert canary._branch_kind(prepared) == "current_target_veto"


def test_v38_branches_on_commit_phase_future_source_abstention(
    monkeypatch,
) -> None:
    candidate = "coding_commit_phase_dense_v38"
    monkeypatch.setattr(canary, "V23", candidate)
    monkeypatch.setattr(canary, "REUSE_ARMS", (candidate, canary.GENERAL))
    monkeypatch.setattr(canary, "TARGET_VETO_CANDIDATE", True)
    monkeypatch.setattr(canary, "SOURCE_ABSTENTION_CANDIDATE", True)
    monkeypatch.setattr(canary, "ABSTENTION_CANDIDATE", False)
    prepared = {
        candidate: {
            "prompt_ids": [16, 17, 18],
            "source": None,
            "target": None,
            "policy_decision": {"mode": "commit_phase_dense_latched"},
        },
        canary.GENERAL: {
            "prompt_ids": [16, 17, 18],
            "source": {
                "source_id": "general-v38",
                "length": 1200,
            },
            "target": None,
            "policy_decision": {"mode": "general_contiguous"},
        },
    }

    assert canary._branch_kind(prepared) == "future_source_plan"
