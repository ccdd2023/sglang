from benchmark.multi_workflow.coding_reuse_policy import (
    effective_copy_cap,
    is_concrete_source_read,
    is_high_value_executable_failure,
    is_low_value_search_miss,
    is_successful_readonly_evidence,
    latest_group_risk_reasons,
    select_failure_memory_groups,
    select_reuse_groups,
)


def group(command: str, output: str = "<returncode>0</returncode>"):
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": command},
                    }
                }
            ],
        },
        {"role": "tool", "content": output},
    ]


def test_general_keeps_latest_and_legacy_coding_always_protects_it():
    groups = [group(f"echo {index}") for index in range(6)]
    general, general_decision = select_reuse_groups(
        "general", groups, latest_group_messages=groups[-1]
    )
    coding, coding_decision = select_reuse_groups(
        "coding_aware", groups, latest_group_messages=groups[-1]
    )

    assert general == groups[1:]
    assert general_decision["latest_group_protected"] is False
    assert coding == groups[1:-1]
    assert coding_decision["latest_group_protected"] is True


def test_general_8k_is_a_matched_non_coding_budget_ablation():
    groups = [group(f"echo {index}") for index in range(6)]

    eligible, decision = select_reuse_groups(
        "general_8k", groups, latest_group_messages=groups[-1]
    )

    assert eligible == groups[1:]
    assert decision["mode"] == "general_contiguous_8k"
    assert decision["latest_group_protected"] is False
    assert effective_copy_cap("general_8k", 4096, decision) == 8192


def test_general_dual_4k_keeps_the_same_middle_budget_as_general():
    groups = [group(f"echo {index}") for index in range(6)]

    eligible, decision = select_reuse_groups(
        "general_dual_4k", groups, latest_group_messages=groups[-1]
    )

    assert eligible == groups[1:]
    assert decision["mode"] == "general_dual_contiguous_4k"
    assert decision["latest_group_protected"] is False
    assert effective_copy_cap("general_dual_4k", 4096, decision) == 4096


def test_evidence_payoff_v7_widens_only_substantial_successful_read():
    evidence = group(
        "\ncd /testbed && sed -n '1,240p' sklearn/ensemble/voting.py\n",
        "<returncode>0</returncode><output>" + ("source line\n" * 80) + "</output>",
    )
    groups = [group(f"echo {index}") for index in range(5)] + [evidence]

    assert is_successful_readonly_evidence(evidence)
    eligible, decision = select_reuse_groups(
        "coding_evidence_payoff_v7",
        groups,
        latest_group_messages=evidence,
    )

    assert eligible == groups[1:]
    assert decision["mode"] == "evidence_payoff_contiguous"
    assert decision["readonly_evidence"] is True
    assert decision["latest_group_protected"] is False
    decision["candidate_tokens"] = 5120
    assert (
        effective_copy_cap(
            "coding_evidence_payoff_v7", 4096, decision
        )
        == 6144
    )


def test_coding_dual_v8_uses_the_v7_gate_and_wide_budget():
    evidence = group(
        "cd /testbed && sed -n '1,240p' package/module.py",
        "<returncode>0</returncode><output>"
        + ("source line\n" * 80)
        + "</output>",
    )
    groups = [group(f"echo {index}") for index in range(5)] + [evidence]

    eligible, decision = select_reuse_groups(
        "coding_dual_v8", groups, latest_group_messages=evidence
    )
    decision["candidate_tokens"] = 5120

    assert eligible == groups[1:]
    assert decision["mode"] == "evidence_payoff_dual_island"
    assert decision["readonly_evidence"] is True
    assert effective_copy_cap("coding_dual_v8", 4096, decision) == 6144


def test_evidence_payoff_v7_requires_real_marginal_copy_opportunity():
    evidence = group(
        'cd /testbed && rg -n "VotingClassifier" sklearn',
        "<returncode>0</returncode><output>" + ("match\n" * 100) + "</output>",
    )
    groups = [group(f"echo {index}") for index in range(5)] + [evidence]
    _, decision = select_reuse_groups(
        "coding_evidence_payoff_v7",
        groups,
        latest_group_messages=evidence,
    )

    decision["candidate_tokens"] = 5119
    assert (
        effective_copy_cap(
            "coding_evidence_payoff_v7", 4096, decision
        )
        == 4096
    )


def test_evidence_payoff_v7_rejects_failures_mutations_and_tests():
    failed_search = group(
        'rg -n "missing" sklearn',
        "<returncode>1</returncode><output>" + ("miss\n" * 100) + "</output>",
    )
    mutation = group(
        "sed -i 's/old/new/' module.py",
        "<returncode>0</returncode><output>" + ("changed\n" * 100) + "</output>",
    )
    test_run = group(
        "python -m pytest tests/test_module.py",
        "<returncode>0</returncode><output>" + ("passed\n" * 100) + "</output>",
    )

    assert not is_successful_readonly_evidence(failed_search)
    assert not is_successful_readonly_evidence(mutation)
    assert not is_successful_readonly_evidence(test_run)
    groups = [group(f"echo {index}") for index in range(5)] + [failed_search]
    _, decision = select_reuse_groups(
        "coding_evidence_payoff_v7",
        groups,
        latest_group_messages=failed_search,
    )
    decision["candidate_tokens"] = 8192
    assert (
        effective_copy_cap(
            "coding_evidence_payoff_v7", 4096, decision
        )
        == 4096
    )


def test_phase_policy_defaults_to_general_for_read_only_success():
    groups = [group(f"grep symbol file{index}.py") for index in range(6)]

    eligible, decision = select_reuse_groups(
        "coding_phase_v1", groups, latest_group_messages=groups[-1]
    )

    assert eligible == groups[1:]
    assert decision["latest_group_protected"] is False
    assert decision["risk_reasons"] == []


def test_failure_policy_protects_nonzero_or_diagnostic_turn():
    failed = group(
        "python -m pytest test_file.py",
        "<returncode>1</returncode><output>1 failed, 2 passed</output>",
    )
    groups = [group(f"echo {index}") for index in range(5)] + [failed]

    eligible, decision = select_reuse_groups(
        "coding_failure_v1", groups, latest_group_messages=failed
    )

    assert eligible == groups[1:-1]
    assert decision["latest_group_protected"] is True
    assert decision["risk_reasons"] == [
        "nonzero_tool_returncode",
        "failure_diagnostic",
    ]


def test_phase_policy_protects_repository_mutation_and_diff():
    changed = group(
        """python - <<'PY'
from pathlib import Path
path = Path("module.py")
path.write_text(path.read_text().replace("old", "new"))
PY""",
        "<returncode>0</returncode><output>diff --git a/module.py b/module.py</output>",
    )

    assert latest_group_risk_reasons(changed) == [
        "repository_mutation_command",
        "repository_diff_observed",
    ]
    groups = [group(f"echo {index}") for index in range(5)] + [changed]
    eligible, decision = select_reuse_groups(
        "coding_phase_v1", groups, latest_group_messages=changed
    )
    assert eligible == groups[1:-1]
    assert decision["latest_group_protected"] is True


def test_failure_only_variant_does_not_protect_successful_edit():
    changed = group(
        'python -c "from pathlib import Path; Path(\'x.py\').write_text(\'x\')"'
    )
    groups = [group(f"echo {index}") for index in range(5)] + [changed]

    eligible, decision = select_reuse_groups(
        "coding_failure_v1", groups, latest_group_messages=changed
    )

    assert eligible == groups[1:]
    assert decision["latest_group_protected"] is False


def test_adaptive_v2_uses_8k_only_for_safe_phase():
    groups = [group(f"grep symbol file{index}.py") for index in range(6)]
    eligible, safe = select_reuse_groups(
        "coding_adaptive_v2", groups, latest_group_messages=groups[-1]
    )
    assert eligible == groups[1:]
    assert effective_copy_cap("coding_adaptive_v2", 4096, safe) == 8192

    failed = group(
        "python -m pytest test_file.py",
        "<returncode>1</returncode><output>1 failed, 2 passed</output>",
    )
    risky_groups = [*groups[:-1], failed]
    eligible, risky = select_reuse_groups(
        "coding_adaptive_v2",
        risky_groups,
        latest_group_messages=failed,
    )
    assert eligible == risky_groups[1:-1]
    assert effective_copy_cap("coding_adaptive_v2", 4096, risky) == 4096


def test_adaptive_v3_ignores_failed_read_only_search():
    search_miss = group(
        'cd /testbed && grep -n "missing_symbol" module.py',
        "<returncode>1</returncode><output></output>",
    )
    groups = [group(f"echo {index}") for index in range(5)] + [search_miss]

    assert is_low_value_search_miss(search_miss)
    eligible, decision = select_reuse_groups(
        "coding_adaptive_v3",
        groups,
        latest_group_messages=search_miss,
    )

    assert eligible == groups[1:]
    assert decision["latest_group_protected"] is False
    assert decision["risk_reasons"] == []
    assert decision["ignored_risk_reasons"] == ["nonzero_tool_returncode"]
    assert decision["low_value_search_miss"] is True
    assert effective_copy_cap("coding_adaptive_v3", 4096, decision) == 8192


def test_adaptive_v3_keeps_executable_reproduction_failure_dense():
    reproducer = group(
        "python - <<'PY'\nraise AssertionError('reproduced')\nPY",
        (
            "<returncode>1</returncode>"
            "<output>Traceback (most recent call last): "
            "AssertionError: reproduced</output>"
        ),
    )
    groups = [group(f"echo {index}") for index in range(5)] + [reproducer]

    assert not is_low_value_search_miss(reproducer)
    eligible, decision = select_reuse_groups(
        "coding_adaptive_v3",
        groups,
        latest_group_messages=reproducer,
    )

    assert eligible == groups[1:-1]
    assert decision["latest_group_protected"] is True
    assert decision["risk_reasons"] == [
        "nonzero_tool_returncode",
        "failure_diagnostic",
    ]
    assert decision["ignored_risk_reasons"] == []
    assert effective_copy_cap("coding_adaptive_v3", 4096, decision) == 4096


def test_adaptive_v3_does_not_hide_mutating_search_failure():
    failed_edit = group(
        (
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "p = Path('x.py')\n"
            "p.write_text(p.read_text().replace('old', 'new'))\n"
            "PY\n"
            "grep -n new x.py"
        ),
        "<returncode>1</returncode><output></output>",
    )
    groups = [group(f"echo {index}") for index in range(5)] + [failed_edit]

    assert not is_low_value_search_miss(failed_edit)
    eligible, decision = select_reuse_groups(
        "coding_adaptive_v3",
        groups,
        latest_group_messages=failed_edit,
    )

    assert eligible == groups[1:-1]
    assert decision["latest_group_protected"] is True
    assert "repository_mutation_command" in decision["risk_reasons"]


def test_budget_v4_widens_safe_phase_without_dropping_latest_group():
    groups = [group(f"grep symbol file{index}.py") for index in range(6)]

    eligible, decision = select_reuse_groups(
        "coding_budget_v4",
        groups,
        latest_group_messages=groups[-1],
    )

    assert eligible == groups[1:]
    assert decision["mode"] == "adaptive_search_aware_copy_budget"
    assert decision["latest_group_protected"] is False
    assert decision["risk_budget_limited"] is False
    assert effective_copy_cap("coding_budget_v4", 4096, decision) == 8192


def test_budget_v4_uses_general_4k_shape_in_risky_phase():
    failed = group(
        "python -m pytest test_file.py",
        "<returncode>1</returncode><output>1 failed, 2 passed</output>",
    )
    groups = [group(f"echo {index}") for index in range(5)] + [failed]

    eligible, decision = select_reuse_groups(
        "coding_budget_v4",
        groups,
        latest_group_messages=failed,
    )

    assert eligible == groups[1:]
    assert decision["latest_group_protected"] is False
    assert decision["risk_budget_limited"] is True
    assert decision["risk_reasons"] == [
        "nonzero_tool_returncode",
        "failure_diagnostic",
    ]
    assert effective_copy_cap("coding_budget_v4", 4096, decision) == 4096


def test_budget_v4_treats_read_only_search_miss_as_safe():
    search_miss = group(
        'cd /testbed && rg "missing_symbol" module.py',
        "<returncode>1</returncode><output></output>",
    )
    groups = [group(f"echo {index}") for index in range(5)] + [search_miss]

    eligible, decision = select_reuse_groups(
        "coding_budget_v4",
        groups,
        latest_group_messages=search_miss,
    )

    assert eligible == groups[1:]
    assert decision["risk_budget_limited"] is False
    assert decision["low_value_search_miss"] is True
    assert decision["ignored_risk_reasons"] == ["nonzero_tool_returncode"]
    assert effective_copy_cap("coding_budget_v4", 4096, decision) == 8192


def test_source_guard_keeps_source_analysis_and_newer_groups_dense():
    source_read = group(
        "cat /testbed/pkg/module.py",
        "<returncode>0</returncode><output>"
        + "def target(): pass\n" * 30
        + "</output>",
    )
    groups = [
        group("echo oldest"),
        group("grep target pkg/module.py"),
        source_read,
        group("python -c 'print(1)'"),
        group("git status --short"),
        group("echo newest"),
    ]

    assert is_concrete_source_read(source_read)
    eligible, decision = select_reuse_groups(
        "coding_source_guard_v6",
        groups,
        latest_group_messages=groups[-1],
    )

    assert eligible == groups[1:2]
    assert decision["source_guard_active"] is True
    assert decision["source_read_index"] == 1
    assert decision["protected_groups"] == 4
    assert effective_copy_cap(
        "coding_source_guard_v6", 4096, decision
    ) == 4096


def test_source_guard_resets_after_repository_mutation():
    source_read = group(
        "cat /testbed/pkg/module.py",
        "<returncode>0</returncode><output>"
        + "def target(): pass\n" * 30
        + "</output>",
    )
    mutation = group(
        "python -c \"from pathlib import Path; "
        "Path('pkg/module.py').write_text('changed')\""
    )
    groups = [
        group("echo oldest"),
        source_read,
        group("echo inspect"),
        mutation,
        group("git diff"),
        group("echo newest"),
    ]

    eligible, decision = select_reuse_groups(
        "coding_source_guard_v6",
        groups,
        latest_group_messages=groups[-1],
    )

    assert eligible == groups[1:]
    assert decision["source_guard_active"] is False
    assert decision["source_guard_reset_by_mutation"] is True


def test_failure_memory_retains_newest_older_executable_failure():
    first_failure = group(
        "python -m pytest test_old.py",
        "<returncode>1</returncode><output>1 failed, 2 passed</output>",
    )
    newest_failure = group(
        "python -m pytest test_new.py",
        "<returncode>1</returncode><output>2 failed, 1 passed</output>",
    )
    recent = [group(f"echo recent-{index}") for index in range(6)]
    groups = [first_failure, newest_failure, *recent]

    selected, decision = select_failure_memory_groups(
        groups, recent_count=6
    )

    assert is_high_value_executable_failure(first_failure)
    assert is_high_value_executable_failure(newest_failure)
    assert selected == [newest_failure, *recent]
    assert decision["memory_anchor_present"] is True
    assert decision["memory_anchor_source_index"] == 1


def test_failure_memory_ignores_read_only_search_miss():
    search_miss = group(
        'rg "missing_symbol" module.py',
        "<returncode>1</returncode><output></output>",
    )
    recent = [group(f"echo recent-{index}") for index in range(6)]

    selected, decision = select_failure_memory_groups(
        [search_miss, *recent],
        recent_count=6,
    )

    assert not is_high_value_executable_failure(search_miss)
    assert selected == recent
    assert decision["memory_anchor_present"] is False


def test_memory_v5_reuses_guaranteed_recent_five_not_old_anchor():
    anchor = group(
        "python -m pytest test_file.py",
        "<returncode>1</returncode><output>1 failed</output>",
    )
    recent = [group(f"echo recent-{index}") for index in range(6)]

    eligible, decision = select_reuse_groups(
        "coding_memory_v5",
        [anchor, *recent],
        latest_group_messages=recent[-1],
    )

    assert eligible == recent[1:]
    assert decision["memory_anchor_present"] is True
    assert decision["mode"] == "failure_memory_plus_general_8k"
    assert effective_copy_cap("coding_memory_v5", 4096, decision) == 8192
