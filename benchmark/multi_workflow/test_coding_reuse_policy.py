from benchmark.multi_workflow.coding_reuse_policy import (
    coding_state_transition_target_reasons,
    coding_version_validation_target_reasons,
    critical_coding_event_reasons,
    effective_copy_cap,
    grounded_observation_candidates,
    is_concrete_source_read,
    is_high_value_executable_failure,
    is_successful_executable_evidence,
    is_successful_focused_validation,
    is_low_value_search_miss,
    is_successful_readonly_evidence,
    latest_group_risk_reasons,
    post_mutation_payoff_guard,
    select_failure_memory_groups,
    select_reuse_groups,
    select_version_graph_groups,
    tool_observation_sha256,
    versioned_evidence_target_guard,
    versioned_grounded_observation_candidates,
    versioned_symbol_observation_candidates,
)


def _command_group(command: str, output: str = "") -> list[dict]:
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
        {
            "role": "tool",
            "content": f"<returncode>0</returncode><output>{output}</output>",
        },
    ]


def test_grounded_observation_excludes_assistant_and_stale_read() -> None:
    stable = _command_group(
        "sed -n '1,200p' /testbed/pkg/stable.py",
        "stable source " * 80,
    )
    stale = _command_group(
        "sed -n '1,200p' /testbed/pkg/stale.py",
        "old source " * 80,
    )
    mutation = _command_group(
        "sed -i 's/old/new/' /testbed/pkg/stale.py"
    )

    candidates, decision = grounded_observation_candidates(
        [stable, stale, mutation]
    )

    assert candidates == [[stable[1]]]
    assert candidates[0][0]["role"] == "tool"
    assert decision["read_only_observations"] == 2
    assert decision["version_invalidated_observations"] == 1
    assert decision["candidate_group_indices"] == [0]
    assert decision["assistant_tokens_selected"] == 0


def test_grounded_observation_rejects_validation_and_diff() -> None:
    validation = _command_group("pytest -q", "1 passed " * 80)
    diff = _command_group("git diff", "diff --git a/a.py b/a.py " * 40)

    candidates, decision = grounded_observation_candidates(
        [validation, diff]
    )

    assert candidates == []
    assert decision["read_only_observations"] == 0


def _python_read(symbol: str = "stable") -> list[dict]:
    return _command_group(
        "sed -n '1,240p' /testbed/pkg/module.py",
        f"def {symbol}():\n    return 1\n" + ("# context\n" * 60),
    )


def _symbol_patch(symbol: str | None) -> list[dict]:
    hunk = f"@@ def {symbol}():" if symbol else "@@"
    return _command_group(
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Update File: pkg/module.py\n"
        f"{hunk}\n"
        "-    return 1\n"
        "+    return 2\n"
        "*** End Patch\n"
        "PATCH"
    )


def test_v45_preserves_only_explicit_same_file_symbol_disjoint_read() -> None:
    source = _python_read("stable")
    disjoint_mutation = _symbol_patch("other")

    v40, _ = grounded_observation_candidates([source, disjoint_mutation])
    v45, decision = versioned_symbol_observation_candidates(
        [source, disjoint_mutation]
    )

    assert v40 == []
    assert v45 == [[source[1]]]
    assert decision["symbol_disjoint_preservations"] == 1
    assert decision["candidate_evidence"] == [
        {
            "group_index": 0,
            "paths": ["pkg/module.py"],
            "symbols": ["stable"],
            "observation_sha256": tool_observation_sha256(source),
            "later_mutation_groups": 1,
            "symbol_disjoint_mutations": 1,
        }
    ]


def test_v45_same_symbol_or_ambiguous_same_file_write_fails_closed() -> None:
    source = _python_read("stable")

    same_symbol, same_decision = versioned_symbol_observation_candidates(
        [source, _symbol_patch("stable")]
    )
    ambiguous, ambiguous_decision = versioned_symbol_observation_candidates(
        [source, _symbol_patch(None)]
    )

    assert same_symbol == []
    assert same_decision["version_invalidation_reasons"] == {
        "same_file_symbol_overlap": 1
    }
    assert ambiguous == []
    assert ambiguous_decision["version_invalidation_reasons"] == {
        "same_file_symbol_ambiguous": 1
    }


def test_v45_target_guard_closes_cross_request_invalidation_window() -> None:
    source = _python_read("stable")
    pending = {
        "source_observation_sha256": tool_observation_sha256(source),
        "source_paths": ["pkg/module.py"],
        "source_symbols": ["stable"],
    }

    invalid = versioned_evidence_target_guard(
        pending, [source, _symbol_patch("stable")]
    )
    preserved = versioned_evidence_target_guard(
        pending, [source, _symbol_patch("other")]
    )

    assert invalid["target_evidence_valid"] is False
    assert invalid["reason"] == "same_file_symbol_overlap"
    assert invalid["invalidating_group_index"] == 1
    assert preserved["target_evidence_valid"] is True
    assert preserved["symbol_disjoint_mutations"] == 1


def test_v45_target_guard_requires_unique_visible_source() -> None:
    source = _python_read("stable")
    pending = {
        "source_observation_sha256": tool_observation_sha256(source),
        "source_paths": ["pkg/module.py"],
        "source_symbols": ["stable"],
    }

    missing = versioned_evidence_target_guard(pending, [])
    duplicate = versioned_evidence_target_guard(pending, [source, source])

    assert missing["reason"] == "source_observation_not_unique"
    assert duplicate["reason"] == "source_observation_not_unique"
    assert duplicate["source_group_matches"] == 2


def test_active_v45_does_not_relax_v40_source_admission() -> None:
    source = _python_read("stable")
    disjoint_mutation = _symbol_patch("other")

    v40, v40_decision = grounded_observation_candidates(
        [source, disjoint_mutation]
    )
    active_v45, v45_decision = versioned_grounded_observation_candidates(
        [source, disjoint_mutation]
    )

    assert active_v45 == v40 == []
    assert v45_decision["candidate_group_indices"] == v40_decision[
        "candidate_group_indices"
    ]
    assert v45_decision["symbol_relaxation_enabled"] is False


def test_active_v45_marks_unlocalized_candidate_without_reordering() -> None:
    source = _command_group(
        "rg stable /testbed/pkg",
        "match without a concrete path " * 30,
    )

    v40, _ = grounded_observation_candidates([source])
    active_v45, decision = versioned_grounded_observation_candidates([source])

    assert v40 == [[source[1]]]
    assert active_v45 == v40 == [[source[1]]]
    assert decision["unlocalized_candidate_observations"] == 1
    assert decision["candidate_evidence"][0]["paths"] == []


def test_v45_target_guard_detects_shell_redirection_write() -> None:
    source = _python_read("stable")
    pending = {
        "source_observation_sha256": tool_observation_sha256(source),
        "source_paths": ["pkg/module.py"],
        "source_symbols": ["stable"],
    }
    shell_write = _command_group(
        "cat > /testbed/pkg/module.py <<'PY'\ndef stable():\n    return 2\nPY"
    )

    guard = versioned_evidence_target_guard(pending, [source, shell_write])

    assert guard["target_evidence_valid"] is False
    assert guard["reason"] == "same_file_symbol_overlap"


def test_version_graph_removes_stale_file_observation() -> None:
    groups = [
        _command_group("cat /testbed/pkg/drop.py", "old source"),
        _command_group("cat /testbed/pkg/keep.py", "stable source"),
        _command_group(
            "apply_patch <<'PATCH'\n"
            "*** Update File: pkg/drop.py\n"
            "@@\n-old\n+new\n"
            "PATCH"
        ),
        _command_group("cat /testbed/pkg/other.py", "other source"),
        _command_group("rg symbol /testbed/pkg/keep.py", "match"),
        _command_group("cat /testbed/pkg/final.py", "final source"),
    ]

    selected, decision = select_version_graph_groups(groups)

    # groups[0] rolls out; retained group 0 (keep.py) remains valid even though
    # a later mutation touches only drop.py.
    assert decision["stale_groups"] == 0
    assert selected
    assert decision["selected_groups"] == 5


def test_version_graph_breaks_at_mutated_retained_read() -> None:
    groups = [
        _command_group("cat /testbed/pkg/oldest.py", "oldest"),
        _command_group("cat /testbed/pkg/drop.py", "old source"),
        _command_group("cat /testbed/pkg/keep.py", "stable source"),
        _command_group(
            "apply_patch <<'PATCH'\n"
            "*** Update File: pkg/drop.py\n"
            "@@\n-old\n+new\n"
            "PATCH"
        ),
        _command_group("cat /testbed/pkg/other.py", "other"),
        _command_group("cat /testbed/pkg/final.py", "final"),
    ]

    selected, decision = select_version_graph_groups(groups)

    assert decision["stale_group_indices"] == [0]
    assert decision["eligible_islands"] == 1
    assert decision["selected_group_indices"] == [1, 2, 3, 4]
    assert len(selected) == 4


def test_post_mutation_v19_drops_stale_but_keeps_latest_risky_group() -> None:
    groups = [
        _command_group("cat /testbed/pkg/oldest.py", "oldest"),
        _command_group("cat /testbed/pkg/drop.py", "old source"),
        _command_group("cat /testbed/pkg/keep.py", "stable source"),
        _command_group(
            "apply_patch <<'PATCH'\n"
            "*** Update File: pkg/drop.py\n"
            "@@\n-old\n+new\n"
            "PATCH"
        ),
        _command_group("cat /testbed/pkg/other.py", "other"),
        _command_group("pytest -q", "1 failed"),
    ]
    groups[-1][1]["content"] = (
        "<returncode>1</returncode><output>1 failed</output>"
    )

    v17, v17_decision = select_reuse_groups(
        "coding_version_graph_v17",
        groups,
        latest_group_messages=groups[-1],
    )
    v19, v19_decision = select_reuse_groups(
        "coding_post_mutation_v19",
        groups,
        latest_group_messages=groups[-1],
    )

    assert v17_decision["stale_group_indices"] == [0]
    assert v17_decision["latest_group_protected"] is True
    assert v19_decision["stale_group_indices"] == [0]
    assert v19_decision["latest_group_protected"] is False
    assert v19_decision["latest_guard_enabled"] is False
    assert v19_decision["mode"] == "post_mutation_contiguous_island"
    assert v19 == [*v17, groups[-1]]

    v20, v20_decision = select_reuse_groups(
        "coding_post_mutation_dual_v20",
        groups,
        latest_group_messages=groups[-1],
    )
    assert v20 == v19
    assert v20_decision["mode"] == "post_mutation_dual_island"

    v22, v22_decision = select_reuse_groups(
        "coding_post_mutation_seam32_v22",
        groups,
        latest_group_messages=groups[-1],
    )
    assert v22 == v19
    assert v22_decision["mode"] == "post_mutation_dual_island"

    v23, v23_decision = select_reuse_groups(
        "coding_post_mutation_target_prefix_v23",
        groups,
        latest_group_messages=groups[-1],
    )
    assert v23 == v19
    assert v23_decision["mode"] == "post_mutation_dual_island"

    v28, v28_decision = select_reuse_groups(
        "coding_post_mutation_payoff_guard_v28",
        groups,
        latest_group_messages=groups[-1],
    )
    assert v28 == v19
    assert v28_decision["mode"] == "post_mutation_dual_island"

    v29, v29_decision = select_reuse_groups(
        "coding_post_mutation_payoff_guard_v29",
        groups,
        latest_group_messages=groups[-1],
    )
    assert v29 == v19
    assert v29_decision["mode"] == "post_mutation_dual_island"


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


def test_v31_abstains_after_mutation_and_executable_failure():
    mutation = group(
        "apply_patch <<'PATCH'\n"
        "*** Update File: pkg/module.py\n"
        "@@\n-old\n+new\n"
        "PATCH"
    )
    failure = group(
        "python -m pytest test_module.py",
        "<returncode>1</returncode><output>1 failed</output>",
    )
    for latest, expected in (
        (mutation, ["repository_mutation_command"]),
        (failure, ["executable_failure"]),
    ):
        groups = [group(f"echo {index}") for index in range(5)] + [latest]
        eligible, decision = select_reuse_groups(
            "coding_critical_event_abstain_v31",
            groups,
            latest_group_messages=latest,
        )
        assert eligible == groups[1:]
        assert decision["mode"] == "critical_event_dense_abstain"
        assert decision["critical_event_reasons"] == expected
        assert critical_coding_event_reasons(latest) == expected


def test_v31_reuses_after_search_miss_and_successful_test():
    search_miss = group(
        'rg "missing_symbol" module.py',
        "<returncode>1</returncode><output></output>",
    )
    successful_test = group(
        "python -m pytest test_module.py",
        "<returncode>0</returncode><output>1 passed</output>",
    )
    for latest in (search_miss, successful_test):
        groups = [group(f"echo {index}") for index in range(5)] + [latest]
        eligible, decision = select_reuse_groups(
            "coding_critical_event_abstain_v31",
            groups,
            latest_group_messages=latest,
        )
        assert eligible == groups[1:]
        assert decision["mode"] == "critical_event_general_reuse"
        assert decision["critical_event_reasons"] == []


def test_state_transition_guard_detects_open_write_mutation():
    mutation = group(
        "python - <<'PY'\n"
        "with open('pkg/module.py', 'w') as stream:\n"
        "    stream.write('changed')\n"
        "PY"
    )

    assert critical_coding_event_reasons(mutation) == [
        "repository_mutation_command"
    ]
    assert coding_state_transition_target_reasons([mutation]) == [
        "repository_mutation_command"
    ]


def test_state_transition_guard_vetoes_only_entry_to_read_phase():
    evidence_output = (
        "<returncode>0</returncode><output>"
        + "def implementation():\n    return 1\n" * 20
        + "</output>"
    )
    first_read = group("cat pkg/module.py", evidence_output)
    second_read = group("rg -n implementation pkg/module.py", evidence_output)
    generic = group("ls pkg")

    assert coding_state_transition_target_reasons(
        [generic, first_read]
    ) == ["readonly_evidence_phase_transition"]
    assert coding_state_transition_target_reasons(
        [first_read, second_read]
    ) == []


def test_state_transition_guard_detects_successful_execution_phase():
    successful = group(
        "python -m pytest tests/test_module.py",
        "<returncode>0</returncode><output>1 passed</output>",
    )

    assert is_successful_executable_evidence(successful)
    assert coding_state_transition_target_reasons(
        [group("cat README.md", "<returncode>0</returncode>"), successful]
    ) == ["successful_execution_phase_transition"]


def test_state_transition_guard_enforces_two_interaction_cooldown():
    mutation = group(
        "python -c \"from pathlib import Path; "
        "Path('pkg/module.py').write_text('changed')\""
    )
    successful = group(
        "python -m pytest tests/test_module.py",
        "<returncode>0</returncode><output>1 passed</output>",
    )
    generic = group("ls pkg")

    assert coding_state_transition_target_reasons(
        [mutation, generic, successful]
    ) == []
    assert coding_state_transition_target_reasons(
        [mutation, generic, generic, successful]
    ) == ["successful_execution_phase_transition"]


def test_version_validation_guard_is_one_shot_per_repository_version():
    mutation = group(
        "python -c \"from pathlib import Path; "
        "Path('pkg/module.py').write_text('changed')\""
    )
    successful = group(
        "python -m pytest tests/test_module.py",
        "<returncode>0</returncode><output>1 passed</output>",
    )

    assert is_successful_focused_validation(successful)
    assert coding_version_validation_target_reasons(
        [mutation, successful]
    ) == ["first_validation_of_version_before_submit"]
    assert coding_version_validation_target_reasons(
        [mutation, successful, successful]
    ) == []


def test_version_validation_guard_protects_failed_repair_decision():
    failure = group(
        "python -m pytest tests/test_module.py",
        "<returncode>1</returncode><output>1 failed</output>",
    )

    assert coding_version_validation_target_reasons([failure]) == [
        "executable_failure_before_repair"
    ]


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


def test_v28_payoff_guard_keeps_coding_protection_at_boundary():
    decision = post_mutation_payoff_guard(
        request_index=7,
        coding_candidate_tokens=1052,
        general_candidate_tokens=2795,
        copy_cap=4096,
    )

    assert decision["mode"] == "payoff_guard_post_mutation_protected"
    assert decision["future_target_upper_bound"] == 13
    assert decision["payoff_ratio"] > 0.60


def test_v28_payoff_guard_uses_general_when_protection_is_too_costly():
    decision = post_mutation_payoff_guard(
        request_index=8,
        coding_candidate_tokens=264,
        general_candidate_tokens=3378,
        copy_cap=4096,
    )

    assert decision["mode"] == "payoff_guard_general_middle_exact_prefix"
    assert decision["future_target_upper_bound"] == 12
    assert decision["payoff_ratio"] < 0.60


def test_v28_payoff_guard_abstains_when_branch_is_too_late():
    decision = post_mutation_payoff_guard(
        request_index=17,
        coding_candidate_tokens=4096,
        general_candidate_tokens=4096,
        copy_cap=4096,
    )

    assert decision["mode"] == "payoff_guard_dense_abstain_late_branch"
    assert decision["future_target_upper_bound"] == 3


def test_v29_stronger_threshold_falls_back_at_v28_protection_ratio():
    v28 = post_mutation_payoff_guard(
        request_index=7,
        coding_candidate_tokens=1052,
        general_candidate_tokens=2795,
        copy_cap=4096,
        payoff_ratio_threshold=0.60,
    )
    v29 = post_mutation_payoff_guard(
        request_index=7,
        coding_candidate_tokens=1052,
        general_candidate_tokens=2795,
        copy_cap=4096,
        payoff_ratio_threshold=1.20,
    )

    assert v28["mode"] == "payoff_guard_post_mutation_protected"
    assert v29["mode"] == "payoff_guard_general_middle_exact_prefix"
