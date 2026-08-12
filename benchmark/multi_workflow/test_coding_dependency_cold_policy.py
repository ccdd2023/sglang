from benchmark.multi_workflow.coding_reuse_policy import (
    coding_dependency_graph_relations,
    coding_dependency_graph_target_guard,
    coding_dependency_relations,
    coding_dependency_target_guard,
    cold_natural_repository_code_candidates,
    dependency_graph_cold_repository_code_candidates,
    dependency_graph_lcb_cost_estimate,
    dependency_graph_mean_cost_estimate,
    natural_repository_code_candidates,
    search_file_section_dependency_cold_candidates,
    visible_python_dependency_graph,
)


def _group(command: str, output: str = "") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
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


def _read() -> list[dict]:
    return _group(
        "sed -n '1,240p' /testbed/pkg/module.py",
        "def important_value():\n    return 1\n" + "# code\n" * 80,
    )


def test_search_file_sections_follow_paths_and_stay_dependency_cold() -> None:
    search = _group(
        "grep -RIn --include='*.py' 'value' /testbed/pkg",
        "\n"
        + "pkg/a.py:10:def a_value(): return 1\n" * 10
        + "pkg/b.py:20:def b_value(): return 2\n" * 10,
    )
    candidates, decision = search_file_section_dependency_cold_candidates(
        [search]
    )
    assert len(candidates) == 2
    assert decision["literal_file_sections"] == 2
    assert decision["dependency_cold_sections"] == 2
    assert [row["paths"] for row in decision["candidate_evidence"]] == [
        ["pkg/a.py"],
        ["pkg/b.py"],
    ]
    assert all(
        row["candidate_char_end"] > row["candidate_char_start"]
        for row in decision["candidate_evidence"]
    )


def test_dependency_relation_marks_same_path_and_symbol_hot() -> None:
    later = _group(
        'grep -n "important_value" /testbed/pkg/module.py',
        "12:def important_value():\n",
    )
    relations = coding_dependency_relations(
        source_paths={"pkg/module.py"},
        source_symbols={"important_value"},
        later_groups=[later],
    )
    assert len(relations) == 1
    assert relations[0]["exact_paths"] == ["pkg/module.py"]
    assert relations[0]["shared_symbols"] == ["important_value"]


def test_cold_candidate_filter_protects_hot_but_keeps_unrelated() -> None:
    read = _read()
    unrelated = _group("cat /testbed/other/file.py", "def other():\n    pass\n")
    hot = _group('grep -n "important_value" /testbed/pkg/module.py')

    cold_candidates, cold = cold_natural_repository_code_candidates(
        [read, unrelated]
    )
    hot_candidates, guarded = cold_natural_repository_code_candidates(
        [read, hot]
    )

    assert cold_candidates == [[read[1]]]
    assert cold["dependency_hot_observations_protected"] == 0
    assert [read[1]] not in hot_candidates
    assert guarded["dependency_hot_observations_protected"] == 1


def test_target_guard_recomputes_hot_and_allows_cold() -> None:
    read = _read()
    _, decision = natural_repository_code_candidates([read])
    evidence = decision["candidate_evidence"][0]
    pending = {
        "source_group_sha256": evidence["group_sha256"],
        "source_observation_sha256": evidence["observation_sha256"],
        "source_paths": evidence["paths"],
        "source_symbols": evidence["symbols"],
        "repository_scope_dependency": False,
    }
    unrelated = _group("cat /testbed/other/file.py", "def other():\n    pass\n")
    hot = _group('grep -n "important_value" /testbed/pkg/module.py')

    cold_guard = coding_dependency_target_guard(pending, [read, unrelated])
    hot_guard = coding_dependency_target_guard(pending, [read, hot])

    assert cold_guard["target_evidence_valid"] is True
    assert cold_guard["dependency_hot"] is False
    assert hot_guard["target_evidence_valid"] is False
    assert hot_guard["dependency_hot"] is True
    assert hot_guard["reason"] == "coding_dependency_hot_protected"


def test_visible_graph_resolves_qualified_symbols_and_import_aliases() -> None:
    source = _group(
        "cat /testbed/pkg/module.py",
        "class Handler:\n"
        "    def __init__(self):\n        pass\n"
        "    def execute(self):\n        return 1\n",
    )
    graph = visible_python_dependency_graph(path="pkg/module.py", group=source)

    assert graph["parse_status"] == "parsed"
    assert "Handler.execute" in graph["qualified_symbols"]
    assert "pkg.module.Handler.execute" in graph["qualified_symbols"]

    consumer = _group(
        "cat /testbed/pkg/consumer.py",
        "from pkg.module import Handler as H\nvalue = H().execute()\n",
    )
    relations = coding_dependency_graph_relations(
        source_paths={"pkg/module.py"},
        source_symbols={"Handler", "__init__", "execute"},
        source_graph=graph,
        later_groups=[consumer],
    )
    assert "one_hop_import" in relations[0]["relation_kinds"]
    assert "qualified_symbol" in relations[0]["relation_kinds"]
    assert relations[0]["import_matches"] == ["pkg.module.Handler"]


def test_common_dunder_alone_does_not_create_cross_file_hot_relation() -> None:
    source = _group(
        "cat /testbed/pkg/module.py",
        "class Handler:\n    def __init__(self):\n        pass\n",
    )
    graph = visible_python_dependency_graph(path="pkg/module.py", group=source)
    unrelated = _group(
        "cat /testbed/other/model.py",
        "class Other:\n    def __init__(self):\n        pass\n",
    )

    assert coding_dependency_graph_relations(
        source_paths={"pkg/module.py"},
        source_symbols={"Handler", "__init__"},
        source_graph=graph,
        later_groups=[unrelated],
    ) == []


def test_graph_candidate_and_target_guard_protect_one_hop_consumer() -> None:
    source = _group(
        "cat /testbed/pkg/module.py",
        "def important_value():\n    return 1\n" + "# code\n" * 80,
    )
    consumer = _group(
        "cat /testbed/pkg/consumer.py",
        "from pkg.module import important_value\n"
        "result = important_value()\n"
        + "# consumer\n" * 50,
    )
    candidates, decision = dependency_graph_cold_repository_code_candidates(
        [source]
    )
    assert candidates == [[source[1]]]
    evidence = decision["candidate_evidence"][0]
    pending = {
        "source_group_sha256": evidence["group_sha256"],
        "source_observation_sha256": evidence["observation_sha256"],
        "source_paths": evidence["paths"],
        "source_symbols": evidence["symbols"],
        "source_dependency_graph": evidence["dependency_graph"],
        "repository_scope_dependency": False,
    }

    guard = coding_dependency_graph_target_guard(
        pending, [source, consumer]
    )
    assert guard["target_evidence_valid"] is False
    assert guard["dependency_hot"] is True
    assert guard["reason"] == "coding_dependency_graph_hot_protected"
    assert "one_hop_import" in guard["dependency_relations"][0][
        "relation_kinds"
    ]


def test_dependency_graph_lcb_is_stricter_than_mean_prediction() -> None:
    rejected = dependency_graph_lcb_cost_estimate(
        island_tokens=400, target_prompt_tokens=4_000
    )
    admitted = dependency_graph_lcb_cost_estimate(
        island_tokens=2_000, target_prompt_tokens=8_000
    )

    assert rejected["predicted_cache_ready_saving_ms"] > 0
    assert rejected["lower_bound_cache_ready_saving_ms"] < 0
    assert rejected["reuse_admitted"] is False
    assert admitted["lower_bound_cache_ready_saving_ms"] > 0
    assert admitted["reuse_admitted"] is True


def test_dependency_graph_mean_counterfactual_removes_only_q10_penalty() -> None:
    lcb = dependency_graph_lcb_cost_estimate(
        island_tokens=643, target_prompt_tokens=2041
    )
    mean = dependency_graph_mean_cost_estimate(
        island_tokens=643, target_prompt_tokens=2041
    )
    assert mean["predicted_cache_ready_saving_ms"] == lcb[
        "predicted_cache_ready_saving_ms"
    ]
    assert mean["reuse_admitted"] is True
    assert lcb["reuse_admitted"] is False
    assert "lower_bound_cache_ready_saving_ms" not in mean
