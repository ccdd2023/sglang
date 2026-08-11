from benchmark.multi_workflow.analyze_discordant_repeat_treatment_attribution import (
    action_signature,
    canonical_tool_calls,
    common_prefix_length,
    trajectory_interactions,
)


def test_tool_call_ids_are_removed_but_arguments_remain() -> None:
    left = {
        "tool_calls": [
            {
                "id": "local-a",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"sed -n 1,4p x.py"}',
                },
            }
        ]
    }
    right = {
        "tool_calls": [
            {
                "id": "local-b",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"sed -n 1,4p x.py"}',
                },
            }
        ]
    }
    assert canonical_tool_calls(left) == canonical_tool_calls(right)


def test_interaction_identity_includes_reasoning_and_tool_result() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "issue"},
        {
            "role": "assistant",
            "content": "inspect",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"cat x.py"}',
                    }
                }
            ],
        },
        {"role": "tool", "tool_call_id": "ignored", "content": "source"},
        {"role": "assistant", "content": "done", "tool_calls": []},
    ]
    groups = trajectory_interactions(messages)
    assert groups == [
        {
            "assistant_content": "inspect",
            "tool_calls": [
                {"name": "bash", "arguments": {"command": "cat x.py"}}
            ],
            "tool_results": ["source"],
        },
        {"assistant_content": "done", "tool_calls": [], "tool_results": []},
    ]
    assert action_signature(groups[0]) == groups[0]["tool_calls"]
    assert action_signature(groups[1]) == {"final": "done"}


def test_common_prefix_stops_on_first_behavior_difference() -> None:
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_length([], [1]) == 0
