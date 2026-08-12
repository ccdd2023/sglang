from benchmark.multi_workflow.analyze_graph_mean_canary_counterfactual import command


def test_command_decodes_serialized_arguments() -> None:
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"sed -n 1,10p x.py"}',
                }
            }
        ]
    }
    assert command(message) == "sed -n 1,10p x.py"
