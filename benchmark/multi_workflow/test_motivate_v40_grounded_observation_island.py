from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from benchmark.multi_workflow.motivate_v40_grounded_observation_island import (
    _mutation_invalidates,
    select_grounded_observation,
)


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content}


def _assistant(command: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {"command": command},
                }
            }
        ],
    }


def test_mutation_invalidates_same_path() -> None:
    source = [_assistant("sed -n '1,80p' pkg/mod.py"), _tool("x")]
    mutation = [
        _assistant("sed -i 's/x/y/' pkg/mod.py"),
        _tool("<returncode>0</returncode>"),
    ]
    assert _mutation_invalidates(source, [mutation])


def test_selector_never_selects_assistant_tokens() -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                "source": 1,
                "evidence": 2,
                "<": 3,
                ">": 4,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    content = "source evidence " * 100
    group = [
        _assistant("sed -n '1,200p' pkg/mod.py"),
        _tool(content + "\n<returncode>0</returncode>"),
    ]
    selected, diagnostics = select_grounded_observation([group], tokenizer)
    assert selected is not None
    assert selected["group_index"] == 0
    assert diagnostics["eligible_observations"] == 1
