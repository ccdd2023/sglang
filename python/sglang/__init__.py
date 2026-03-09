"""SGLang public APIs."""

from importlib import import_module

from sglang.version import __version__

_EXPORTS = {
    "Anthropic": ("sglang.lang.backend.anthropic", "Anthropic"),
    "LiteLLM": ("sglang.lang.backend.litellm", "LiteLLM"),
    "OpenAI": ("sglang.lang.backend.openai", "OpenAI"),
    "Runtime": ("sglang.lang.api", "Runtime"),
    "RuntimeEndpoint": ("sglang.lang.backend.runtime_endpoint", "RuntimeEndpoint"),
    "ServerArgs": ("sglang.srt.server_args", "ServerArgs"),
    "VertexAI": ("sglang.lang.backend.vertexai", "VertexAI"),
    "assistant": ("sglang.lang.api", "assistant"),
    "assistant_begin": ("sglang.lang.api", "assistant_begin"),
    "assistant_end": ("sglang.lang.api", "assistant_end"),
    "flush_cache": ("sglang.lang.api", "flush_cache"),
    "function": ("sglang.lang.api", "function"),
    "gen": ("sglang.lang.api", "gen"),
    "gen_int": ("sglang.lang.api", "gen_int"),
    "gen_string": ("sglang.lang.api", "gen_string"),
    "get_server_info": ("sglang.lang.api", "get_server_info"),
    "global_config": ("sglang.global_config", "global_config"),
    "greedy_token_selection": ("sglang.lang.choices", "greedy_token_selection"),
    "image": ("sglang.lang.api", "image"),
    "select": ("sglang.lang.api", "select"),
    "separate_reasoning": ("sglang.lang.api", "separate_reasoning"),
    "set_default_backend": ("sglang.lang.api", "set_default_backend"),
    "system": ("sglang.lang.api", "system"),
    "system_begin": ("sglang.lang.api", "system_begin"),
    "system_end": ("sglang.lang.api", "system_end"),
    "token_length_normalized": (
        "sglang.lang.choices",
        "token_length_normalized",
    ),
    "unconditional_likelihood_normalized": (
        "sglang.lang.choices",
        "unconditional_likelihood_normalized",
    ),
    "user": ("sglang.lang.api", "user"),
    "user_begin": ("sglang.lang.api", "user_begin"),
    "user_end": ("sglang.lang.api", "user_end"),
    "video": ("sglang.lang.api", "video"),
    "Engine": ("sglang.srt.entrypoints.engine", "Engine"),
}


def __getattr__(name: str):
    if name == "__version__":
        return __version__

    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "Anthropic",
    "Engine",
    "LiteLLM",
    "OpenAI",
    "Runtime",
    "RuntimeEndpoint",
    "ServerArgs",
    "VertexAI",
    "__version__",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "global_config",
    "greedy_token_selection",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "user",
    "user_begin",
    "user_end",
    "video",
]
