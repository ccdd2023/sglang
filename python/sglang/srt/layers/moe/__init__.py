from importlib import import_module

_EXPORTS = {
    "DeepEPMode": ("sglang.srt.layers.moe.utils", "DeepEPMode"),
    "MoeA2ABackend": ("sglang.srt.layers.moe.utils", "MoeA2ABackend"),
    "MoeRunner": ("sglang.srt.layers.moe.moe_runner", "MoeRunner"),
    "MoeRunnerBackend": ("sglang.srt.layers.moe.utils", "MoeRunnerBackend"),
    "MoeRunnerConfig": ("sglang.srt.layers.moe.moe_runner", "MoeRunnerConfig"),
    "get_deepep_config": ("sglang.srt.layers.moe.utils", "get_deepep_config"),
    "get_deepep_mode": ("sglang.srt.layers.moe.utils", "get_deepep_mode"),
    "get_moe_a2a_backend": ("sglang.srt.layers.moe.utils", "get_moe_a2a_backend"),
    "get_moe_runner_backend": (
        "sglang.srt.layers.moe.utils",
        "get_moe_runner_backend",
    ),
    "get_tbo_token_distribution_threshold": (
        "sglang.srt.layers.moe.utils",
        "get_tbo_token_distribution_threshold",
    ),
    "initialize_moe_config": (
        "sglang.srt.layers.moe.utils",
        "initialize_moe_config",
    ),
    "is_tbo_enabled": ("sglang.srt.layers.moe.utils", "is_tbo_enabled"),
    "should_use_flashinfer_cutlass_moe_fp4_allgather": (
        "sglang.srt.layers.moe.utils",
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
    ),
}


def __getattr__(name: str):
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
    "DeepEPMode",
    "MoeA2ABackend",
    "MoeRunner",
    "MoeRunnerBackend",
    "MoeRunnerConfig",
    "get_deepep_config",
    "get_deepep_mode",
    "get_moe_a2a_backend",
    "get_moe_runner_backend",
    "get_tbo_token_distribution_threshold",
    "initialize_moe_config",
    "is_tbo_enabled",
    "should_use_flashinfer_cutlass_moe_fp4_allgather",
]
