from importlib import import_module

_EXPORTS = {
    "MoeRunner": ("sglang.srt.layers.moe.moe_runner.runner", "MoeRunner"),
    "MoeRunnerConfig": ("sglang.srt.layers.moe.moe_runner.base", "MoeRunnerConfig"),
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


__all__ = ["MoeRunner", "MoeRunnerConfig"]
