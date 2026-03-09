from importlib import import_module

_EXPORTS = {
    "AfmoeConfig": ("sglang.srt.configs.afmoe", "AfmoeConfig"),
    "BailingHybridConfig": (
        "sglang.srt.configs.bailing_hybrid",
        "BailingHybridConfig",
    ),
    "ChatGLMConfig": ("sglang.srt.configs.chatglm", "ChatGLMConfig"),
    "DbrxConfig": ("sglang.srt.configs.dbrx", "DbrxConfig"),
    "DeepseekVL2Config": ("sglang.srt.configs.deepseekvl2", "DeepseekVL2Config"),
    "DotsOCRConfig": ("sglang.srt.configs.dots_ocr", "DotsOCRConfig"),
    "DotsVLMConfig": ("sglang.srt.configs.dots_vlm", "DotsVLMConfig"),
    "ExaoneConfig": ("sglang.srt.configs.exaone", "ExaoneConfig"),
    "FalconH1Config": ("sglang.srt.configs.falcon_h1", "FalconH1Config"),
    "GraniteMoeHybridConfig": (
        "sglang.srt.configs.granitemoehybrid",
        "GraniteMoeHybridConfig",
    ),
    "JetNemotronConfig": (
        "sglang.srt.configs.jet_nemotron",
        "JetNemotronConfig",
    ),
    "JetVLMConfig": ("sglang.srt.configs.jet_vlm", "JetVLMConfig"),
    "KimiK25Config": ("sglang.srt.configs.kimi_k25", "KimiK25Config"),
    "KimiLinearConfig": ("sglang.srt.configs.kimi_linear", "KimiLinearConfig"),
    "KimiVLConfig": ("sglang.srt.configs.kimi_vl", "KimiVLConfig"),
    "Lfm2Config": ("sglang.srt.configs.lfm2", "Lfm2Config"),
    "Lfm2MoeConfig": ("sglang.srt.configs.lfm2_moe", "Lfm2MoeConfig"),
    "LongcatFlashConfig": (
        "sglang.srt.configs.longcat_flash",
        "LongcatFlashConfig",
    ),
    "MoonViTConfig": ("sglang.srt.configs.kimi_vl_moonvit", "MoonViTConfig"),
    "MultiModalityConfig": (
        "sglang.srt.configs.janus_pro",
        "MultiModalityConfig",
    ),
    "NemotronHConfig": ("sglang.srt.configs.nemotron_h", "NemotronHConfig"),
    "NemotronH_Nano_VL_V2_Config": (
        "sglang.srt.configs.nano_nemotron_vl",
        "NemotronH_Nano_VL_V2_Config",
    ),
    "Olmo3Config": ("sglang.srt.configs.olmo3", "Olmo3Config"),
    "Qwen3NextConfig": ("sglang.srt.configs.qwen3_next", "Qwen3NextConfig"),
    "Qwen3_5Config": ("sglang.srt.configs.qwen3_5", "Qwen3_5Config"),
    "Qwen3_5MoeConfig": ("sglang.srt.configs.qwen3_5", "Qwen3_5MoeConfig"),
    "Step3TextConfig": ("sglang.srt.configs.step3_vl", "Step3TextConfig"),
    "Step3VLConfig": ("sglang.srt.configs.step3_vl", "Step3VLConfig"),
    "Step3VisionEncoderConfig": (
        "sglang.srt.configs.step3_vl",
        "Step3VisionEncoderConfig",
    ),
    "Step3p5Config": ("sglang.srt.configs.step3p5", "Step3p5Config"),
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
    "AfmoeConfig",
    "BailingHybridConfig",
    "ChatGLMConfig",
    "DbrxConfig",
    "DeepseekVL2Config",
    "DotsOCRConfig",
    "DotsVLMConfig",
    "ExaoneConfig",
    "FalconH1Config",
    "GraniteMoeHybridConfig",
    "JetNemotronConfig",
    "JetVLMConfig",
    "KimiK25Config",
    "KimiLinearConfig",
    "KimiVLConfig",
    "Lfm2Config",
    "Lfm2MoeConfig",
    "LongcatFlashConfig",
    "MoonViTConfig",
    "MultiModalityConfig",
    "NemotronHConfig",
    "NemotronH_Nano_VL_V2_Config",
    "Olmo3Config",
    "Qwen3NextConfig",
    "Qwen3_5Config",
    "Qwen3_5MoeConfig",
    "Step3TextConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3p5Config",
]
