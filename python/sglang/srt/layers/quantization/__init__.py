from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Type

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import is_cuda, is_hip, mxfp_supported


class LazyQuantizationMethods(Mapping[str, Type[QuantizationConfig]]):
    def __init__(self, specs: dict[str, tuple[str, str]]):
        self._specs = dict(specs)
        self._cache: dict[str, Type[QuantizationConfig]] = {}

    def __getitem__(self, key: str) -> Type[QuantizationConfig]:
        try:
            module_name, class_name = self._specs[key]
        except KeyError as exc:
            raise KeyError(key) from exc

        if key not in self._cache:
            module = import_module(module_name)
            self._cache[key] = getattr(module, class_name)
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


_QUANTIZATION_METHOD_SPECS: dict[str, tuple[str, str]] = {
    "fp8": ("sglang.srt.layers.quantization.fp8", "Fp8Config"),
    "mxfp8": ("sglang.srt.layers.quantization.fp8", "Fp8Config"),
    "blockwise_int8": (
        "sglang.srt.layers.quantization.blockwise_int8",
        "BlockInt8Config",
    ),
    "modelopt": ("sglang.srt.layers.quantization.modelopt_quant", "ModelOptFp8Config"),
    "modelopt_fp8": (
        "sglang.srt.layers.quantization.modelopt_quant",
        "ModelOptFp8Config",
    ),
    "modelopt_fp4": (
        "sglang.srt.layers.quantization.modelopt_quant",
        "ModelOptFp4Config",
    ),
    "w8a8_int8": ("sglang.srt.layers.quantization.w8a8_int8", "W8A8Int8Config"),
    "w8a8_fp8": ("sglang.srt.layers.quantization.w8a8_fp8", "W8A8Fp8Config"),
    "awq": ("sglang.srt.layers.quantization.awq", "AWQConfig"),
    "awq_marlin": ("sglang.srt.layers.quantization.awq", "AWQMarlinConfig"),
    "bitsandbytes": (
        "sglang.srt.layers.quantization.bitsandbytes",
        "BitsAndBytesConfig",
    ),
    "gguf": ("sglang.srt.layers.quantization.gguf", "GGUFConfig"),
    "gptq": ("sglang.srt.layers.quantization.gptq", "GPTQConfig"),
    "gptq_marlin": ("sglang.srt.layers.quantization.gptq", "GPTQMarlinConfig"),
    "moe_wna16": ("sglang.srt.layers.quantization.moe_wna16", "MoeWNA16Config"),
    "compressed-tensors": (
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors",
        "CompressedTensorsConfig",
    ),
    "qoq": ("sglang.srt.layers.quantization.qoq", "QoQConfig"),
    "w4afp8": ("sglang.srt.layers.quantization.w4afp8", "W4AFp8Config"),
    "petit_nvfp4": ("sglang.srt.layers.quantization.petit", "PetitNvFp4Config"),
    "fbgemm_fp8": ("sglang.srt.layers.quantization.fpgemm_fp8", "FBGEMMFp8Config"),
    "quark": ("sglang.srt.layers.quantization.quark.quark", "QuarkConfig"),
    "auto-round": ("sglang.srt.layers.quantization.auto_round", "AutoRoundConfig"),
    "modelslim": (
        "sglang.srt.layers.quantization.modelslim.modelslim",
        "ModelSlimConfig",
    ),
    "quark_int4fp8_moe": (
        "sglang.srt.layers.quantization.quark_int4fp8_moe",
        "QuarkInt4Fp8Config",
    ),
}

if is_cuda() or (mxfp_supported() and is_hip()):
    _QUANTIZATION_METHOD_SPECS["mxfp4"] = (
        "sglang.srt.layers.quantization.mxfp4",
        "Mxfp4Config",
    )

QUANTIZATION_METHODS: Mapping[str, Type[QuantizationConfig]] = LazyQuantizationMethods(
    _QUANTIZATION_METHOD_SPECS
)


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )

    return QUANTIZATION_METHODS[quantization]
