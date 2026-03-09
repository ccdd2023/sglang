from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.utils import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class Fp8GemmRunnerBackend(Enum):
    """Enum for FP8 GEMM runner backend selection."""

    AUTO = "auto"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_DEEPGEMM = "flashinfer_deepgemm"
    CUTLASS = "cutlass"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    AITER = "aiter"

    def is_auto(self) -> bool:
        return self == Fp8GemmRunnerBackend.AUTO

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_deepgemm(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER_DEEPGEMM

    def is_cutlass(self) -> bool:
        return self == Fp8GemmRunnerBackend.CUTLASS

    def is_deep_gemm(self) -> bool:
        return self == Fp8GemmRunnerBackend.DEEP_GEMM

    def is_triton(self) -> bool:
        return self == Fp8GemmRunnerBackend.TRITON

    def is_aiter(self) -> bool:
        return self == Fp8GemmRunnerBackend.AITER


FP8_GEMM_RUNNER_BACKEND: Fp8GemmRunnerBackend | None = None


def initialize_fp8_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP8 GEMM configuration."""
    global FP8_GEMM_RUNNER_BACKEND

    backend = server_args.fp8_gemm_runner_backend

    if backend == "auto":
        if envs.SGLANG_ENABLE_FLASHINFER_FP8_GEMM.get():
            backend = "flashinfer_trtllm"
        elif envs.SGLANG_SUPPORT_CUTLASS_BLOCK_FP8.get():
            backend = "cutlass"
    else:
        if (
            envs.SGLANG_ENABLE_FLASHINFER_FP8_GEMM.get()
            or envs.SGLANG_SUPPORT_CUTLASS_BLOCK_FP8.get()
        ):
            logger.warning(
                f"FP8 GEMM backend set to '{backend}' via --fp8-gemm-backend overrides "
                "environment variables SGLANG_ENABLE_FLASHINFER_FP8_GEMM and "
                "SGLANG_SUPPORT_CUTLASS_BLOCK_FP8. Using server argument value."
            )

    if backend == "auto" and is_sm120_supported():
        backend = "triton"

    FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend(backend)


def get_fp8_gemm_runner_backend() -> Fp8GemmRunnerBackend:
    """Get the current FP8 GEMM runner backend."""
    global FP8_GEMM_RUNNER_BACKEND
    if FP8_GEMM_RUNNER_BACKEND is None:
        FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend.AUTO
    return FP8_GEMM_RUNNER_BACKEND


__all__ = [
    "FP8_GEMM_RUNNER_BACKEND",
    "Fp8GemmRunnerBackend",
    "get_fp8_gemm_runner_backend",
    "initialize_fp8_gemm_config",
]
