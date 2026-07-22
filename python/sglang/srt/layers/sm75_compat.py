from typing import Optional, Tuple

CudaCapability = Tuple[Optional[int], Optional[int]]


def should_use_native_cuda_fallback(
    is_cuda_platform: bool,
    capability: CudaCapability,
) -> bool:
    if not is_cuda_platform:
        return False

    major, _ = capability
    return major is None or major < 8
