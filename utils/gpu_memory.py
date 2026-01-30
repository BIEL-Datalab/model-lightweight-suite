from __future__ import annotations

from typing import Optional


def get_gpu_memory_usage_mb(gpu_index: int = 0) -> float:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(mem_info.used / 1024 / 1024)
    except Exception:
        return 0.0


def clear_torch_cuda_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        return

