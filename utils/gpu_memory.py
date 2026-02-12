"""GPU 显存辅助工具。

本模块提供轻量的 GPU 显存查询与 PyTorch CUDA 缓存清理函数；当相关依赖不可用或运行环境不支持时，
函数会以“尽力而为”的方式返回默认值或直接返回，不抛出异常。
"""

from __future__ import annotations

from typing import Optional


def get_gpu_memory_usage_mb(gpu_index: int = 0) -> float:
    """获取指定 GPU 的已用显存（MB）。

    功能描述：
    优先通过 ``pynvml`` 查询 GPU 显存使用情况；若依赖不可用或查询失败，则返回 ``0.0``。

    参数说明：
    - gpu_index (int): GPU 索引，默认 0。

    返回值说明：
    - float: 已用显存（MB）；当无法获取时返回 ``0.0``。

    可能抛出的异常：
    - 无。函数内部捕获所有异常并返回默认值。

    使用示例：
    >>> from utils.gpu_memory import get_gpu_memory_usage_mb
    >>> isinstance(get_gpu_memory_usage_mb(), float)
    True
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(mem_info.used / 1024 / 1024)
    except Exception:
        return 0.0


def clear_torch_cuda_cache() -> None:
    """清理 PyTorch CUDA 缓存。

    功能描述：
    调用 ``torch.cuda.empty_cache()`` 释放 PyTorch 维护的 CUDA 缓存；当 PyTorch 不可用或运行环境不支持 CUDA 时，
    函数直接返回。

    参数说明：
    - 无。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - 无。函数内部捕获所有异常并直接返回。

    使用示例：
    >>> from utils.gpu_memory import clear_torch_cuda_cache
    >>> clear_torch_cuda_cache()
    """
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        return
