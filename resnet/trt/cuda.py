"""CUDA 设备上下文管理工具。

本模块基于 PyCUDA 提供一个上下文管理器，用于在指定设备上创建并释放 CUDA context，
便于在 TensorRT 构建/推理时确保正确的设备上下文。
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def cuda_device_context(device_id: int) -> object:
    """创建指定 CUDA 设备的上下文并在退出时释放。

    功能描述：
    使用 PyCUDA 初始化 CUDA，检查 ``device_id`` 合法性后创建 context，并在 ``with`` 代码块结束后确保释放。

    参数说明：
    - device_id (int): CUDA 设备索引。

    返回值说明：
    - object: 上下文管理器的生成器对象（由 ``contextmanager`` 包装）。

    可能抛出的异常：
    - ValueError: 当 ``device_id`` 超出可用设备数量范围时触发。
    - Exception: 当 PyCUDA 初始化或 context 创建失败时由底层依赖触发。

    使用示例：
    >>> from resnet.trt.cuda import cuda_device_context
    >>> with cuda_device_context(0):  # doctest: +SKIP
    ...     pass
    """
    import pycuda.driver as cuda

    cuda.init()
    if device_id >= cuda.Device.count():
        raise ValueError(f"CUDA设备索引超出范围: {device_id}")

    device = cuda.Device(device_id)
    ctx = device.make_context()
    try:
        yield
    finally:
        try:
            ctx.pop()
        finally:
            ctx.detach()
