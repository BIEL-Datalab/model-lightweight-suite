from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def cuda_device_context(device_id: int):
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

