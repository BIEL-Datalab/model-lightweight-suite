"""TensorRT 引擎序列化与推理验证工具。

本模块提供：
- ``serialize_engine``：基于 WTS 权重构建并保存 TensorRT 引擎（可选 INT8 校准）
- ``test_inference``：加载引擎并执行一次随机输入推理用于快速验证
"""

from __future__ import annotations

import os

import numpy as np

from .._io import ensure_parent_dir, require_exists
from .calibrator import create_int8_calibrator
from .network import build_resnet50_network
from .wts import load_weights


def serialize_engine(
    max_batch_size: int,
    use_int8: bool,
    weight_path: str,
    input_blob_name: str,
    input_h: int,
    input_w: int,
    output_size: int,
    output_blob_name: str,
    eps: float,
    calib_dir: str,
    calib_batch_size: int,
    calib_dataset_size: int,
    engine_path: str,
) -> None:
    """构建并保存 TensorRT 引擎文件。

    功能描述：
    加载 WTS 权重并创建显式 batch 的 TensorRT 网络，构建 ResNet50 拓扑并标记输出；
    当 ``use_int8`` 为 True 时创建 INT8 校准器并设置到 builder config；
    最终构建序列化引擎并写入 ``engine_path``。

    参数说明：
    - max_batch_size (int): 显式 batch 维度大小（shape 第 0 维）。
    - use_int8 (bool): 是否启用 INT8 校准与 INT8 构建标志。
    - weight_path (str): WTS 权重文件路径。
    - input_blob_name (str): 输入张量名称。
    - input_h (int): 输入高度。
    - input_w (int): 输入宽度。
    - output_size (int): 输出类别数。
    - output_blob_name (str): 输出张量名称。
    - eps (float): BatchNorm 数值稳定项。
    - calib_dir (str): 校准图片目录路径。
    - calib_batch_size (int): 校准 batch 大小。
    - calib_dataset_size (int): 校准数据集大小上限。
    - engine_path (str): 引擎输出路径。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - FileNotFoundError: 当权重文件或校准目录不存在时由 ``require_exists`` 触发。
    - RuntimeError: 当引擎构建失败时触发。
    - OSError: 当写入引擎文件失败时触发。

    使用示例：
    >>> from resnet.trt.engine import serialize_engine
    >>> serialize_engine(  # doctest: +SKIP
    ...     max_batch_size=1,
    ...     use_int8=False,
    ...     weight_path="model.wts",
    ...     input_blob_name="data",
    ...     input_h=224,
    ...     input_w=224,
    ...     output_size=10,
    ...     output_blob_name="prob",
    ...     eps=1e-5,
    ...     calib_dir="data_set/calib_images",
    ...     calib_batch_size=8,
    ...     calib_dataset_size=2000,
    ...     engine_path="model.engine",
    ... )
    """
    import tensorrt as trt

    if use_int8:
        require_exists(calib_dir, "校准图像目录")

    require_exists(weight_path, "WTS权重文件")
    ensure_parent_dir(engine_path)

    weight_map = load_weights(weight_path)

    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    input_tensor = network.add_input(
        name=input_blob_name,
        dtype=trt.float32,
        shape=trt.Dims([max_batch_size, 3, input_h, input_w]),
    )
    build_resnet50_network(
        network=network,
        input_tensor=input_tensor,
        weight_map=weight_map,
        input_h=input_h,
        input_w=input_w,
        output_size=output_size,
        output_blob_name=output_blob_name,
        eps=eps,
        use_int8=use_int8,
    )

    if use_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        calibrator = create_int8_calibrator(
            calib_image_dir=calib_dir,
            batch_size=calib_batch_size,
            input_shape=(3, input_h, input_w),
            cache_file="calib_cache.bin",
            input_h=input_h,
            input_w=input_w,
            calib_dataset_size=calib_dataset_size,
        )
        config.int8_calibrator = calibrator
    else:
        config.set_flag(trt.BuilderFlag.FP32)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("引擎构建失败: build_serialized_network返回None")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)


def test_inference(engine_path: str) -> None:
    """加载 TensorRT 引擎并执行一次推理验证。

    功能描述：
    反序列化 ``engine_path`` 指向的引擎，创建执行上下文并用随机输入执行一次异步推理，
    主要用于验证引擎可被正确加载与执行。

    参数说明：
    - engine_path (str): TensorRT 引擎文件路径。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - FileNotFoundError: 当引擎文件不存在时由 ``require_exists`` 触发。
    - RuntimeError: 当引擎加载或上下文创建失败时触发。
    - Exception: 当 PyCUDA/TensorRT 执行失败时由底层依赖触发。

    使用示例：
    >>> from resnet.trt.engine import test_inference
    >>> test_inference("model.engine")  # doctest: +SKIP
    """
    import pycuda.driver as cuda
    import tensorrt as trt

    require_exists(engine_path, "TensorRT引擎文件")

    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("引擎加载失败")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("执行上下文创建失败")

    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)

    host_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)

    rng = np.random.default_rng(42)
    test_data = rng.standard_normal(size=tuple(input_shape)).astype(np.float32)
    np.copyto(host_input, test_data.ravel())

    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)
    bindings = [int(device_input), int(device_output)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    device_input.free()
    device_output.free()
