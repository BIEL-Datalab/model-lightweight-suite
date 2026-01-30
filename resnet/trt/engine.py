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

