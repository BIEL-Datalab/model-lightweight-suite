"""模型加载与推理后端适配工具。

本模块为评估流水线提供统一的模型加载接口，支持：
- PyTorch ``nn.Module``
- ONNX Runtime ``InferenceSession``
- TensorRT ``engine/context``
并提供输入输出张量名称的统一查询函数。
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def load_pytorch_resnet50(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """加载 ResNet50（PyTorch）并载入权重。

    功能描述：
    构建 ``resnet50(weights=None)``，按 ``num_classes`` 重建最后一层全连接，
    从 ``model_path`` 加载权重并迁移到指定 ``device``，最后切换到 eval 模式。

    参数说明：
    - model_path (str): 权重文件路径。
    - num_classes (int): 分类类别数。
    - device (torch.device): 推理设备。

    返回值说明：
    - nn.Module: 已加载权重且处于 eval 模式的模型。

    可能抛出的异常：
    - FileNotFoundError: 当权重文件不存在时触发。
    - RuntimeError: 权重加载或模型迁移失败时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from evaluation.modelopt.loaders import load_pytorch_resnet50
    >>> _ = load_pytorch_resnet50("model.pth", num_classes=10, device=torch.device("cpu"))  # doctest: +SKIP
    """
    from torchvision.models.resnet import resnet50

    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    map_location = "cpu" if device.type == "cpu" else ("cuda" if device.index is None else f"cuda:{device.index}")
    state_dict = torch.load(model_path, map_location=map_location, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_onnx_session(model_path: str, device: torch.device, providers: list[Any]) -> Any:
    """创建 ONNX Runtime 推理会话。

    功能描述：
    使用给定 ``providers`` 创建 ``onnxruntime.InferenceSession`` 并返回。
    ``device`` 参数用于保持调用侧接口一致性，当前实现不直接使用。

    参数说明：
    - model_path (str): ONNX 模型文件路径。
    - device (torch.device): 推理设备占位参数（当前实现不使用）。
    - providers (list[Any]): ONNX Runtime providers 配置列表。

    返回值说明：
    - Any: ONNX Runtime 推理会话对象（通常为 ``onnxruntime.InferenceSession``）。

    可能抛出的异常：
    - Exception: 当 onnxruntime 未安装或会话创建失败时由依赖触发。

    使用示例：
    >>> import torch
    >>> from evaluation.modelopt.loaders import load_onnx_session, build_onnx_providers
    >>> _ = load_onnx_session("model.onnx", device=torch.device("cpu"), providers=build_onnx_providers(torch.device("cpu")))  # doctest: +SKIP
    """
    import onnxruntime as ort

    return ort.InferenceSession(model_path, providers=providers)


def build_onnx_providers(device: torch.device) -> list[Any]:
    """构建 ONNX 原始模型的 providers 配置列表。

    功能描述：
    - CPU 设备：仅返回 ``CPUExecutionProvider``；
    - CUDA 设备：返回 ``CUDAExecutionProvider`` + ``CPUExecutionProvider`` 的组合，并配置显存上限等参数。

    参数说明：
    - device (torch.device): 推理设备。

    返回值说明：
    - list[Any]: providers 配置列表。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> import torch
    >>> from evaluation.modelopt.loaders import build_onnx_providers
    >>> isinstance(build_onnx_providers(torch.device("cpu")), list)
    True
    """
    if device.type != "cuda":
        return ["CPUExecutionProvider"]
    device_id = 0 if device.index is None else device.index
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 1 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                "enable_cuda_graph": False,
            },
        ),
        "CPUExecutionProvider",
    ]


def build_onnx_quantized_providers(device: torch.device, trt_cache_dir: str) -> list[Any]:
    """构建 ONNX 量化模型（含 TensorRT EP）的 providers 配置列表。

    功能描述：
    - CPU 设备：仅返回 ``CPUExecutionProvider``；
    - CUDA 设备：优先使用 ``TensorrtExecutionProvider``，并启用引擎缓存目录 ``trt_cache_dir``，
      同时配置 ``CUDAExecutionProvider`` 与 ``CPUExecutionProvider`` 作为回退。

    参数说明：
    - device (torch.device): 推理设备。
    - trt_cache_dir (str): TensorRT 引擎缓存目录（不存在时会创建）。

    返回值说明：
    - list[Any]: providers 配置列表。

    可能抛出的异常：
    - OSError: 当缓存目录创建失败时触发。

    使用示例：
    >>> import torch
    >>> from evaluation.modelopt.loaders import build_onnx_quantized_providers
    >>> isinstance(build_onnx_quantized_providers(torch.device("cpu"), "trt_cache"), list)
    True
    """
    if device.type != "cuda":
        return ["CPUExecutionProvider"]
    device_id = 0 if device.index is None else device.index
    os.makedirs(trt_cache_dir, exist_ok=True)
    return [
        (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache_dir,
                "trt_fp16_enable": True,
                "device_id": device_id,
            },
        ),
        (
            "CUDAExecutionProvider",
            {
                "device_id": device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 1 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                "enable_cuda_graph": False,
            },
        ),
        "CPUExecutionProvider",
    ]


def load_tensorrt_engine(engine_path: str, device: torch.device) -> tuple[object, object]:
    """加载 TensorRT 引擎并创建执行上下文。

    功能描述：
    读取 ``engine_path`` 的二进制内容，反序列化为 TensorRT 引擎并创建执行上下文；
    当 ``device`` 为 CUDA 时同时设置当前 CUDA 设备。

    参数说明：
    - engine_path (str): TensorRT 引擎文件路径。
    - device (torch.device): 推理设备。

    返回值说明：
    - tuple[object, object]: ``(engine, context)`` 二元组。

    可能抛出的异常：
    - FileNotFoundError: 当引擎文件不存在时触发。
    - Exception: 当 tensorrt 不可用或反序列化失败时由依赖触发。

    使用示例：
    >>> import torch
    >>> from evaluation.modelopt.loaders import load_tensorrt_engine
    >>> _ = load_tensorrt_engine("model.engine", device=torch.device("cuda:0"))  # doctest: +SKIP
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    return engine, context


def get_io_info(model_type: str, model: object) -> tuple[str | None, str | None]:
    """获取不同模型后端的输入/输出张量名称。

    功能描述：
    - pytorch：返回 ``(None, None)``（PyTorch 直接张量调用无需名称）；
    - onnx：读取 ``InferenceSession.get_inputs/get_outputs`` 的第 0 个名称；
    - tensorrt：优先使用 ``engine.get_tensor_name``，否则回退到 ``engine.get_binding_name``。

    参数说明：
    - model_type (str): 模型类型标识（``pytorch/onnx/tensorrt``）。
    - model (object): 模型对象或 ``(engine, context)`` 二元组。

    返回值说明：
    - tuple[str | None, str | None]: ``(input_name, output_name)``。

    可能抛出的异常：
    - AttributeError: 当 TensorRT 引擎不支持名称查询时触发。
    - ValueError: 当 ``model_type`` 不受支持时触发。

    使用示例：
    >>> from evaluation.modelopt.loaders import get_io_info
    >>> get_io_info("pytorch", object())
    (None, None)
    """
    if model_type == "pytorch":
        return None, None
    if model_type == "onnx":
        input_details = model.get_inputs()[0]
        output_details = model.get_outputs()[0]
        return input_details.name, output_details.name
    if model_type == "tensorrt":
        engine, _ = model
        if hasattr(engine, "get_tensor_name"):
            return engine.get_tensor_name(0), engine.get_tensor_name(1)
        if hasattr(engine, "get_binding_name"):
            return engine.get_binding_name(0), engine.get_binding_name(1)
        raise AttributeError("无法获取TensorRT引擎的输入/输出名称")
    raise ValueError(f"未知模型类型: {model_type}")
