from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def load_pytorch_resnet50(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
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
    import onnxruntime as ort

    return ort.InferenceSession(model_path, providers=providers)


def build_onnx_providers(device: torch.device) -> list[Any]:
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


def load_tensorrt_engine(engine_path: str, device: torch.device):
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

