"""模型推理评估器。

本模块提供面向多种模型载体（PyTorch / ONNX Runtime / TensorRT）的统一评估逻辑，
包括精度评估与性能评估，并输出结构化结果字典以供报告与可视化模块消费。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from utils.eval_metrics import compute_classification_metrics
from utils.gpu_memory import clear_torch_cuda_cache, get_gpu_memory_usage_mb

from .loaders import get_io_info


def _infer_batch(
    model_type: str, model: object, images: torch.Tensor, device: torch.device, num_classes: int
) -> "np.ndarray":
    """对一个 batch 执行推理并返回预测类别。

    功能描述：
    根据 ``model_type`` 选择对应的推理后端：
    - ``pytorch``：对模型做前向并取 argmax
    - ``onnx``：调用 ``InferenceSession.run`` 并取 argmax
    - ``tensorrt``：通过执行上下文执行并取 argmax

    参数说明：
    - model_type (str): 模型类型标识，支持 ``'pytorch'``、``'onnx'``、``'tensorrt'``。
    - model (object): 模型对象。不同类型对应不同结构：
      - pytorch: ``nn.Module`` 风格
      - onnx: ``onnxruntime.InferenceSession``
      - tensorrt: ``(engine, context)`` 二元组
    - images (torch.Tensor): 输入图片 batch 张量。
    - device (torch.device): 目标设备。
    - num_classes (int): 类别数（用于 TensorRT 输出 buffer 形状）。

    返回值说明：
    - np.ndarray: 预测类别数组，形状为 ``(N,)``，元素为 int。

    可能抛出的异常：
    - ValueError: 当 ``model_type`` 不受支持时触发。
    - Exception: 当底层推理后端执行失败时由依赖触发。
    """
    if model_type == "pytorch":
        images_dev = images.to(device)
        with torch.no_grad():
            outputs = model(images_dev)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        return preds

    input_name, output_name = get_io_info(model_type, model)

    if model_type == "onnx":
        images_np = images.numpy()
        outputs = model.run([output_name], {input_name: images_np})
        return np.argmax(outputs[0], axis=1)

    if model_type == "tensorrt":
        import tensorrt as trt

        engine, context = model
        images_cuda = images.to(device)
        context.set_input_shape(input_name, images_cuda.shape)
        output_buffer = torch.empty((images_cuda.shape[0], num_classes), dtype=torch.float32, device=device)

        bindings = [None] * engine.num_io_tensors
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                bindings[i] = images_cuda.data_ptr()
            else:
                bindings[i] = output_buffer.data_ptr()

        context.execute_v2(bindings)
        return np.argmax(output_buffer.detach().cpu().numpy(), axis=1)

    raise ValueError(f"未知模型类型: {model_type}")


def _forward_only(model_type: str, model: object, images: torch.Tensor, device: torch.device, num_classes: int) -> None:
    """仅执行前向推理（不返回结果），用于预热或性能计时。

    功能描述：
    根据 ``model_type`` 选择推理后端执行前向推理，以触发内核加载、图优化或缓存构建等“预热”行为。

    参数说明：
    - model_type (str): 模型类型标识。
    - model (object): 模型对象（约定同 ``_infer_batch``）。
    - images (torch.Tensor): 输入图片 batch 张量。
    - device (torch.device): 目标设备。
    - num_classes (int): 类别数（用于 TensorRT 输出 buffer 形状）。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - ValueError: 当 ``model_type`` 不受支持时触发。
    - Exception: 当底层推理后端执行失败时由依赖触发。
    """
    if model_type == "pytorch":
        images_dev = images.to(device)
        with torch.no_grad():
            _ = model(images_dev)
        return

    input_name, output_name = get_io_info(model_type, model)

    if model_type == "onnx":
        images_np = images.numpy()
        _ = model.run([output_name], {input_name: images_np})
        return

    if model_type == "tensorrt":
        import tensorrt as trt

        engine, context = model
        images_cuda = images.to(device)
        context.set_input_shape(input_name, images_cuda.shape)
        output_buffer = torch.empty((images_cuda.shape[0], num_classes), dtype=torch.float32, device=device)

        bindings = [None] * engine.num_io_tensors
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                bindings[i] = images_cuda.data_ptr()
            else:
                bindings[i] = output_buffer.data_ptr()
        context.execute_v2(bindings)
        return

    raise ValueError(f"未知模型类型: {model_type}")


def evaluate_accuracy(
    model_type: str,
    model: object,
    data_loader: Any,
    model_name: str,
    device: torch.device,
    batch_size: int,
    num_warmup_batches: int,
    num_classes: int,
) -> dict:
    """评估模型精度并返回指标与原始预测数据。

    功能描述：
    先执行 ``num_warmup_batches`` 个 batch 的预热前向推理，然后在 ``data_loader`` 上完成推理，
    统计分类指标与平均推理时间/吞吐量，并返回包含预测与标签序列的结果字典。

    参数说明：
    - model_type (str): 模型类型标识（``pytorch/onnx/tensorrt``）。
    - model (object): 模型对象（约定同 ``_infer_batch``）。
    - data_loader (Any): 迭代得到 ``(images, labels)`` 的数据加载器。
    - model_name (str): 模型名称（用于日志展示）。
    - device (torch.device): 推理设备。
    - batch_size (int): 推理批次大小（用于吞吐量计算）。
    - num_warmup_batches (int): 预热 batch 数。
    - num_classes (int): 类别数。

    返回值说明：
    - dict: 结果字典，包含分类指标、平均推理时间（秒）、吞吐量（FPS）、以及 ``predictions``/``labels`` 列表。

    可能抛出的异常：
    - Exception: 当数据加载或底层推理失败时由依赖触发。

    使用示例：
    >>> from evaluation.modelopt.evaluator import evaluate_accuracy
    >>> _ = evaluate_accuracy("pytorch", object(), object(), "PyTorch", device=object(), batch_size=1, num_warmup_batches=0, num_classes=10)  # doctest: +SKIP
    """
    print(f"\n评估 {model_name} 模型精度...")

    clear_torch_cuda_cache()
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    for i, (images, _) in enumerate(tqdm(data_loader, desc=f"{model_name} 预热", leave=False)):
        if i >= num_warmup_batches:
            break
        _forward_only(model_type, model, images, device=device, num_classes=num_classes)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    clear_torch_cuda_cache()

    all_preds: list[int] = []
    all_labels: list[int] = []
    inference_times: list[float] = []

    for images, labels in tqdm(data_loader, desc=f"{model_name} 精度评估", leave=False):
        labels_np = labels.numpy()
        start_time = time.time()
        preds = _infer_batch(model_type, model, images, device=device, num_classes=num_classes)
        end_time = time.time()

        inference_times.append(end_time - start_time)
        all_preds.extend([int(x) for x in preds])
        all_labels.extend([int(x) for x in labels_np])

    metrics = compute_classification_metrics(all_labels, all_preds)
    avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0
    throughput = float(batch_size / avg_inference_time) if avg_inference_time > 0 else 0.0

    return {
        **metrics.as_dict(),
        "avg_inference_time": avg_inference_time,
        "throughput": throughput,
        "predictions": all_preds,
        "labels": all_labels,
    }


def evaluate_performance(
    model_type: str,
    model: object,
    data_loader: Any,
    model_name: str,
    device: torch.device,
    batch_size: int,
    num_warmup: int,
    num_iterations: int,
    num_classes: int,
) -> dict:
    """评估模型性能并返回计时与资源占用信息。

    功能描述：
    先执行 ``num_warmup`` 次预热前向推理，再执行 ``num_iterations`` 次计时推理，
    统计平均/标准差推理耗时（ms）、吞吐量（FPS）以及平均 GPU 显存占用（MB）。

    参数说明：
    - model_type (str): 模型类型标识（``pytorch/onnx/tensorrt``）。
    - model (object): 模型对象（约定同 ``_infer_batch``）。
    - data_loader (Any): 迭代得到 ``(images, labels)`` 的数据加载器。
    - model_name (str): 模型名称（用于日志展示）。
    - device (torch.device): 推理设备。
    - batch_size (int): 推理批次大小（用于吞吐量计算）。
    - num_warmup (int): 预热迭代数。
    - num_iterations (int): 测试迭代数上限。
    - num_classes (int): 类别数。

    返回值说明：
    - dict: 结果字典，包含 ``avg_inference_time_ms``、``std_inference_time_ms``、``throughput``、
      ``avg_gpu_memory_mb`` 与 ``inference_times`` 列表等。

    可能抛出的异常：
    - Exception: 当数据加载或底层推理失败时由依赖触发。

    使用示例：
    >>> from evaluation.modelopt.evaluator import evaluate_performance
    >>> _ = evaluate_performance("pytorch", object(), object(), "PyTorch", device=object(), batch_size=1, num_warmup=0, num_iterations=1, num_classes=10)  # doctest: +SKIP
    """
    print(f"\n评估 {model_name} 模型性能...")

    clear_torch_cuda_cache()
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    for i, (images, _) in enumerate(tqdm(data_loader, desc=f"{model_name} 性能预热", leave=False)):
        if i >= num_warmup:
            break
        _forward_only(model_type, model, images, device=device, num_classes=num_classes)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    clear_torch_cuda_cache()

    inference_times: list[float] = []
    gpu_memory_usages: list[float] = []

    for i, (images, _) in enumerate(tqdm(data_loader, desc=f"{model_name} 性能测试", leave=False)):
        if i >= num_iterations:
            break

        if device.type == "cuda":
            gpu_memory_usages.append(get_gpu_memory_usage_mb(0 if device.index is None else device.index))

        start_time = time.time()
        _forward_only(model_type, model, images, device=device, num_classes=num_classes)
        end_time = time.time()
        inference_times.append(end_time - start_time)

        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    avg_time_ms = float(np.mean(inference_times) * 1000) if inference_times else 0.0
    std_time_ms = float(np.std(inference_times) * 1000) if inference_times else 0.0
    throughput = float(batch_size / (avg_time_ms / 1000)) if avg_time_ms > 0 else 0.0
    avg_gpu_memory = float(np.mean(gpu_memory_usages)) if gpu_memory_usages else 0.0

    return {
        "avg_inference_time_ms": avg_time_ms,
        "std_inference_time_ms": std_time_ms,
        "throughput": throughput,
        "avg_gpu_memory_mb": avg_gpu_memory,
        "inference_times": inference_times,
    }
