from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from utils.eval_metrics import compute_classification_metrics
from utils.gpu_memory import clear_torch_cuda_cache, get_gpu_memory_usage_mb

from .loaders import get_io_info


def _infer_batch(model_type: str, model: object, images: torch.Tensor, device: torch.device, num_classes: int):
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

