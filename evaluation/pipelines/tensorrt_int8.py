from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np
import torch
import torchvision.models as models
from sklearn.metrics import accuracy_score

from utils.eval_artifacts import create_timestamped_dir, validate_file_path, write_json, write_text
from utils.eval_dataset import build_imagenette_val_subset_loader


@dataclass(frozen=True)
class TensorRTEvalPaths:
    fp32_model_path: str
    tensorrt_model_path: str
    dataset_path: str
    results_root: str


@dataclass(frozen=True)
class TensorRTEvalConfig:
    batch_size: int
    test_sample_count: int
    num_classes: int
    num_workers: int


def _load_fp32_model(fp32_model_path: str, num_classes: int) -> torch.nn.Module:
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(fp32_model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def _load_tensorrt_engine(engine_path: str):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    return engine, context, input_name, output_name


def _evaluate_fp32_model(model: torch.nn.Module, data_loader, device: torch.device) -> tuple[float, float, float]:
    model = model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    inference_times: list[float] = []

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= 3:
                break
            image = images[0:1].float().to(device, non_blocking=True)
            model(image)
            del image

        for images, labels in data_loader:
            labels_np = labels.numpy()
            start_time = time.time()

            image = images[0:1].float().to(device, non_blocking=True)
            output = model(image)

            output_np = output.detach().cpu().numpy()
            exp_output = np.exp(output_np - np.max(output_np, axis=1, keepdims=True))
            softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            pred = int(np.argmax(softmax_output, axis=1)[0])

            end_time = time.time()
            inference_times.append(end_time - start_time)

            all_preds.append(pred)
            all_labels.append(int(labels_np[0]))
            del image, output, output_np

    acc = float(accuracy_score(all_labels, all_preds))
    avg_time_ms = float(np.mean(inference_times) * 1000) if inference_times else 0.0
    throughput = float(1 / (avg_time_ms / 1000)) if avg_time_ms > 0 else 0.0
    return acc, avg_time_ms, throughput


def _evaluate_tensorrt_model(engine, context, data_loader, num_classes: int) -> tuple[float, float, float]:
    import tensorrt as trt

    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    device = torch.device("cuda:0")
    all_preds: list[int] = []
    all_labels: list[int] = []
    inference_times: list[float] = []

    for i, (images, _) in enumerate(data_loader):
        if i >= 3:
            break
        image = images[0:1].float().to(device, non_blocking=True)
        output = torch.empty((1, num_classes), dtype=torch.float32, device=device)
        bindings = [image.data_ptr(), output.data_ptr()]
        context.execute_v2(bindings)
        del image, output

    for images, labels in data_loader:
        labels_np = labels.numpy()
        start_time = time.time()

        image = images[0:1].float().to(device, non_blocking=True)
        output = torch.empty((1, num_classes), dtype=torch.float32, device=device)
        bindings = [image.data_ptr(), output.data_ptr()]
        context.execute_v2(bindings)

        output_np = output.detach().cpu().numpy()
        exp_output = np.exp(output_np - np.max(output_np, axis=1, keepdims=True))
        softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        pred = int(np.argmax(softmax_output, axis=1)[0])

        end_time = time.time()
        inference_times.append(end_time - start_time)
        all_preds.append(pred)
        all_labels.append(int(labels_np[0]))
        del image, output, output_np

    acc = float(accuracy_score(all_labels, all_preds))
    avg_time_ms = float(np.mean(inference_times) * 1000) if inference_times else 0.0
    throughput = float(1 / (avg_time_ms / 1000)) if avg_time_ms > 0 else 0.0
    return acc, avg_time_ms, throughput


def _build_markdown_report(paths: TensorRTEvalPaths, config: TensorRTEvalConfig, fp32, trt, actual_test_count: int) -> str:
    accuracy_fp32, time_fp32, throughput_fp32 = fp32
    accuracy_trt, time_trt, throughput_trt = trt
    speedup = time_fp32 / time_trt if time_trt > 0 else 0.0
    accuracy_drop = accuracy_fp32 - accuracy_trt
    return f"""# TensorRT INT8 模型评估报告

## 模型路径

- **FP32 模型**: {paths.fp32_model_path}
- **TensorRT INT8 模型**: {paths.tensorrt_model_path}

## 评估基本信息

- **测试样本数量**: {actual_test_count}
- **批次大小**: {config.batch_size}

## 评估结果

| 模型类型 | 准确率 | 平均推理时间(ms) | 吞吐量(FPS) |
|---------|--------|------------------|------------|
| PyTorch FP32 | {accuracy_fp32:.4f} | {time_fp32:.2f} | {throughput_fp32:.1f} |
| TensorRT INT8 | {accuracy_trt:.4f} | {time_trt:.2f} | {throughput_trt:.1f} |

## 性能对比

- **加速比**: {speedup:.2f}x
- **精度下降**: {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)
"""


def run(config: TensorRTEvalConfig, paths: TensorRTEvalPaths) -> str:
    validate_file_path(paths.fp32_model_path, "FP32模型")
    validate_file_path(paths.tensorrt_model_path, "TensorRT模型")
    validate_file_path(paths.dataset_path, "数据集")

    subset = build_imagenette_val_subset_loader(
        dataset_root=paths.dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sample_count=config.test_sample_count,
        seed=42,
        pin_memory=False,
    )

    fp32_model = _load_fp32_model(paths.fp32_model_path, num_classes=config.num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fp32_results = _evaluate_fp32_model(fp32_model, subset.data_loader, device=device)

    engine, context, _, _ = _load_tensorrt_engine(paths.tensorrt_model_path)
    trt_results = _evaluate_tensorrt_model(engine, context, subset.data_loader, num_classes=config.num_classes)

    result_folder = create_timestamped_dir(paths.results_root, "evaluation_result_int8")
    json_path = os.path.join(result_folder, "evaluation_results.json")
    md_path = os.path.join(result_folder, "benchmark_report.md")

    report_json = {
        "test_config": {
            "batch_size": config.batch_size,
            "test_sample_count": subset.sample_count,
            "num_classes": config.num_classes,
            "num_workers": config.num_workers,
        },
        "fp32_model": {
            "model_path": paths.fp32_model_path,
            "accuracy": float(fp32_results[0]),
            "avg_inference_time_ms": float(fp32_results[1]),
            "throughput_fps": float(fp32_results[2]),
        },
        "tensorrt_model": {
            "model_path": paths.tensorrt_model_path,
            "accuracy": float(trt_results[0]),
            "avg_inference_time_ms": float(trt_results[1]),
            "throughput_fps": float(trt_results[2]),
        },
        "comparison": {
            "speedup": float(fp32_results[1] / trt_results[1]) if trt_results[1] > 0 else 0.0,
            "accuracy_drop": float(fp32_results[0] - trt_results[0]),
            "accuracy_drop_percent": float((fp32_results[0] - trt_results[0]) * 100),
        },
    }

    write_json(json_path, report_json)
    write_text(md_path, _build_markdown_report(paths, config, fp32_results, trt_results, subset.sample_count))
    return result_folder

