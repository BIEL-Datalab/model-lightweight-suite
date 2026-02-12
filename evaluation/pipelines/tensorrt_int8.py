"""TensorRT INT8 模型评估流水线。

本模块对比评估：
- PyTorch FP32 模型（权重文件）
- TensorRT INT8 引擎（engine 文件）

输出 JSON/Markdown 报告。
"""

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
    """TensorRT INT8 评估所需路径配置。

    功能描述：
    聚合 FP32 权重路径、TensorRT 引擎路径、数据集路径与结果输出根目录。

    参数说明：
    - fp32_model_path (str): FP32 权重文件路径。
    - tensorrt_model_path (str): TensorRT 引擎文件路径。
    - dataset_path (str): 数据集根目录路径。
    - results_root (str): 结果输出根目录。

    返回值说明：
    - TensorRTEvalPaths: 配置对象。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from evaluation.pipelines.tensorrt_int8 import TensorRTEvalPaths
    >>> _ = TensorRTEvalPaths("fp32.pth", "model.engine", "data_set/imagenette", "results")
    """
    fp32_model_path: str
    tensorrt_model_path: str
    dataset_path: str
    results_root: str


@dataclass(frozen=True)
class TensorRTEvalConfig:
    """TensorRT INT8 评估参数配置。

    功能描述：
    定义评估所需的批次大小、抽样样本数、类别数与 DataLoader 线程数。

    参数说明：
    - batch_size (int): 推理批次大小。
    - test_sample_count (int): 抽样样本数上限。
    - num_classes (int): 分类类别数。
    - num_workers (int): DataLoader 工作线程数。

    返回值说明：
    - TensorRTEvalConfig: 配置对象。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from evaluation.pipelines.tensorrt_int8 import TensorRTEvalConfig
    >>> _ = TensorRTEvalConfig(batch_size=32, test_sample_count=100, num_classes=10, num_workers=4)
    """
    batch_size: int
    test_sample_count: int
    num_classes: int
    num_workers: int


def _load_fp32_model(fp32_model_path: str, num_classes: int) -> torch.nn.Module:
    """加载 FP32 ResNet50 模型（内部函数）。

    功能描述：
    构建不带预训练权重的 ResNet50，根据 ``num_classes`` 重建最后一层全连接，并从 ``fp32_model_path`` 加载权重。
    兼容 checkpoint 既可能是 state_dict，也可能是包含 ``state_dict`` 键的字典。

    参数说明：
    - fp32_model_path (str): FP32 权重文件路径。
    - num_classes (int): 分类类别数。

    返回值说明：
    - torch.nn.Module: 处于 eval 模式的模型实例（在 CPU 上）。

    可能抛出的异常：
    - FileNotFoundError: 当权重文件不存在时触发。
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> from evaluation.pipelines.tensorrt_int8 import _load_fp32_model
    >>> _ = _load_fp32_model("fp32.pth", num_classes=10)  # doctest: +SKIP
    """
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
    """加载 TensorRT 引擎并返回引擎/上下文与 I/O 名称（内部函数）。

    功能描述：
    读取 ``engine_path`` 的 engine 二进制内容，通过 TensorRT Runtime 反序列化得到引擎，
    并创建执行上下文；同时提取第 0/1 个 tensor 的名称作为输入/输出名称返回。

    参数说明：
    - engine_path (str): TensorRT 引擎文件路径（``.engine``）。

    返回值说明：
    - tuple: ``(engine, context, input_name, output_name)``。

    可能抛出的异常：
    - FileNotFoundError: 当引擎文件不存在时触发。
    - RuntimeError: 当反序列化失败或上下文创建失败时由 TensorRT 触发。

    使用示例：
    >>> from evaluation.pipelines.tensorrt_int8 import _load_tensorrt_engine
    >>> _ = _load_tensorrt_engine("model.engine")  # doctest: +SKIP
    """
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
    """评估 FP32 模型的准确率、平均推理时间与吞吐量（内部函数）。

    功能描述：
    对数据加载器中的样本逐条执行推理，统计 top-1 准确率；同时记录每次推理耗时并计算平均推理时延与吞吐量。
    为尽量贴近单样本推理，本实现对每个 batch 仅取第一个样本进行推理（``images[0:1]``）。

    参数说明：
    - model (torch.nn.Module): FP32 PyTorch 模型。
    - data_loader: 验证集 DataLoader（迭代产生 ``(images, labels)``）。
    - device (torch.device): 推理设备。

    返回值说明：
    - tuple[float, float, float]: ``(accuracy, avg_time_ms, throughput_fps)``。

    可能抛出的异常：
    - RuntimeError: 当模型推理或张量迁移失败时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from evaluation.pipelines.tensorrt_int8 import _evaluate_fp32_model
    >>> _ = _evaluate_fp32_model(torch.nn.Identity(), object(), device=torch.device("cpu"))  # doctest: +SKIP
    """
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
    """评估 TensorRT 引擎的准确率、平均推理时间与吞吐量（内部函数）。

    功能描述：
    在 CUDA 可用时，将输入/输出 tensor 分配在 GPU 上，通过 ``context.execute_v2`` 执行推理；
    对每个 batch 仅取第一个样本进行推理，统计 top-1 准确率，并计算平均推理时延与吞吐量。
    当 CUDA 不可用时返回 ``(0.0, 0.0, 0.0)``。

    参数说明：
    - engine: TensorRT engine 对象（当前实现中不直接使用）。
    - context: TensorRT execution context。
    - data_loader: 验证集 DataLoader（迭代产生 ``(images, labels)``）。
    - num_classes (int): 分类类别数（用于输出张量形状）。

    返回值说明：
    - tuple[float, float, float]: ``(accuracy, avg_time_ms, throughput_fps)``。

    可能抛出的异常：
    - RuntimeError: 当 CUDA 张量分配或 TensorRT 执行失败时触发。

    使用示例：
    >>> from evaluation.pipelines.tensorrt_int8 import _evaluate_tensorrt_model
    >>> _ = _evaluate_tensorrt_model(object(), object(), object(), num_classes=10)  # doctest: +SKIP
    """
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
    """构建 Markdown 报告文本（内部函数）。

    功能描述：
    将 FP32 与 TensorRT 的评估结果格式化为 Markdown 文本，包含模型路径、评估配置、结果表格与性能对比结论。

    参数说明：
    - paths (TensorRTEvalPaths): 路径配置。
    - config (TensorRTEvalConfig): 参数配置。
    - fp32: FP32 评估结果三元组 ``(accuracy, avg_time_ms, throughput_fps)``。
    - trt: TensorRT 评估结果三元组 ``(accuracy, avg_time_ms, throughput_fps)``。
    - actual_test_count (int): 实际测试样本数量（可能小于配置的 sample_count）。

    返回值说明：
    - str: Markdown 文本内容。

    可能抛出的异常：
    - Exception: 当输入结果结构不符合预期导致格式化失败时触发。

    使用示例：
    >>> from evaluation.pipelines.tensorrt_int8 import _build_markdown_report, TensorRTEvalConfig, TensorRTEvalPaths
    >>> _ = _build_markdown_report(  # doctest: +SKIP
    ...     paths=TensorRTEvalPaths("fp32.pth", "model.engine", "data_set/imagenette", "results"),
    ...     config=TensorRTEvalConfig(batch_size=1, test_sample_count=10, num_classes=10, num_workers=0),
    ...     fp32=(0.8, 10.0, 100.0),
    ...     trt=(0.79, 5.0, 200.0),
    ...     actual_test_count=10,
    ... )
    """
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
    """执行 TensorRT INT8 评估并返回结果目录。

    功能描述：
    校验输入路径后，构建验证集子集 DataLoader，评估 FP32 PyTorch 模型与 TensorRT 引擎，
    最终在带时间戳的结果目录下写入 JSON 与 Markdown 报告。

    参数说明：
    - config (TensorRTEvalConfig): 评估参数配置。
    - paths (TensorRTEvalPaths): 评估路径配置。

    返回值说明：
    - str: 结果目录路径。

    可能抛出的异常：
    - FileNotFoundError: 当模型或数据集路径不存在时触发。
    - Exception: 当引擎加载、评估或写文件失败时由底层依赖触发。

    使用示例：
    >>> from evaluation.pipelines.tensorrt_int8 import run, TensorRTEvalConfig, TensorRTEvalPaths
    >>> _ = run(  # doctest: +SKIP
    ...     config=TensorRTEvalConfig(batch_size=1, test_sample_count=10, num_classes=10, num_workers=0),
    ...     paths=TensorRTEvalPaths(fp32_model_path="fp32.pth", tensorrt_model_path="model.engine", dataset_path="data_set/imagenette", results_root="results"),
    ... )
    """
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
