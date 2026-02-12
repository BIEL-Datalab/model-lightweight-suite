"""Torch INT8 模型评估流水线。

本模块对比评估：
- FP32 PyTorch 模型（权重文件）
- INT8 TorchScript 模型（torch.jit.load）

输出 JSON/Markdown 报告，并可选生成可视化图表。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from utils.eval_artifacts import create_timestamped_dir, validate_file_path, write_json, write_text
from utils.eval_dataset import build_imagenette_val_subset_loader
from utils.eval_metrics import compute_classification_metrics


@dataclass(frozen=True)
class TorchInt8EvalPaths:
    """Torch INT8 评估所需路径配置。

    功能描述：
    聚合 FP32/INT8 模型路径、数据集路径与结果输出根目录。

    参数说明：
    - fp32_model_path (str): FP32 权重文件路径。
    - int8_model_path (str): INT8 TorchScript 模型路径。
    - dataset_path (str): 数据集根目录路径（ImageFolder 风格）。
    - results_root (str): 结果输出根目录。

    返回值说明：
    - TorchInt8EvalPaths: 配置对象。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from evaluation.pipelines.torch_int8 import TorchInt8EvalPaths
    >>> _ = TorchInt8EvalPaths("fp32.pth", "int8.pt", "data_set/imagenette", "results")
    """
    fp32_model_path: str
    int8_model_path: str
    dataset_path: str
    results_root: str


@dataclass(frozen=True)
class TorchInt8EvalConfig:
    """Torch INT8 评估参数配置。

    功能描述：
    定义评估所需的批次大小、抽样样本数、类别数、线程数、预热批次数、随机种子与设备配置。

    参数说明：
    - batch_size (int): 推理批次大小。
    - test_sample_count (int): 抽样样本数上限。
    - num_classes (int): 分类类别数。
    - num_workers (int): DataLoader 工作线程数。
    - num_warmup_batches (int): 预热 batch 数。
    - random_seed (int): 随机种子。
    - device (str): 推理设备字符串（例如 ``'cpu'`` 或 ``'cuda:0'``）。

    返回值说明：
    - TorchInt8EvalConfig: 配置对象。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from evaluation.pipelines.torch_int8 import TorchInt8EvalConfig
    >>> _ = TorchInt8EvalConfig(batch_size=32, test_sample_count=100, num_classes=10, num_workers=4, num_warmup_batches=1, random_seed=42, device="cpu")
    """
    batch_size: int
    test_sample_count: int
    num_classes: int
    num_workers: int
    num_warmup_batches: int
    random_seed: int
    device: str


def load_fp32_model(fp32_model_path: str, num_classes: int) -> torch.nn.Module:
    """加载 FP32 ResNet50 模型并载入权重。

    功能描述：
    构建不带预训练权重的 ResNet50，根据 ``num_classes`` 重建全连接层，并从 ``fp32_model_path`` 加载权重。
    兼容 checkpoint 中既可能是 state_dict，也可能是包含 ``state_dict`` 键的字典。

    参数说明：
    - fp32_model_path (str): FP32 权重文件路径。
    - num_classes (int): 分类类别数。

    返回值说明：
    - torch.nn.Module: 处于 eval 模式的模型实例。

    可能抛出的异常：
    - FileNotFoundError: 当权重文件不存在时触发。
    - RuntimeError: 当权重加载或结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> from evaluation.pipelines.torch_int8 import load_fp32_model
    >>> _ = load_fp32_model("fp32.pth", num_classes=10)  # doctest: +SKIP
    """
    model = torchvision.models.resnet.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    checkpoint = torch.load(fp32_model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def load_int8_model(int8_model_path: str) -> torch.jit.ScriptModule:
    """加载 INT8 TorchScript 模型。

    功能描述：
    使用 ``torch.jit.load`` 从 ``int8_model_path`` 加载 TorchScript 模型，并切换到 eval 模式。

    参数说明：
    - int8_model_path (str): TorchScript 模型路径。

    返回值说明：
    - torch.jit.ScriptModule: TorchScript 模型对象。

    可能抛出的异常：
    - FileNotFoundError: 当模型文件不存在时触发。
    - RuntimeError: 当 TorchScript 反序列化失败时由 PyTorch 触发。

    使用示例：
    >>> from evaluation.pipelines.torch_int8 import load_int8_model
    >>> _ = load_int8_model("int8.pt")  # doctest: +SKIP
    """
    model = torch.jit.load(int8_model_path, map_location="cpu")
    model.eval()
    return model


def _evaluate_model(
    model: torch.nn.Module | torch.jit.ScriptModule,
    data_loader: DataLoader[object],
    model_name: str,
    batch_size: int,
    num_warmup_batches: int,
    device: str,
) -> dict:
    """评估单个模型的精度、性能与资源占用（内部函数）。

    功能描述：
    对给定模型执行预热推理与完整评估推理，统计分类指标、平均/方差推理时间、吞吐量、
    以及 GPU/CPU 内存占用变化，并返回结构化结果字典。

    参数说明：
    - model (torch.nn.Module | torch.jit.ScriptModule): 待评估模型。
    - data_loader (DataLoader[object]): 验证集数据加载器。
    - model_name (str): 模型名称（用于日志展示）。
    - batch_size (int): 推理批次大小（用于吞吐量计算）。
    - num_warmup_batches (int): 预热 batch 数。
    - device (str): 推理设备字符串。

    返回值说明：
    - dict: 评估结果字典（包含指标、时间、资源占用与预测/标签序列）。

    可能抛出的异常：
    - RuntimeError: 当推理执行失败时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from evaluation.pipelines.torch_int8 import _evaluate_model
    >>> _ = _evaluate_model(torch.nn.Identity(), data_loader=object(), model_name="dummy", batch_size=1, num_warmup_batches=0, device="cpu")  # doctest: +SKIP
    """
    eval_device = torch.device(device)
    try:
        model = model.to(eval_device)
    except Exception:
        eval_device = torch.device("cpu")
        model = model.to(eval_device)

    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    inference_times: list[float] = []
    gpu_memory_usages: list[float] = []
    cpu_memory_usages: list[float] = []

    initial_cpu_memory = psutil.Process().memory_info().rss / 1024**2

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_warmup_batches:
                break
            images = images.to(eval_device, non_blocking=True)
            _ = model(images)
            if eval_device.type == "cuda":
                torch.cuda.synchronize()
            del images

        for images, labels in data_loader:
            images = images.to(eval_device, non_blocking=True)
            labels_np = labels.numpy()

            current_cpu_memory = psutil.Process().memory_info().rss / 1024**2
            cpu_memory_usages.append(current_cpu_memory - initial_cpu_memory)

            if eval_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.max_memory_allocated() / 1024**2

            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            if eval_device.type == "cuda":
                torch.cuda.synchronize()
                end_memory = torch.cuda.max_memory_allocated() / 1024**2
                gpu_memory_usages.append(end_memory - start_memory)

            inference_times.append(end_time - start_time)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels_np.tolist())
            del images, outputs

    metrics = compute_classification_metrics(all_labels, all_preds)
    avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0
    std_inference_time = float(np.std(inference_times)) if inference_times else 0.0
    avg_throughput = float(batch_size / avg_inference_time) if avg_inference_time > 0 else 0.0
    avg_gpu_memory = float(np.mean(gpu_memory_usages)) if gpu_memory_usages else 0.0
    avg_cpu_memory = float(np.mean(cpu_memory_usages)) if cpu_memory_usages else 0.0

    result = {
        **metrics.as_dict(),
        "avg_inference_time_s": avg_inference_time,
        "std_inference_time_s": std_inference_time,
        "avg_throughput_fps": avg_throughput,
        "avg_gpu_memory_mb": avg_gpu_memory,
        "avg_cpu_memory_mb": avg_cpu_memory,
        "predictions": all_preds,
        "labels": all_labels,
    }
    print(
        f"\n{model_name} 评估结果: accuracy={result['accuracy']:.4f}, "
        f"avg_time={avg_inference_time*1000:.2f}ms, throughput={avg_throughput:.1f}FPS"
    )
    return result


def _build_json_report(config: TorchInt8EvalConfig, paths: TorchInt8EvalPaths, results: dict) -> dict:
    """构建 JSON 报告字典（内部函数）。

    功能描述：
    将 FP32 与 INT8 两个模型的评估结果整理为便于落盘与后续分析的 JSON 结构，
    并计算加速比、吞吐提升、显存变化、准确率下降等对比指标。

    参数说明：
    - config (TorchInt8EvalConfig): 评估参数配置。
    - paths (TorchInt8EvalPaths): 模型与数据集路径配置。
    - results (dict): 评估结果字典，需包含 ``results['fp32']`` 与 ``results['int8']`` 两个条目。

    返回值说明：
    - dict: JSON 可序列化的报告字典。

    可能抛出的异常：
    - KeyError: 当 ``results`` 缺少必要键时触发。
    - TypeError: 当结果值类型不符合预期导致计算失败时触发。

    使用示例：
    >>> from evaluation.pipelines.torch_int8 import _build_json_report, TorchInt8EvalConfig, TorchInt8EvalPaths
    >>> _ = _build_json_report(  # doctest: +SKIP
    ...     config=TorchInt8EvalConfig(batch_size=1, test_sample_count=10, num_classes=10, num_workers=0, num_warmup_batches=0, random_seed=42, device="cpu"),
    ...     paths=TorchInt8EvalPaths(fp32_model_path="fp32.pth", int8_model_path="int8.pt", dataset_path="data_set/imagenette", results_root="results"),
    ...     results={"fp32": {"labels":[0], "avg_inference_time_s": 0.01, "avg_throughput_fps": 100.0, "avg_gpu_memory_mb": 0.0, "avg_cpu_memory_mb": 0.0, "accuracy": 0.8}, "int8": {"labels":[0], "avg_inference_time_s": 0.005, "avg_throughput_fps": 200.0, "avg_gpu_memory_mb": 0.0, "avg_cpu_memory_mb": 0.0, "accuracy": 0.79}},
    ... )
    """
    fp32_time = results["fp32"]["avg_inference_time_s"]
    int8_time = results["int8"]["avg_inference_time_s"]
    fp32_throughput = results["fp32"]["avg_throughput_fps"]
    int8_throughput = results["int8"]["avg_throughput_fps"]
    fp32_memory_gpu = results["fp32"]["avg_gpu_memory_mb"]
    int8_memory_gpu = results["int8"]["avg_gpu_memory_mb"]
    fp32_memory_cpu = results["fp32"]["avg_cpu_memory_mb"]
    int8_memory_cpu = results["int8"]["avg_cpu_memory_mb"]

    speedup = fp32_time / int8_time if int8_time > 0 else 0.0
    throughput_improvement = (
        (int8_throughput - fp32_throughput) / fp32_throughput * 100 if fp32_throughput > 0 else 0.0
    )
    gpu_memory_reduction = (
        (fp32_memory_gpu - int8_memory_gpu) / fp32_memory_gpu * 100 if fp32_memory_gpu > 0 else 0.0
    )
    cpu_memory_change = (
        (int8_memory_cpu - fp32_memory_cpu) / fp32_memory_cpu * 100 if fp32_memory_cpu > 0 else 0.0
    )

    return {
        "test_config": {
            "batch_size": config.batch_size,
            "test_sample_count": len(results["fp32"]["labels"]),
            "num_classes": config.num_classes,
            "num_workers": config.num_workers,
            "device": config.device,
            "random_seed": config.random_seed,
            "num_warmup_batches": config.num_warmup_batches,
        },
        "model_paths": {
            "fp32": paths.fp32_model_path,
            "int8": paths.int8_model_path,
        },
        "results": {
            "fp32": {k: results["fp32"][k] for k in results["fp32"] if k not in {"predictions", "labels"}},
            "int8": {k: results["int8"][k] for k in results["int8"] if k not in {"predictions", "labels"}},
        },
        "comparison": {
            "speedup": speedup,
            "throughput_improvement_percent": throughput_improvement,
            "gpu_memory_reduction_percent": gpu_memory_reduction,
            "cpu_memory_change_percent": cpu_memory_change,
            "accuracy_drop": results["fp32"]["accuracy"] - results["int8"]["accuracy"],
        },
    }


def _build_markdown_report(config: TorchInt8EvalConfig, paths: TorchInt8EvalPaths, results: dict) -> str:
    """构建 Markdown 报告文本（内部函数）。

    功能描述：
    将 FP32 与 INT8 的评估结果格式化为 Markdown 文本，包含模型路径、配置摘要、结果表格与对比结论。

    参数说明：
    - config (TorchInt8EvalConfig): 评估参数配置。
    - paths (TorchInt8EvalPaths): 路径配置。
    - results (dict): 评估结果字典，需包含 ``fp32`` 与 ``int8`` 两个条目。

    返回值说明：
    - str: Markdown 文本内容。

    可能抛出的异常：
    - KeyError: 当 ``results`` 缺少必要键时触发。

    使用示例：
    >>> from evaluation.pipelines.torch_int8 import _build_markdown_report, TorchInt8EvalConfig, TorchInt8EvalPaths
    >>> _ = _build_markdown_report(  # doctest: +SKIP
    ...     config=TorchInt8EvalConfig(batch_size=1, test_sample_count=10, num_classes=10, num_workers=0, num_warmup_batches=0, random_seed=42, device="cpu"),
    ...     paths=TorchInt8EvalPaths(fp32_model_path="fp32.pth", int8_model_path="int8.pt", dataset_path="data_set/imagenette", results_root="results"),
    ...     results={"fp32": {"labels":[0], "accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8, "avg_inference_time_s": 0.01, "avg_throughput_fps": 100.0, "avg_gpu_memory_mb": 0.0, "avg_cpu_memory_mb": 0.0},
    ...              "int8": {"labels":[0], "accuracy": 0.79, "precision": 0.79, "recall": 0.79, "f1_score": 0.79, "avg_inference_time_s": 0.005, "avg_throughput_fps": 200.0, "avg_gpu_memory_mb": 0.0, "avg_cpu_memory_mb": 0.0}},
    ... )
    """
    fp32_time = results["fp32"]["avg_inference_time_s"]
    int8_time = results["int8"]["avg_inference_time_s"]
    fp32_throughput = results["fp32"]["avg_throughput_fps"]
    int8_throughput = results["int8"]["avg_throughput_fps"]
    speedup = fp32_time / int8_time if int8_time > 0 else 0.0
    throughput_improvement = (
        (int8_throughput - fp32_throughput) / fp32_throughput * 100 if fp32_throughput > 0 else 0.0
    )
    accuracy_drop = results["fp32"]["accuracy"] - results["int8"]["accuracy"]

    return f"""# PyTorch INT8 模型评估报告

## 模型路径

- **FP32 模型**: {paths.fp32_model_path}
- **INT8 模型**: {paths.int8_model_path}

## 评估配置

- **测试样本数量**: {len(results['fp32']['labels'])}
- **批次大小**: {config.batch_size}
- **设备**: {config.device}

## 详细结果

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | 平均推理时间(s) | 吞吐量(FPS) | 平均GPU内存(MB) | 平均CPU内存(MB) |
|------|--------|--------|--------|--------|------------------|-------------|-------------------|-------------------|
| FP32 | {results['fp32']['accuracy']:.4f} | {results['fp32']['precision']:.4f} | {results['fp32']['recall']:.4f} | {results['fp32']['f1_score']:.4f} | {results['fp32']['avg_inference_time_s']:.4f} | {results['fp32']['avg_throughput_fps']:.2f} | {results['fp32']['avg_gpu_memory_mb']:.2f} | {results['fp32']['avg_cpu_memory_mb']:.2f} |
| INT8 | {results['int8']['accuracy']:.4f} | {results['int8']['precision']:.4f} | {results['int8']['recall']:.4f} | {results['int8']['f1_score']:.4f} | {results['int8']['avg_inference_time_s']:.4f} | {results['int8']['avg_throughput_fps']:.2f} | {results['int8']['avg_gpu_memory_mb']:.2f} | {results['int8']['avg_cpu_memory_mb']:.2f} |

## 对比结论 (INT8 vs FP32)

- **加速比**: {speedup:.2f}x
- **吞吐量提升**: {throughput_improvement:+.2f}%
- **准确率下降**: {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)
"""


def run(config: TorchInt8EvalConfig, paths: TorchInt8EvalPaths, enable_visualization: bool) -> str:
    """执行 Torch INT8 评估并返回结果目录。

    功能描述：
    校验输入路径后，构建验证集子集 DataLoader，加载 FP32/INT8 模型并评估，
    最终在带时间戳的结果目录下写入 JSON 与 Markdown 报告，并可选生成可视化图表。

    参数说明：
    - config (TorchInt8EvalConfig): 评估参数配置。
    - paths (TorchInt8EvalPaths): 评估路径配置。
    - enable_visualization (bool): 是否生成可视化图表。

    返回值说明：
    - str: 结果目录路径。

    可能抛出的异常：
    - FileNotFoundError: 当模型或数据集路径不存在时触发。
    - Exception: 当模型加载、评估或写文件失败时由底层依赖触发。

    使用示例：
    >>> from evaluation.pipelines.torch_int8 import run, TorchInt8EvalConfig, TorchInt8EvalPaths
    >>> _ = run(  # doctest: +SKIP
    ...     config=TorchInt8EvalConfig(batch_size=1, test_sample_count=10, num_classes=10, num_workers=0, num_warmup_batches=0, random_seed=42, device="cpu"),
    ...     paths=TorchInt8EvalPaths(fp32_model_path="fp32.pth", int8_model_path="int8.pt", dataset_path="data_set/imagenette", results_root="results"),
    ...     enable_visualization=False,
    ... )
    """
    validate_file_path(paths.fp32_model_path, "FP32模型")
    validate_file_path(paths.int8_model_path, "INT8模型")
    validate_file_path(paths.dataset_path, "数据集")

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    subset = build_imagenette_val_subset_loader(
        dataset_root=paths.dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sample_count=config.test_sample_count,
        seed=config.random_seed,
        pin_memory=False,
    )

    fp32_model = load_fp32_model(paths.fp32_model_path, num_classes=config.num_classes)
    int8_model = load_int8_model(paths.int8_model_path)

    results: dict = {}
    results["fp32"] = _evaluate_model(
        fp32_model,
        subset.data_loader,
        "FP32",
        batch_size=config.batch_size,
        num_warmup_batches=config.num_warmup_batches,
        device=config.device,
    )
    results["int8"] = _evaluate_model(
        int8_model,
        subset.data_loader,
        "INT8",
        batch_size=config.batch_size,
        num_warmup_batches=config.num_warmup_batches,
        device=config.device,
    )

    result_folder = create_timestamped_dir(paths.results_root, "evaluation_result_int8")
    json_path = os.path.join(result_folder, "evaluation_results.json")
    md_path = os.path.join(result_folder, "benchmark_report.md")

    write_json(json_path, _build_json_report(config, paths, results))
    write_text(md_path, _build_markdown_report(config, paths, results))

    if enable_visualization:
        from utils.visualization_utils import generate_visualizations

        generate_visualizations(results, result_folder)

    return result_folder

