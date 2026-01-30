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
    fp32_model_path: str
    int8_model_path: str
    dataset_path: str
    results_root: str


@dataclass(frozen=True)
class TorchInt8EvalConfig:
    batch_size: int
    test_sample_count: int
    num_classes: int
    num_workers: int
    num_warmup_batches: int
    random_seed: int
    device: str


def load_fp32_model(fp32_model_path: str, num_classes: int) -> torch.nn.Module:
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
    model = torch.jit.load(int8_model_path, map_location="cpu")
    model.eval()
    return model


def _evaluate_model(
    model: torch.nn.Module | torch.jit.ScriptModule,
    data_loader: DataLoader,
    model_name: str,
    batch_size: int,
    num_warmup_batches: int,
    device: str,
) -> dict:
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

