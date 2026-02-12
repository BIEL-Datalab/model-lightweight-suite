"""ModelOpt 评估流水线封装。

本模块将数据集子集构建、不同后端模型加载、精度/性能评估、报告落盘与可视化生成串联起来，
对外提供一个统一的 ``run_benchmark`` 入口函数。
"""

from __future__ import annotations

import os

import numpy as np
import torch

from utils.eval_artifacts import create_timestamped_dir, validate_file_path, write_json, write_text
from utils.eval_dataset import build_imagenette_val_subset_loader
from utils.visualization_utils import generate_visualizations

from .config import ModelOptEvalConfig, ModelOptEvalPaths
from .evaluator import evaluate_accuracy, evaluate_performance
from .loaders import (
    build_onnx_providers,
    build_onnx_quantized_providers,
    load_onnx_session,
    load_pytorch_resnet50,
    load_tensorrt_engine,
)
from .reporting import build_json_report, build_markdown_report


def run_benchmark(config: ModelOptEvalConfig, paths: ModelOptEvalPaths, device: torch.device, enable_visualization: bool) -> str:
    """运行基于 ModelOpt 的 INT8 评估基准并输出结果目录。

    功能描述：
    该函数按固定流程完成：
    1) 校验模型/数据集路径；
    2) 构建可复现的验证集子集 DataLoader；
    3) 加载 PyTorch/ONNX/TensorRT 模型；
    4) 执行精度评估与性能评估；
    5) 生成 JSON 与 Markdown 报告并写入结果目录；
    6) 可选地生成可视化图表。

    参数说明：
    - config (ModelOptEvalConfig): 评估参数配置。
    - paths (ModelOptEvalPaths): 评估路径配置。
    - device (torch.device): 推理设备。
    - enable_visualization (bool): 是否生成可视化图表。

    返回值说明：
    - str: 结果目录路径（包含 JSON/Markdown/图表等文件）。

    可能抛出的异常：
    - FileNotFoundError: 当模型或数据集路径不存在时由 ``validate_file_path`` 触发。
    - Exception: 当模型加载、推理或写文件失败时由底层依赖触发。

    使用示例：
    >>> import torch
    >>> from evaluation.modelopt.config import ModelOptEvalConfig, ModelOptEvalPaths
    >>> from evaluation.modelopt.pipeline import run_benchmark
    >>> _ = run_benchmark(  # doctest: +SKIP
    ...     config=ModelOptEvalConfig(
    ...         batch_size=1,
    ...         num_classes=10,
    ...         num_workers=0,
    ...         test_sample_count=10,
    ...         random_seed=42,
    ...         num_warmup_batches=0,
    ...         num_warmup=0,
    ...         num_iterations=1,
    ...     ),
    ...     paths=ModelOptEvalPaths(
    ...         pytorch_model_path="m.pth",
    ...         onnx_original_path="m.onnx",
    ...         onnx_quantized_path="m_int8.onnx",
    ...         tensorrt_engine_path="m.engine",
    ...         dataset_path="data_set/imagenette",
    ...         results_root="results",
    ...     ),
    ...     device=torch.device("cpu"),
    ...     enable_visualization=False,
    ... )
    """
    model_paths = {
        "pytorch": paths.pytorch_model_path,
        "onnx_original": paths.onnx_original_path,
        "onnx_quantized": paths.onnx_quantized_path,
        "tensorrt": paths.tensorrt_engine_path,
    }
    for name, path in model_paths.items():
        validate_file_path(path, f"{name}模型")
    validate_file_path(paths.dataset_path, "数据集")

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    result_folder = create_timestamped_dir(paths.results_root, "evaluation_result_int8")
    trt_cache_dir = os.path.join(result_folder, "trt_cache")

    subset = build_imagenette_val_subset_loader(
        dataset_root=paths.dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sample_count=config.test_sample_count,
        seed=config.random_seed,
        pin_memory=False,
    )

    pytorch_model = load_pytorch_resnet50(paths.pytorch_model_path, config.num_classes, device=device)
    onnx_original = load_onnx_session(
        paths.onnx_original_path, device=device, providers=build_onnx_providers(device)
    )
    onnx_quantized = load_onnx_session(
        paths.onnx_quantized_path, device=device, providers=build_onnx_quantized_providers(device, trt_cache_dir)
    )
    tensorrt_engine, tensorrt_context = load_tensorrt_engine(paths.tensorrt_engine_path, device=device)

    results: dict[str, dict[str, object]] = {
        "pytorch": {},
        "onnx_original": {},
        "onnx_quantized": {},
        "tensorrt": {},
    }

    results["pytorch"].update(
        evaluate_accuracy(
            "pytorch",
            pytorch_model,
            subset.data_loader,
            "PyTorch",
            device=device,
            batch_size=config.batch_size,
            num_warmup_batches=config.num_warmup_batches,
            num_classes=config.num_classes,
        )
    )
    results["onnx_original"].update(
        evaluate_accuracy(
            "onnx",
            onnx_original,
            subset.data_loader,
            "ONNX原始",
            device=device,
            batch_size=config.batch_size,
            num_warmup_batches=config.num_warmup_batches,
            num_classes=config.num_classes,
        )
    )
    results["onnx_quantized"].update(
        evaluate_accuracy(
            "onnx",
            onnx_quantized,
            subset.data_loader,
            "ONNX量化",
            device=device,
            batch_size=config.batch_size,
            num_warmup_batches=config.num_warmup_batches,
            num_classes=config.num_classes,
        )
    )
    results["tensorrt"].update(
        evaluate_accuracy(
            "tensorrt",
            (tensorrt_engine, tensorrt_context),
            subset.data_loader,
            "TensorRT",
            device=device,
            batch_size=config.batch_size,
            num_warmup_batches=config.num_warmup_batches,
            num_classes=config.num_classes,
        )
    )

    results["pytorch"].update(
        evaluate_performance(
            "pytorch",
            pytorch_model,
            subset.data_loader,
            "PyTorch",
            device=device,
            batch_size=config.batch_size,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations,
            num_classes=config.num_classes,
        )
    )
    results["onnx_original"].update(
        evaluate_performance(
            "onnx",
            onnx_original,
            subset.data_loader,
            "ONNX原始",
            device=device,
            batch_size=config.batch_size,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations,
            num_classes=config.num_classes,
        )
    )
    results["onnx_quantized"].update(
        evaluate_performance(
            "onnx",
            onnx_quantized,
            subset.data_loader,
            "ONNX量化",
            device=device,
            batch_size=config.batch_size,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations,
            num_classes=config.num_classes,
        )
    )
    results["tensorrt"].update(
        evaluate_performance(
            "tensorrt",
            (tensorrt_engine, tensorrt_context),
            subset.data_loader,
            "TensorRT",
            device=device,
            batch_size=config.batch_size,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations,
            num_classes=config.num_classes,
        )
    )

    json_path = os.path.join(result_folder, "evaluation_results.json")
    md_path = os.path.join(result_folder, "benchmark_report.md")

    test_config = {
        "batch_size": config.batch_size,
        "test_sample_count": len(results["pytorch"]["labels"]),
        "num_classes": config.num_classes,
        "num_workers": config.num_workers,
        "device": str(device),
        "random_seed": config.random_seed,
        "num_warmup_batches": config.num_warmup_batches,
    }

    write_json(json_path, build_json_report(results, model_paths=model_paths, test_config=test_config))
    write_text(md_path, build_markdown_report(results, model_paths=model_paths, device_display=str(device), batch_size=config.batch_size))

    if enable_visualization:
        viz_results: dict[str, dict[str, object]] = {}
        for model_key, model_type in [("pytorch", "fp32"), ("tensorrt", "int8")]:
            result = dict(results[model_key])
            if "avg_inference_time" in result:
                result["avg_inference_time_s"] = result["avg_inference_time"]
            if "throughput" in result:
                result["avg_throughput_fps"] = result["throughput"]
            if "avg_cpu_memory_mb" not in result:
                result["avg_cpu_memory_mb"] = 0
            viz_results[model_type] = result
        generate_visualizations(viz_results, result_folder)

    return result_folder
