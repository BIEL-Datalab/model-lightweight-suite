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

    results: dict = {
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
        viz_results: dict = {}
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

