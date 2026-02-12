"""ModelOpt INT8 评估综合脚本入口。

本脚本用于在固定环境与默认路径配置下运行 ModelOpt INT8 评估流水线，并将结果落盘到 results 目录。
"""

import os
import sys
import warnings

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env_checks import ensure_conda_env

ensure_conda_env("modelopt_env", "ModelOpt/INT8评估")

warnings.filterwarnings("ignore")

from evaluation.modelopt.config import ModelOptEvalConfig, ModelOptEvalPaths
from evaluation.modelopt.pipeline import run_benchmark


def main(device: torch.device, enable_visualization: bool = False) -> str:
    """运行 ModelOpt INT8 评估并返回结果目录。

    功能描述：
    按本仓库约定的默认路径构建配置与路径对象，调用 ``evaluation.modelopt.pipeline.run_benchmark`` 执行评估，
    并返回结果目录路径。

    参数说明：
    - device (torch.device): 推理设备。
    - enable_visualization (bool): 是否启用评估结果可视化输出。

    返回值说明：
    - str: 结果目录路径。

    可能抛出的异常：
    - FileNotFoundError: 当默认模型/数据集路径不存在时触发。
    - Exception: 当评估流程内部依赖执行失败时触发。

    使用示例：
    >>> import torch
    >>> from evaluation.comprehensive_evaluation_modelopt_int8 import main
    >>> _ = main(device=torch.device("cpu"), enable_visualization=False)  # doctest: +SKIP
    """
    paths = ModelOptEvalPaths(
        pytorch_model_path=os.path.join(project_root, "models/trained/resnet50_imagenette_best_8031.pth"),
        onnx_original_path=os.path.join(project_root, "models/converted/resnet50_imagenette_modelopt_x86.onnx"),
        onnx_quantized_path=os.path.join(project_root, "models/quantized/int8/resnet50_imagenette_modelopt_int8_x86.onnx"),
        tensorrt_engine_path=os.path.join(
            project_root, "models/quantized/int8/resnet50_imagenette_modelopt_int8_x86.engine"
        ),
        dataset_path=os.path.join(project_root, "data_set/imagenette"),
        results_root=os.path.join(project_root, "results/ModelOpt"),
    )
    config = ModelOptEvalConfig(
        batch_size=16,
        num_classes=10,
        num_workers=47,
        test_sample_count=3000,
        random_seed=42,
        num_warmup_batches=10,
        num_warmup=10,
        num_iterations=200,
    )
    return run_benchmark(config=config, paths=paths, device=device, enable_visualization=enable_visualization)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ModelOpt INT8模型评估脚本")
    parser.add_argument("--device", type=str, default="cuda", help="指定设备，例如: cpu, cuda, cuda:0, cuda:1等")
    parser.add_argument("--visualization", action="store_true", help="启用评估结果可视化")
    args = parser.parse_args()

    device = torch.device(args.device)
    result_dir = main(device=device, enable_visualization=args.visualization)
    print(f"所有结果已保存到: {result_dir}")
