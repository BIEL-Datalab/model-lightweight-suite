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

