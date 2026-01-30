import os
import sys
import warnings

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env_checks import ensure_conda_env

ensure_conda_env("torch241_cu118_py310", "Torch/INT8评估")

warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from evaluation.pipelines.torch_int8 import TorchInt8EvalConfig, TorchInt8EvalPaths, run


def main(enable_visualization: bool = False) -> str:
    fp32_model_path = os.path.join(project_root, "models/trained/resnet50_imagenette_best_8031.pth")
    int8_model_path = os.path.join(project_root, "models/quantized/int8/resnet50_imagenette_torch_int8.pth")
    dataset_path = os.path.join(project_root, "data_set/imagenette")
    results_root = os.path.join(project_root, "results/Torch")

    config = TorchInt8EvalConfig(
        batch_size=32,
        test_sample_count=3000,
        num_classes=10,
        num_workers=47,
        num_warmup_batches=10,
        random_seed=42,
        device="cpu",
    )
    paths = TorchInt8EvalPaths(
        fp32_model_path=fp32_model_path,
        int8_model_path=int8_model_path,
        dataset_path=dataset_path,
        results_root=results_root,
    )
    return run(config=config, paths=paths, enable_visualization=enable_visualization)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch INT8模型评估脚本")
    parser.add_argument("--visualization", action="store_true", help="启用评估结果可视化")
    args = parser.parse_args()

    result_dir = main(enable_visualization=args.visualization)
    print(f"所有结果已保存到: {result_dir}")

