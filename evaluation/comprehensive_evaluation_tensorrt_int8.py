import os
import sys
import warnings

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env_checks import ensure_conda_env

ensure_conda_env("torch241_cu118_py310", "TensorRT/INT8评估")

warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from evaluation.pipelines.tensorrt_int8 import TensorRTEvalConfig, TensorRTEvalPaths, run


def main() -> str:
    fp32_model_path = os.path.join(project_root, "models/trained/resnet50_imagenette_best_8031.pth")
    tensorrt_model_path = os.path.join(
        project_root, "models/quantized/int8/resnet50_imagenette_trt_int8_x86.engine"
    )
    dataset_path = os.path.join(project_root, "data_set/imagenette")
    results_root = os.path.join(project_root, "results/TensorRT")

    config = TensorRTEvalConfig(batch_size=8, test_sample_count=3000, num_classes=10, num_workers=47)
    paths = TensorRTEvalPaths(
        fp32_model_path=fp32_model_path,
        tensorrt_model_path=tensorrt_model_path,
        dataset_path=dataset_path,
        results_root=results_root,
    )
    return run(config=config, paths=paths)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT INT8模型评估脚本")
    _ = parser.parse_args()

    result_dir = main()
    print(f"所有结果已保存到: {result_dir}")
