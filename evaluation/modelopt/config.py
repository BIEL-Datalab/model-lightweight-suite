"""ModelOpt INT8 评估配置结构定义。

本模块以 dataclass 的形式定义评估流程所需的路径集合与参数集合，便于在命令行脚本/流水线中统一传递。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOptEvalPaths:
    """ModelOpt 评估相关路径配置。

    功能描述：
    聚合评估流程需要使用的模型文件、数据集路径与结果输出根目录路径。

    参数说明：
    - pytorch_model_path (str): PyTorch 权重文件路径。
    - onnx_original_path (str): 原始 ONNX 模型路径。
    - onnx_quantized_path (str): 量化后 ONNX 模型路径。
    - tensorrt_engine_path (str): TensorRT 引擎文件路径。
    - dataset_path (str): 数据集根目录路径（ImageFolder 风格）。
    - results_root (str): 结果输出根目录。

    返回值说明：
    - ModelOptEvalPaths: 配置对象。

    可能抛出的异常：
    - 无。该类本身不执行 I/O 校验。

    使用示例：
    >>> from evaluation.modelopt.config import ModelOptEvalPaths
    >>> _ = ModelOptEvalPaths(
    ...     pytorch_model_path="m.pth",
    ...     onnx_original_path="m.onnx",
    ...     onnx_quantized_path="m_int8.onnx",
    ...     tensorrt_engine_path="m.engine",
    ...     dataset_path="data_set/imagenette",
    ...     results_root="results",
    ... )
    """
    pytorch_model_path: str
    onnx_original_path: str
    onnx_quantized_path: str
    tensorrt_engine_path: str
    dataset_path: str
    results_root: str


@dataclass(frozen=True)
class ModelOptEvalConfig:
    """ModelOpt 评估参数配置。

    功能描述：
    聚合评估过程中与批次、抽样、预热与迭代次数相关的参数，确保不同模型类型对比使用一致设置。

    参数说明：
    - batch_size (int): 推理批次大小。
    - num_classes (int): 分类类别数。
    - num_workers (int): DataLoader 工作线程数。
    - test_sample_count (int): 评估抽样样本数上限。
    - random_seed (int): 随机种子。
    - num_warmup_batches (int): 精度评估阶段的预热 batch 数。
    - num_warmup (int): 性能评估阶段的预热迭代数。
    - num_iterations (int): 性能评估阶段的测试迭代数。

    返回值说明：
    - ModelOptEvalConfig: 配置对象。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from evaluation.modelopt.config import ModelOptEvalConfig
    >>> _ = ModelOptEvalConfig(
    ...     batch_size=32,
    ...     num_classes=10,
    ...     num_workers=4,
    ...     test_sample_count=100,
    ...     random_seed=42,
    ...     num_warmup_batches=5,
    ...     num_warmup=5,
    ...     num_iterations=10,
    ... )
    """
    batch_size: int
    num_classes: int
    num_workers: int
    test_sample_count: int
    random_seed: int
    num_warmup_batches: int
    num_warmup: int
    num_iterations: int
