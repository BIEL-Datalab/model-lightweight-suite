"""ResNet/TensorRT 工具链配置结构定义。

本模块以 dataclass 的形式定义校准图像准备与 TensorRT PTQ 流程所需的参数集合，
用于命令行入口与工作流函数之间的结构化参数传递。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibImagesConfig:
    """校准图像准备参数配置。

    功能描述：
    定义从 ImageFolder 目录抽样并生成 TensorRT INT8 校准图像所需的输入输出路径与采样/尺寸参数。

    参数说明：
    - input_dir (str): 源图像目录。
    - output_dir (str): 校准图像输出目录。
    - num_images (int): 抽样图片数量。
    - target_h (int): 目标高度。
    - target_w (int): 目标宽度。
    - seed (int): 随机种子。

    返回值说明：
    - CalibImagesConfig: 配置对象。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from resnet.config import CalibImagesConfig
    >>> _ = CalibImagesConfig("data_set/imagenette/train", "data_set/calib_images", 2000, 224, 224, 42)
    """
    input_dir: str
    output_dir: str
    num_images: int
    target_h: int
    target_w: int
    seed: int


@dataclass(frozen=True)
class TensorRTPTQConfig:
    """TensorRT PTQ（可选 INT8）流程参数配置。

    功能描述：
    定义从 PyTorch 权重转换到 WTS、构建/保存 TensorRT 引擎与推理验证所需的全部参数。

    参数说明：
    - cuda_id (int): CUDA 设备索引。
    - pth_path (str): PyTorch 权重文件路径。
    - wts_path (str): WTS 输出路径。
    - weight_path (str): 引擎构建所用权重路径（通常为 WTS）。
    - engine_path (str): TensorRT 引擎文件路径。
    - batch_size (int): 构建引擎时使用的 batch 维度大小。
    - input_h (int): 输入高度。
    - input_w (int): 输入宽度。
    - output_size (int): 输出类别数。
    - input_blob_name (str): 输入张量名称。
    - output_blob_name (str): 输出张量名称。
    - eps (float): BatchNorm 数值稳定项。
    - use_int8 (bool): 是否启用 INT8（启用时需要校准数据）。
    - calib_dir (str): 校准图片目录。
    - calib_batch_size (int): 校准 batch 大小。
    - calib_dataset_size (int): 校准数据集大小上限。
    - skip_convert (bool): 是否跳过 pth->wts 转换步骤。
    - serialize (bool): 是否执行引擎构建并保存。
    - deserialize (bool): 是否执行引擎加载并做推理验证。

    返回值说明：
    - TensorRTPTQConfig: 配置对象。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from resnet.config import TensorRTPTQConfig
    >>> _ = TensorRTPTQConfig(
    ...     cuda_id=0,
    ...     pth_path="m.pth",
    ...     wts_path="m.wts",
    ...     weight_path="m.wts",
    ...     engine_path="m.engine",
    ...     batch_size=1,
    ...     input_h=224,
    ...     input_w=224,
    ...     output_size=10,
    ...     input_blob_name="data",
    ...     output_blob_name="prob",
    ...     eps=1e-5,
    ...     use_int8=True,
    ...     calib_dir="data_set/calib_images",
    ...     calib_batch_size=8,
    ...     calib_dataset_size=2000,
    ...     skip_convert=False,
    ...     serialize=True,
    ...     deserialize=False,
    ... )
    """
    cuda_id: int
    pth_path: str
    wts_path: str
    weight_path: str
    engine_path: str
    batch_size: int
    input_h: int
    input_w: int
    output_size: int
    input_blob_name: str
    output_blob_name: str
    eps: float
    use_int8: bool
    calib_dir: str
    calib_batch_size: int
    calib_dataset_size: int
    skip_convert: bool
    serialize: bool
    deserialize: bool
