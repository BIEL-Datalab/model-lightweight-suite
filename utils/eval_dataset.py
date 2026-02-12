"""评估数据集构建工具。

本模块提供针对 ImageFolder 风格数据集（例如 ImageNette）的验证集预处理与子集抽样逻辑，
用于在评估流程中快速构建可复现的小规模验证集 DataLoader。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.datasets as datasets


def get_imagenette_val_transform() -> transforms.Compose:
    """获取 ImageNette 验证集预处理变换。

    功能描述：
    返回与常见 ResNet 系列评估一致的验证预处理（Resize/CenterCrop/Normalize）。

    参数说明：
    - 无。

    返回值说明：
    - transforms.Compose: torchvision 预处理流水线。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from utils.eval_dataset import get_imagenette_val_transform
    >>> t = get_imagenette_val_transform()
    >>> hasattr(t, "__call__")
    True
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass(frozen=True)
class ImageFolderSubset:
    """ImageFolder 子集 DataLoader 的封装结果。

    功能描述：
    该数据类将抽样得到的 DataLoader 与类别名称列表、实际抽样数量一起返回，便于上层评估流程复用。

    参数说明：
    - data_loader (DataLoader[object]): 子集数据加载器。
    - class_names (list[str]): 类别名称列表（来源于 ``ImageFolder.classes``）。
    - sample_count (int): 实际抽样样本数（不超过数据集总长度）。

    返回值说明：
    - ImageFolderSubset: 封装后的结果对象。

    可能抛出的异常：
    - 无。该类本身不执行抽样与 I/O。

    使用示例：
    >>> from utils.eval_dataset import ImageFolderSubset
    >>> s = ImageFolderSubset(data_loader=object(), class_names=["a"], sample_count=1)
    >>> s.sample_count
    1
    """
    data_loader: DataLoader[object]
    class_names: list[str]
    sample_count: int


def build_imagenette_val_subset_loader(
    dataset_root: str,
    batch_size: int,
    num_workers: int,
    sample_count: int,
    seed: int,
    pin_memory: bool = False,
) -> ImageFolderSubset:
    """构建 ImageNette 验证集的随机子集 DataLoader。

    功能描述：
    从 ``{dataset_root}/val`` 加载 ImageFolder 数据集，按 ``seed`` 固定随机种子抽样 ``sample_count`` 个样本，
    并返回不打乱（shuffle=False）的 DataLoader。

    参数说明：
    - dataset_root (str): 数据集根目录路径。
    - batch_size (int): DataLoader 批次大小。
    - num_workers (int): DataLoader 工作线程数。
    - sample_count (int): 期望抽样样本数；若大于数据集长度则使用数据集全量。
    - seed (int): 随机种子，用于可复现的抽样结果。
    - pin_memory (bool): 是否启用 pin_memory，默认 False。

    返回值说明：
    - ImageFolderSubset: 包含 DataLoader、类别名与实际抽样数量的封装结果。

    可能抛出的异常：
    - FileNotFoundError: 当 ``{dataset_root}/val`` 不存在时由 ImageFolder 触发。
    - ValueError: 当 ``sample_count`` 非法导致抽样失败时由 NumPy 触发。

    使用示例：
    >>> from utils.eval_dataset import build_imagenette_val_subset_loader
    >>> _ = build_imagenette_val_subset_loader(  # doctest: +SKIP
    ...     dataset_root="path/to/imagenette",
    ...     batch_size=32,
    ...     num_workers=4,
    ...     sample_count=100,
    ...     seed=42,
    ... )
    """
    val_transform = get_imagenette_val_transform()
    val_dataset = datasets.ImageFolder(f"{dataset_root}/val", transform=val_transform)

    actual_count = min(sample_count, len(val_dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(val_dataset), actual_count, replace=False)
    subset = Subset(val_dataset, indices.tolist())
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return ImageFolderSubset(data_loader=loader, class_names=val_dataset.classes, sample_count=actual_count)

