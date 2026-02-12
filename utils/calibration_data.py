"""校准数据准备脚本与工具函数。

本模块从验证集（val）中按类别均匀抽样，生成用于量化校准（calibration）的输入张量集合并保存为 ``.npy`` 文件。
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.data_loader import get_data_loaders

def prepare_calibration_data(
    data_root: str,
    total_samples: int = 500,
    save_path: str = 'data_set/calibration_data_500.npy',
) -> "np.typing.NDArray[np.float32]":
    """生成并保存量化校准所需的输入数据数组。

    功能描述：
    从验证集（``{data_root}/val``）中按类别均匀采样，将样本经过与验证一致的预处理（Resize/CenterCrop/Normalize），
    最终将 ``(N, 3, 224, 224)`` 的 NumPy 数组保存到 ``save_path`` 并返回。

    参数说明：
    - data_root (str): 数据集根目录路径。
    - total_samples (int): 总采样数量（将按类别平均分配并向下取整）。
    - save_path (str): 校准数据保存路径（``.npy``）。

    返回值说明：
    - np.typing.NDArray[np.float32]: 预处理后的校准数据数组，形状为 ``(samples, 3, 224, 224)``。

    可能抛出的异常：
    - FileNotFoundError: 当 ``{data_root}/val`` 不存在时，由 ImageFolder 触发。
    - OSError: 当无法写入 ``save_path`` 时由 ``numpy.save``（即 ``np.save``）触发。
    - ValueError: 当数据集类别数为 0 或采样参数不合法导致采样失败时触发（由 NumPy 触发）。

    使用示例：
    >>> from utils.calibration_data import prepare_calibration_data
    >>> # 需要存在 ImageFolder 结构的数据集目录；此示例仅展示调用方式
    >>> _ = prepare_calibration_data("path/to/dataset", total_samples=10, save_path="calib.npy")  # doctest: +SKIP
    """
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = datasets.ImageFolder(f"{data_root}/val", transform=val_transform)
    num_classes = len(val_dataset.classes)
    
    class_indices: dict[int, list[int]] = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(val_dataset):
        class_indices[label].append(idx)
    
    samples_per_class = total_samples // num_classes
    selected_indices = []
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        selected = np.random.choice(indices, min(samples_per_class, len(indices)), replace=False)
        selected_indices.extend(selected.tolist())
    
    calibration_subset = Subset(val_dataset, selected_indices)
    
    calibration_data: list["np.typing.NDArray[np.float32]"] = []
    for i in range(len(calibration_subset)):
        image, _ = calibration_subset[i]
        calibration_data.append(image.numpy())
    
    calibration_data = np.array(calibration_data)  # type: ignore[assignment]
    np.save(save_path, calibration_data)
    return calibration_data  # type: ignore[return-value]

if __name__ == "__main__":
    dataset_path = 'data_set/imagenette'
    total_samples = 500 
    output_file = 'data_set/calibration_data_' + f'{total_samples}.npy'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    calibration_data = prepare_calibration_data(
        data_root=dataset_path,
        total_samples=total_samples,
        save_path=output_file
    )
