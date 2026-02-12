"""数据加载相关工具。

本模块封装了基于 torchvision.datasets.ImageFolder 的数据集构建与 DataLoader 创建逻辑，
用于训练、验证以及量化校准（calibration）等流程的统一数据输入。
"""

import torch
import os  
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset  
import copy
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_data_loaders(
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 47,
    device: "torch.device | None" = None,
) -> "tuple[DataLoader[object], DataLoader[object], DataLoader[object], Dataset[object], Dataset[object]]":
    """创建训练/验证/校准数据加载器。

    功能描述：
    从指定的数据根目录创建训练集、验证集与校准集的数据加载器（DataLoader），并返回对应的数据集对象。

    参数说明：
    - data_root (str): 数据集根目录路径，目录下应包含 train 与 val 子目录（ImageFolder 结构）。
    - batch_size (int): 批次大小。
    - num_workers (int): DataLoader 的工作线程数。
    - device (torch.device | None): 目标设备。当前实现中未使用该参数，仅作为调用侧配置占位。

    返回值说明：
    - tuple[DataLoader[object], DataLoader[object], DataLoader[object], Dataset[object], Dataset[object]]:
      依次为训练集 DataLoader、验证集 DataLoader、校准集 DataLoader、训练集 Dataset、验证集 Dataset。

    可能抛出的异常：
    - FileNotFoundError: 当 data_root/train 或 data_root/val 不存在时，由 ImageFolder 触发。
    - RuntimeError: 当数据解码/读取失败或 DataLoader 多进程加载异常时，由底层依赖触发。

    使用示例：
    >>> from utils.data_loader import get_data_loaders
    >>> # 需要存在 ImageFolder 结构的数据集目录；此示例仅展示调用方式
    >>> _ = get_data_loaders("path/to/dataset", batch_size=32)  # doctest: +SKIP
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.ImageFolder(f"{data_root}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_root}/val", transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    cal_dataset = datasets.ImageFolder(f"{data_root}/val", transform=val_transform)
    calibration_loader = DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, calibration_loader, train_dataset, val_dataset
