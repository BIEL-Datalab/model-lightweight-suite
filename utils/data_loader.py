import torch
import os  
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset  
import copy

def get_data_loaders(data_root: str, batch_size: int = 256, num_workers: int = 47, device: torch.device = None) -> tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset]:
    """获取数据加载器。
    
    从指定的数据根目录创建训练集、验证集和校准集的数据加载器，以及对应的数据集对象。
    
    Args:
        data_root (str): 数据集根目录路径，包含train和val子目录。
        batch_size (int, optional): 批次大小，默认256。
        num_workers (int, optional): 数据加载器工作线程数，默认47。
        device (torch.device, optional): 设备类型，默认为None。
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset]: 
            - 第1个元素：训练集数据加载器
            - 第2个元素：验证集数据加载器
            - 第3个元素：校准集数据加载器
            - 第4个元素：训练集数据集对象
            - 第5个元素：验证集数据集对象
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

