import sys
import os
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.data_loader import get_data_loaders

def prepare_calibration_data(data_root: str, total_samples: int = 500, save_path: str = 'data_set/calibration_data_500.npy') -> np.ndarray:
    """准备校准数据。
    
    从验证集中均匀采样各类别的样本，预处理后保存为numpy数组，用于模型量化校准。
    
    Args:
        data_root (str): 数据集根目录路径。
        total_samples (int, optional): 总采样数量，默认500。
        save_path (str, optional): 校准数据保存路径，默认'data_set/calibration_data_500.npy'。
    
    Returns:
        np.ndarray: 预处理后的校准数据，形状为(samples, 3, 224, 224)。
    """
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = datasets.ImageFolder(f"{data_root}/val", transform=val_transform)
    num_classes = len(val_dataset.classes)
    
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(val_dataset):
        class_indices[label].append(idx)
    
    samples_per_class = total_samples // num_classes
    selected_indices = []
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        selected = np.random.choice(indices, min(samples_per_class, len(indices)), replace=False)
        selected_indices.extend(selected.tolist())
    
    calibration_subset = Subset(val_dataset, selected_indices)
    
    calibration_data = []
    for i in range(len(calibration_subset)):
        image, _ = calibration_subset[i]
        calibration_data.append(image.numpy())
    
    calibration_data = np.array(calibration_data)
    np.save(save_path, calibration_data)
    return calibration_data

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