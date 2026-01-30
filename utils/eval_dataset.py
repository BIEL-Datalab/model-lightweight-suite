from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.datasets as datasets


def get_imagenette_val_transform() -> transforms.Compose:
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
    data_loader: DataLoader
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

