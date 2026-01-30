from __future__ import annotations

import glob
import os
import random
from collections import defaultdict

from PIL import Image


def prepare_calib_images(
    input_dir: str,
    output_dir: str,
    num_images: int,
    target_size: tuple[int, int],
    shuffle_seed: int,
) -> None:
    random.seed(shuffle_seed)

    supported_formats = (".JPEG", ".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    all_images: list[str] = []
    for ext in supported_formats:
        all_images.extend(glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True))

    if not all_images:
        raise FileNotFoundError(f"在目录 {input_dir} 中未找到支持的图像文件")

    class_stats: dict[str, int] = defaultdict(int)
    for img_path in all_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        class_stats[class_name] += 1

    if len(all_images) < num_images:
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, num_images)

    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    for i, img_path in enumerate(selected_images):
        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                output_path = os.path.join(output_dir, f"calib_{i:04d}.jpg")
                img_resized.save(output_path, quality=95, optimize=True)
                success_count += 1
        except Exception:
            continue

    if success_count == 0:
        raise RuntimeError("未能成功生成任何校准图像")

