"""校准图片准备工具函数。

本模块提供从 ImageFolder 风格目录抽样并统一尺寸保存图片的能力，
用于 TensorRT INT8 量化校准阶段的输入数据准备。
"""

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
    """从输入目录抽样并生成校准图片集合。

    功能描述：
    递归搜集 ``input_dir`` 下的常见图片文件，按 ``shuffle_seed`` 固定随机性后抽样 ``num_images`` 张，
    将图片转换为 RGB 并 resize 到 ``target_size``，保存到 ``output_dir``。
    当无法生成任何图片时抛出 ``RuntimeError``。

    参数说明：
    - input_dir (str): 源图像目录（包含类别子目录）。
    - output_dir (str): 输出目录（将写入 ``calib_XXXX.jpg``）。
    - num_images (int): 抽样图片数量。
    - target_size (tuple[int, int]): 目标尺寸（宽, 高）。
    - shuffle_seed (int): 随机种子。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - FileNotFoundError: 当未找到任何支持格式的图片文件时触发。
    - RuntimeError: 当所有图片处理失败导致无法生成任何输出时触发。
    - OSError: 当图片读取或写入失败时由 PIL/文件系统触发。

    使用示例：
    >>> from resnet.calib.prepare_images import prepare_calib_images
    >>> prepare_calib_images("data_set/imagenette/train", "data_set/calib_images", 10, (224, 224), 42)  # doctest: +SKIP
    """
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
