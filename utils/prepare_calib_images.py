#!/usr/bin/env python3

import os
import random
import glob
from PIL import Image
import argparse
from collections import defaultdict

def prepare_calib_images(input_dir: str, output_dir: str, num_images: int = 200, target_size: tuple = (224, 224), shuffle_seed: int = 42) -> None:
    """准备TensorRT INT8量化的校准图像。
    
    从输入目录中随机选择指定数量的图像，调整大小后保存到输出目录，用于TensorRT INT8量化校准。
    
    Args:
        input_dir (str): 源图像目录，支持ImageNet格式（包含类别子目录）。
        output_dir (str): 校准图像保存目录。
        num_images (int, optional): 选择的图像数量，默认200。
        target_size (tuple, optional): 目标图像尺寸，默认(224, 224)。
        shuffle_seed (int, optional): 随机种子，用于可复现的结果，默认42。
    
    Returns:
        None
    """
    random.seed(shuffle_seed)
    
    supported_formats = ('.JPEG', '.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    all_images = []
    for ext in supported_formats:
        all_images.extend(glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True))
    
    if not all_images:
        print(f"在目录 {input_dir} 中未找到支持的图像文件")
        return
    
    class_stats = defaultdict(int)
    for img_path in all_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        class_stats[class_name] += 1
    
    print(f"发现 {len(class_stats)} 个类别")
    print("每个类别图像数量：")
    for class_name, count in sorted(class_stats.items()):
        print(f"  {class_name}: {count} 张")
    
    if len(all_images) < num_images:
        print(f"警告：源目录中只有 {len(all_images)} 张图像，将使用所有图像")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, num_images)
    
    selected_class_stats = defaultdict(int)
    for img_path in selected_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        selected_class_stats[class_name] += 1
    
    print(f"\n选中的 {num_images} 张图像类别分布：")
    for class_name, count in sorted(selected_class_stats.items()):
        print(f"  {class_name}: {count} 张")
    
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for i, img_path in enumerate(selected_images):
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                output_path = os.path.join(output_dir, f"calib_{i:04d}.jpg")
                img_resized.save(output_path, quality=95, optimize=True)
                
                success_count += 1
                if (i + 1) % 50 == 0 or i + 1 == len(selected_images):
                    print(f"已处理 {i + 1}/{len(selected_images)} 张图像")
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            continue
    
    print(f"\n校准图像准备完成！")
    print(f"源图像总数: {len(all_images)}")
    print(f"选中的图像数量: {len(selected_images)}")
    print(f"成功处理的图像数量: {success_count}")
    print(f"校准图像保存目录: {output_dir}")
    print(f"目标图像尺寸: {target_size[0]}x{target_size[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备TensorRT INT8量化的校准图像")
    parser.add_argument("--input-dir", default="data_set/imagenette/train", help="源图像目录（支持ImageNet格式，包含类别子目录）")
    parser.add_argument("--output-dir", default="data_set/calib_images", help="校准图像保存目录")
    parser.add_argument("--num-images", type=int, default=2000, help="选择的图像数量")
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224], help="目标图像尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于可复现的结果")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TensorRT INT8 校准图像准备脚本")
    print("=" * 60)
    print(f"源目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标数量: {args.num_images} 张")
    print(f"目标尺寸: {args.target_size[0]}x{args.target_size[1]}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    prepare_calib_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        target_size=tuple(args.target_size),
        shuffle_seed=args.seed
    )
