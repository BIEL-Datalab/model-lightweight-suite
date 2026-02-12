"""TensorRT INT8 校准器构建工具。

本模块提供校准图片预处理函数与校准器工厂函数 ``create_int8_calibrator``，
用于在 TensorRT 构建阶段提供 INT8 校准数据 batch。
"""

from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

def _preprocess_image(image_path: str, input_h: int, input_w: int) -> np.ndarray:
    """读取并预处理单张图片为模型输入格式。

    功能描述：
    将图片读取为 RGB，Resize 到 256x256 后做中心裁剪到 (input_w, input_h)，
    转为 CHW 布局并按 ImageNet 归一化（mean/std）。

    参数说明：
    - image_path (str): 图片路径。
    - input_h (int): 目标高度。
    - input_w (int): 目标宽度。

    返回值说明：
    - np.ndarray: 预处理后的 float32 数组，形状为 ``(3, input_h, input_w)``。

    可能抛出的异常：
    - OSError: 当图片无法读取或解码失败时由 PIL 触发。

    使用示例：
    >>> from resnet.trt.calibrator import _preprocess_image
    >>> _ = _preprocess_image("a.jpg", 224, 224)  # doctest: +SKIP
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    left = (256 - input_w) // 2
    top = (256 - input_h) // 2
    right = left + input_w
    bottom = top + input_h
    image = image.crop((left, top, right, bottom))
    image = np.asarray(image).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image = (image / 255.0 - mean) / std
    return image


def create_int8_calibrator(
    calib_image_dir: str,
    batch_size: int,
    input_shape: tuple[int, int, int],
    cache_file: str,
    input_h: int,
    input_w: int,
    calib_dataset_size: int,
) -> object:
    """创建一个 TensorRT INT8 MinMax 校准器实例。

    功能描述：
    扫描 ``calib_image_dir`` 下的图片文件，按随机顺序组织 batch，并在 TensorRT 调用时提供设备指针列表；
    同时支持读取/写入 ``cache_file`` 以复用历史校准结果。

    参数说明：
    - calib_image_dir (str): 校准图片目录。
    - batch_size (int): 校准 batch 大小。
    - input_shape (tuple[int, int, int]): 输入形状（C, H, W）。
    - cache_file (str): 校准缓存文件路径。
    - input_h (int): 输入高度。
    - input_w (int): 输入宽度。
    - calib_dataset_size (int): 校准数据集大小上限。

    返回值说明：
    - object: TensorRT 校准器实例（实现 ``IInt8MinMaxCalibrator``）。

    可能抛出的异常：
    - ValueError: 当校准图片数量不足 ``batch_size`` 时触发。
    - Exception: 当 PyCUDA/TensorRT 依赖不可用或内存申请失败时由底层触发。

    使用示例：
    >>> from resnet.trt.calibrator import create_int8_calibrator
    >>> _ = create_int8_calibrator("calib_images", 8, (3, 224, 224), "calib_cache.bin", 224, 224, 2000)  # doctest: +SKIP
    """
    import pycuda.driver as cuda
    import tensorrt as trt

    class Int8Calibrator(trt.IInt8MinMaxCalibrator):
        def __init__(self) -> None:
            """初始化校准器并分配 host/device 输入缓冲区。

            功能描述：
            扫描校准目录中的图片列表，裁剪到 ``calib_dataset_size`` 后打乱顺序；
            并按 ``batch_size`` 与 ``input_shape`` 分配 host 缓冲区与 device 输入缓冲区。

            参数说明：
            - 无。

            返回值说明：
            - None: 无返回值。

            可能抛出的异常：
            - ValueError: 当校准图片数量不足批次大小时触发。
            - Exception: 当 CUDA 内存分配失败时由 PyCUDA 触发。
            """
            trt.IInt8MinMaxCalibrator.__init__(self)
            self.batch_size = batch_size
            self.input_shape = input_shape
            self.cache_file = cache_file
            self.input_h = input_h
            self.input_w = input_w

            image_list = (
                glob.glob(os.path.join(calib_image_dir, "*.jpg"))
                + glob.glob(os.path.join(calib_image_dir, "*.png"))
                + glob.glob(os.path.join(calib_image_dir, "*.jpeg"))
            )
            if len(image_list) < batch_size:
                raise ValueError(f"校准图像数量({len(image_list)})不足批次大小({batch_size})")
            if len(image_list) > calib_dataset_size:
                image_list = image_list[:calib_dataset_size]
            rng = np.random.default_rng(42)
            rng.shuffle(image_list)
            self.image_list = image_list
            self.current_index = 0

            memory_size = trt.volume(input_shape) * batch_size * np.dtype(np.float32).itemsize
            self.device_input = cuda.mem_alloc(memory_size)
            self.host_input = np.zeros([batch_size] + list(input_shape), dtype=np.float32)

        def get_batch_size(self) -> int:
            """返回校准 batch 大小。

            功能描述：
            返回 TensorRT 校准阶段使用的 batch 大小。

            参数说明：
            - 无。

            返回值说明：
            - int: batch 大小。

            可能抛出的异常：
            - 无。
            """
            return self.batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            """提供一个校准 batch 的设备指针列表。

            功能描述：
            读取一批图片并预处理后拷贝到设备输入缓冲区，返回设备指针列表；无更多 batch 时返回 ``None``。

            参数说明：
            - names (list[str]): TensorRT 输入名称列表（当前实现不使用该参数）。

            返回值说明：
            - list[int] | None: 设备输入缓冲区指针列表；无更多 batch 时为 ``None``。

            可能抛出的异常：
            - Exception: 当图片读取、预处理或 CUDA 拷贝失败时由底层触发。
            """
            if self.current_index >= len(self.image_list):
                return None
            end_index = self.current_index + self.batch_size
            if end_index > len(self.image_list):
                remaining = end_index - len(self.image_list)
                batch_images = self.image_list[self.current_index :] + self.image_list[:remaining]
            else:
                batch_images = self.image_list[self.current_index : end_index]

            for i, image_path in enumerate(batch_images):
                try:
                    self.host_input[i] = _preprocess_image(image_path, self.input_h, self.input_w)
                except Exception:
                    self.host_input[i] = np.zeros(self.input_shape, dtype=np.float32)

            cuda.memcpy_htod(self.device_input, self.host_input.ravel())
            self.current_index = end_index
            return [int(self.device_input)]

        def read_calibration_cache(self) -> bytes | None:
            """读取校准缓存文件内容；不存在则返回 None。

            功能描述：
            若 ``cache_file`` 存在则读取其二进制内容并返回，以便复用历史校准结果；
            否则返回 ``None`` 触发重新校准。

            参数说明：
            - 无。

            返回值说明：
            - bytes | None: 缓存内容；不存在时返回 ``None``。

            可能抛出的异常：
            - OSError: 当读取文件失败时由文件系统触发。
            """
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            """写入校准缓存文件内容。

            功能描述：
            将 TensorRT 生成的校准缓存写入 ``cache_file``，便于下次构建引擎时直接复用。

            参数说明：
            - cache (bytes): TensorRT 生成的校准缓存内容。

            返回值说明：
            - None: 无返回值。

            可能抛出的异常：
            - OSError: 当写文件失败时由文件系统触发。
            """
            with open(self.cache_file, "wb") as f:
                f.write(cache)

        def __del__(self) -> None:
            """析构时释放设备输入缓冲区。

            功能描述：
            在对象销毁时尽力释放 ``device_input``，避免 CUDA 内存泄漏。

            参数说明：
            - 无。

            返回值说明：
            - None: 无返回值。

            可能抛出的异常：
            - 无。异常会被捕获并忽略。
            """
            if hasattr(self, "device_input") and self.device_input is not None:
                try:
                    self.device_input.free()
                except Exception:
                    pass

    return Int8Calibrator()
