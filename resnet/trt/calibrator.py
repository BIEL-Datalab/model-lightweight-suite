from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image


def _preprocess_image(image_path: str, input_h: int, input_w: int) -> np.ndarray:
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
):
    import pycuda.driver as cuda
    import tensorrt as trt

    class Int8Calibrator(trt.IInt8MinMaxCalibrator):
        def __init__(self) -> None:
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
            return self.batch_size

        def get_batch(self, names: list) -> list | None:
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
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            with open(self.cache_file, "wb") as f:
                f.write(cache)

        def __del__(self) -> None:
            if hasattr(self, "device_input") and self.device_input is not None:
                try:
                    self.device_input.free()
                except Exception:
                    pass

    return Int8Calibrator()

