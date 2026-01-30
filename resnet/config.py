from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibImagesConfig:
    input_dir: str
    output_dir: str
    num_images: int
    target_h: int
    target_w: int
    seed: int


@dataclass(frozen=True)
class TensorRTPTQConfig:
    cuda_id: int
    pth_path: str
    wts_path: str
    weight_path: str
    engine_path: str
    batch_size: int
    input_h: int
    input_w: int
    output_size: int
    input_blob_name: str
    output_blob_name: str
    eps: float
    use_int8: bool
    calib_dir: str
    calib_batch_size: int
    calib_dataset_size: int
    skip_convert: bool
    serialize: bool
    deserialize: bool
