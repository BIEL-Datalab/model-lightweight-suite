from __future__ import annotations

import os

from ._io import ensure_parent_dir, require_exists
from .calib.prepare_images import prepare_calib_images
from .config import CalibImagesConfig, TensorRTPTQConfig
from .trt.cuda import cuda_device_context
from .trt.engine import serialize_engine, test_inference
from .trt.wts import convert_pth_to_wts


def run_prepare_calib_images(cfg: CalibImagesConfig) -> None:
    require_exists(cfg.input_dir, "源图像目录")
    os.makedirs(cfg.output_dir, exist_ok=True)
    prepare_calib_images(
        input_dir=cfg.input_dir,
        output_dir=cfg.output_dir,
        num_images=cfg.num_images,
        target_size=(cfg.target_w, cfg.target_h),
        shuffle_seed=cfg.seed,
    )


def run_tensorrt_ptq(cfg: TensorRTPTQConfig) -> None:
    if cfg.serialize and cfg.deserialize:
        raise ValueError("不能同时指定 serialize 与 deserialize")
    if not cfg.serialize and not cfg.deserialize:
        raise ValueError("必须指定 serialize 或 deserialize")

    if not cfg.skip_convert:
        ensure_parent_dir(cfg.wts_path)
        convert_pth_to_wts(cfg.pth_path, cfg.wts_path)

    with cuda_device_context(cfg.cuda_id):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(cfg.cuda_id)
        except Exception:
            pass

        if cfg.serialize:
            serialize_engine(
                max_batch_size=cfg.batch_size,
                use_int8=cfg.use_int8,
                weight_path=cfg.weight_path,
                input_blob_name=cfg.input_blob_name,
                input_h=cfg.input_h,
                input_w=cfg.input_w,
                output_size=cfg.output_size,
                output_blob_name=cfg.output_blob_name,
                eps=cfg.eps,
                calib_dir=cfg.calib_dir,
                calib_batch_size=cfg.calib_batch_size,
                calib_dataset_size=cfg.calib_dataset_size,
                engine_path=cfg.engine_path,
            )
        else:
            test_inference(cfg.engine_path)
