"""ResNet TensorRT PTQ 工作流封装。

本模块将多个底层工具函数组合为更高层的工作流入口，供命令行脚本或上层程序调用。
"""

from __future__ import annotations

import os

from ._io import ensure_parent_dir, require_exists
from .calib.prepare_images import prepare_calib_images
from .config import CalibImagesConfig, TensorRTPTQConfig
from .trt.cuda import cuda_device_context
from .trt.engine import serialize_engine, test_inference
from .trt.wts import convert_pth_to_wts


def run_prepare_calib_images(cfg: CalibImagesConfig) -> None:
    """执行校准图片准备工作流。

    功能描述：
    校验输入目录存在后创建输出目录，并调用 ``prepare_calib_images`` 生成校准图片集合。

    参数说明：
    - cfg (CalibImagesConfig): 校准图片准备参数配置。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - FileNotFoundError: 当 ``cfg.input_dir`` 不存在时由 ``require_exists`` 触发。
    - Exception: 当图片处理或写入失败时由底层依赖触发。

    使用示例：
    >>> from resnet.config import CalibImagesConfig
    >>> from resnet.workflows import run_prepare_calib_images
    >>> run_prepare_calib_images(CalibImagesConfig("train", "calib", 10, 224, 224, 42))  # doctest: +SKIP
    """
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
    """执行 TensorRT PTQ（可选 INT8）工作流。

    功能描述：
    根据 ``cfg.serialize`` / ``cfg.deserialize`` 选择执行“构建并保存引擎”或“加载并测试引擎”的流程；
    在未设置 ``cfg.skip_convert`` 时先执行 ``pth -> wts`` 权重转换。若同时指定 serialize 与 deserialize 会直接报错。

    参数说明：
    - cfg (TensorRTPTQConfig): TensorRT PTQ 流程参数配置。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - ValueError: 当 serialize/deserialize 参数组合非法时触发。
    - FileNotFoundError: 当权重文件或校准目录不存在时触发（由下游函数触发）。
    - Exception: 当 TensorRT/pycuda 等依赖执行失败时由底层触发。

    使用示例：
    >>> from resnet.config import TensorRTPTQConfig
    >>> from resnet.workflows import run_tensorrt_ptq
    >>> run_tensorrt_ptq(TensorRTPTQConfig(  # doctest: +SKIP
    ...     cuda_id=0, pth_path="m.pth", wts_path="m.wts", weight_path="m.wts", engine_path="m.engine",
    ...     batch_size=1, input_h=224, input_w=224, output_size=10, input_blob_name="data", output_blob_name="prob",
    ...     eps=1e-5, use_int8=True, calib_dir="data_set/calib_images", calib_batch_size=8, calib_dataset_size=2000,
    ...     skip_convert=True, serialize=True, deserialize=False,
    ... ))
    """
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
