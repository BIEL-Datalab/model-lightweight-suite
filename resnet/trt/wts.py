"""WTS 权重文件读写工具。

本模块用于在 PyTorch 权重与 TensorRT 常见 WTS 文本格式之间转换，并提供 WTS 权重加载能力。
"""

from __future__ import annotations

import os
import struct

import numpy as np
import torch

from .._io import ensure_parent_dir, require_exists


def convert_pth_to_wts(pth_path: str, wts_path: str) -> None:
    """将 PyTorch 权重/模型对象转换为 WTS 文件。

    功能描述：
    从 ``pth_path`` 加载 checkpoint，并按以下优先级提取 state_dict：
    1) 若为 dict 且包含 ``state_dict`` 键且其值为 dict，则使用该 ``state_dict``；
    2) 若为 dict 且键均为 str，则直接视为 state_dict；
    3) 否则认为是模型对象并调用 ``state_dict()``。
    随后以 TensorRT 常见 WTS 文本格式写入到 ``wts_path``。

    参数说明：
    - pth_path (str): 输入模型文件路径。
    - wts_path (str): 输出 WTS 文件路径。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - FileNotFoundError: 当 ``pth_path`` 不存在时由 ``require_exists`` 触发。
    - OSError: 当无法写入 ``wts_path`` 时触发。
    - RuntimeError: 当 PyTorch 反序列化失败时触发。

    使用示例：
    >>> from resnet.trt.wts import convert_pth_to_wts
    >>> convert_pth_to_wts("model.pth", "model.wts")  # doctest: +SKIP
    """
    require_exists(pth_path, "输入模型文件")
    ensure_parent_dir(wts_path)

    checkpoint = torch.load(pth_path, map_location=torch.device("cpu"), weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    with open(wts_path, "w", encoding="utf-8") as f:
        f.write(f"{len(state_dict.keys())}\n")
        for k, v in state_dict.items():
            vr = v.reshape(-1).detach().cpu().numpy()
            f.write(f"{k} {len(vr)}")
            for vv in vr:
                f.write(" " + struct.pack(">f", float(vv)).hex())
            f.write("\n")


def load_weights(file_path: str) -> dict[str, np.ndarray]:
    """加载 WTS 权重文件并返回权重字典。

    功能描述：
    读取 WTS 文本格式文件，将每一层的 hex float 权重解析为 ``np.ndarray`` 并存入字典返回。

    参数说明：
    - file_path (str): WTS 文件路径（必须以 ``.wts`` 结尾）。

    返回值说明：
    - dict[str, np.ndarray]: 权重字典，键为层名称，值为权重数组（float32）。

    可能抛出的异常：
    - FileNotFoundError: 当 ``file_path`` 不存在时由 ``require_exists`` 触发。
    - ValueError: 当文件格式不合法、层权重数量不匹配或后缀不是 ``.wts`` 时触发。

    使用示例：
    >>> from resnet.trt.wts import load_weights
    >>> _ = load_weights("model.wts")  # doctest: +SKIP
    """
    require_exists(file_path, "WTS权重文件")
    if not file_path.endswith(".wts"):
        raise ValueError(f"Unsupported weight file format: {file_path}. Only .wts is supported")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    count = int(lines[0])
    weight_map: dict[str, np.ndarray] = {}
    for i in range(1, count + 1):
        splits = lines[i].split()
        name = splits[0]
        cur_count = int(splits[1])
        if cur_count + 2 != len(splits):
            raise ValueError(f"Layer {name} has incorrect weight count")
        values = [struct.unpack(">f", bytes.fromhex(splits[j]))[0] for j in range(2, len(splits))]
        weight_map[name] = np.array(values, dtype=np.float32)
    return weight_map
