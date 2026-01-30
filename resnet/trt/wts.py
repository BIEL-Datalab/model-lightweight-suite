from __future__ import annotations

import os
import struct

import numpy as np
import torch

from .._io import ensure_parent_dir, require_exists


def convert_pth_to_wts(pth_path: str, wts_path: str) -> None:
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

