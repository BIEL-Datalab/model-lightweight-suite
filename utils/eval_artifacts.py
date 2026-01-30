import json
import os
from datetime import datetime
from typing import Any


def validate_file_path(file_path: str, description: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} 路径不存在: {file_path}")


def create_timestamped_dir(root_dir: str, prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(root_dir, f"{prefix}_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)
    return result_folder


def write_json(file_path: str, data: Any) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(file_path: str, content: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

