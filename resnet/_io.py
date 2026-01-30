from __future__ import annotations

import os


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def require_exists(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} 不存在: {path}")

