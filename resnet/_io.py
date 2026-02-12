"""ResNet 工具链的 I/O 辅助函数。

本模块提供简单的路径创建与存在性校验函数，用于工作流内部的文件系统操作前置检查。
"""

from __future__ import annotations

import os


def ensure_parent_dir(file_path: str) -> None:
    """确保目标文件路径的父目录存在（不存在则创建）。

    功能描述：
    根据 ``file_path`` 推导父目录路径并在其非空且不存在时创建目录。

    参数说明：
    - file_path (str): 目标文件路径。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - OSError: 当目录创建失败时由 ``os.makedirs`` 触发。

    使用示例：
    >>> import os
    >>> import tempfile
    >>> from resnet._io import ensure_parent_dir
    >>> tmp = tempfile.TemporaryDirectory()
    >>> p = os.path.join(tmp.name, "a", "b.txt")
    >>> ensure_parent_dir(p)
    >>> os.path.isdir(os.path.dirname(p))
    True
    >>> tmp.cleanup()
    """
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def require_exists(path: str, description: str) -> None:
    """要求路径存在，否则抛出异常。

    功能描述：
    当 ``path`` 不存在时抛出 ``FileNotFoundError``，异常消息包含 ``description`` 便于定位。

    参数说明：
    - path (str): 需要校验的路径。
    - description (str): 路径用途描述（用于异常消息）。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - FileNotFoundError: 当 ``path`` 不存在时触发。

    使用示例：
    >>> from resnet._io import require_exists
    >>> require_exists("path/that/does/not/exist", "测试路径")  # doctest: +SKIP
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} 不存在: {path}")
