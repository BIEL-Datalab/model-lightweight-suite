"""评估产物（artifacts）读写与路径辅助工具。

本模块提供若干轻量级文件操作函数，用于在评估/量化流程中统一：
- 校验输入输出路径是否存在；
- 创建带时间戳的结果目录；
- 将评估结果写入 JSON 或纯文本文件。
"""

import json
import os
from datetime import datetime
from typing import Any


def validate_file_path(file_path: str, description: str) -> None:
    """校验文件或目录路径是否存在。

    功能描述：
    当 ``file_path`` 不存在时抛出 ``FileNotFoundError``，并在异常消息中附带 ``description`` 便于定位。

    参数说明：
    - file_path (str): 需要校验的文件或目录路径。
    - description (str): 路径用途描述（用于异常消息）。

    返回值说明：
    - None: 校验通过时无返回值。

    可能抛出的异常：
    - FileNotFoundError: 当 ``file_path`` 不存在时触发。

    使用示例：
    >>> from utils.eval_artifacts import validate_file_path
    >>> validate_file_path("path/that/does/not/exist", "测试路径")  # doctest: +SKIP
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} 路径不存在: {file_path}")


def create_timestamped_dir(root_dir: str, prefix: str) -> str:
    """创建一个带时间戳的子目录并返回其路径。

    功能描述：
    在 ``root_dir`` 下创建形如 ``{prefix}_YYYYMMDD_HHMMSS`` 的目录（若已存在则复用），并返回该目录路径。

    参数说明：
    - root_dir (str): 父目录路径。
    - prefix (str): 子目录名前缀。

    返回值说明：
    - str: 创建得到的结果目录路径。

    可能抛出的异常：
    - OSError: 当目录创建失败（权限不足、路径非法等）时触发。

    使用示例：
    >>> import os
    >>> import tempfile
    >>> from utils.eval_artifacts import create_timestamped_dir
    >>> tmp = tempfile.TemporaryDirectory()
    >>> out_dir = create_timestamped_dir(tmp.name, prefix="run")  # doctest: +ELLIPSIS
    >>> os.path.isdir(out_dir)
    True
    >>> tmp.cleanup()
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(root_dir, f"{prefix}_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)
    return result_folder


def write_json(file_path: str, data: Any) -> None:
    """将数据写入 JSON 文件（UTF-8）。

    功能描述：
    若 ``file_path`` 的父目录不存在则创建；随后以 UTF-8 编码写入 JSON（``ensure_ascii=False``）以保留中文。

    参数说明：
    - file_path (str): 输出 JSON 文件路径。
    - data (Any): 需要写入的对象（需可被 ``json`` 序列化）。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - OSError: 文件写入失败时触发。
    - TypeError: 当 ``data`` 含不可 JSON 序列化对象时由 ``json.dump`` 触发。

    使用示例：
    >>> import json
    >>> import os
    >>> import tempfile
    >>> from utils.eval_artifacts import write_json
    >>> tmp = tempfile.TemporaryDirectory()
    >>> p = os.path.join(tmp.name, "a", "b.json")
    >>> write_json(p, {"x": 1})
    >>> with open(p, "r", encoding="utf-8") as f:
    ...     json.load(f)["x"]
    1
    >>> tmp.cleanup()
    """
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(file_path: str, content: str) -> None:
    """将文本内容写入文件（UTF-8）。

    功能描述：
    若 ``file_path`` 的父目录不存在则创建；随后以 UTF-8 编码写入 ``content``。

    参数说明：
    - file_path (str): 输出文本文件路径。
    - content (str): 写入内容。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - OSError: 文件写入失败时触发。

    使用示例：
    >>> import os
    >>> import tempfile
    >>> from utils.eval_artifacts import write_text
    >>> tmp = tempfile.TemporaryDirectory()
    >>> p = os.path.join(tmp.name, "a.txt")
    >>> write_text(p, "hello")
    >>> open(p, "r", encoding="utf-8").read()
    'hello'
    >>> tmp.cleanup()
    """
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

