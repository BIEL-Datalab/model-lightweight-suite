"""运行环境检查工具。

本模块用于在脚本入口处进行轻量级环境约束检查（例如 Conda 环境名称），以避免在错误环境中运行导致难以诊断的问题。
"""

import os
import sys
from typing import Iterable


def _get_conda_env_name() -> str | None:
    """获取当前 Conda 环境名称。

    功能描述：
    读取环境变量 ``CONDA_DEFAULT_ENV`` 并将其转换为 ``str | None``。

    参数说明：
    - 无。

    返回值说明：
    - str | None: 当前 Conda 环境名称；当未检测到 Conda 环境时返回 ``None``。

    可能抛出的异常：
    - 无。
    """
    name = os.environ.get("CONDA_DEFAULT_ENV")
    return name if name else None


def ensure_conda_env(expected_names: str | Iterable[str], feature: str) -> None:
    """确保当前进程运行在期望的 Conda 环境中。

    功能描述：
    在未显式跳过检查且非 ``-h/--help`` 场景下，检查当前 Conda 环境是否属于 ``expected_names``；
    若不满足则抛出 ``RuntimeError``。当无法获取当前环境名称时（例如非 Conda 环境），函数直接返回。

    参数说明：
    - expected_names (str | Iterable[str]): 允许的 Conda 环境名称或名称列表。
    - feature (str): 当前功能/脚本名称，用于错误提示文案。

    返回值说明：
    - None: 满足约束或无法判定时返回。

    可能抛出的异常：
    - RuntimeError: 当检测到当前 Conda 环境名称不在 ``expected_names`` 中时触发。

    使用示例：
    >>> from utils.env_checks import ensure_conda_env
    >>> # 该检查依赖当前进程环境变量；此示例仅展示调用方式
    >>> ensure_conda_env("my-env", feature="quantization")  # doctest: +SKIP
    """
    if os.environ.get("MODEL_TINY_SUITE_SKIP_ENV_CHECK") == "1":
        return
    if "-h" in sys.argv or "--help" in sys.argv:
        return
    expected = [expected_names] if isinstance(expected_names, str) else list(expected_names)
    current = _get_conda_env_name()
    if current is None:
        return
    if current not in expected:
        expected_display = ", ".join(expected)
        raise RuntimeError(
            f"{feature} 需要在 Conda 环境 [{expected_display}] 中运行，当前环境为 [{current}]。"
        )

