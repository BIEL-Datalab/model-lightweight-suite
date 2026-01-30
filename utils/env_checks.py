import os
import sys
from typing import Iterable


def _get_conda_env_name() -> str | None:
    name = os.environ.get("CONDA_DEFAULT_ENV")
    return name if name else None


def ensure_conda_env(expected_names: str | Iterable[str], feature: str) -> None:
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

