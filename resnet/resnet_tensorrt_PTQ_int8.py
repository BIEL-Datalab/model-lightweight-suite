"""TensorRT PTQ 引擎构建脚本入口（兼容旧调用方式）。

本脚本将命令行参数转发给 ``resnet.run.main``，并在未显式指定子命令时默认使用 ``trt-ptq``。
"""

import sys

from resnet.run import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"calib-images", "trt-ptq"}:
        argv = ["trt-ptq", *argv]
    main(argv)
