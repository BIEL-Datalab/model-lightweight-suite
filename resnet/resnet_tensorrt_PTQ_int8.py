import sys

from resnet.run import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"calib-images", "trt-ptq"}:
        argv = ["trt-ptq", *argv]
    main(argv)

