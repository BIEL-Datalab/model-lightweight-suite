import argparse

from resnet.config import CalibImagesConfig, TensorRTPTQConfig
from resnet.workflows import run_prepare_calib_images, run_tensorrt_ptq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resnet.run", description="ResNet TensorRT PTQ 工具集")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_calib = sub.add_parser("calib-images", help="准备 INT8 校准图像")
    p_calib.add_argument("--input-dir", default="data_set/imagenette/train")
    p_calib.add_argument("--output-dir", default="data_set/calib_images")
    p_calib.add_argument("--num-images", type=int, default=2000)
    p_calib.add_argument("--target-size", type=int, nargs=2, default=[224, 224])
    p_calib.add_argument("--seed", type=int, default=42)

    p_trt = sub.add_parser("trt-ptq", help="构建/测试 TensorRT 引擎（可选 INT8）")
    p_trt.add_argument("--cuda-id", type=int, default=0)
    p_trt.add_argument("--input", dest="pth_path", default="models/trained/resnet50_imagenette_best_8031.pth")
    p_trt.add_argument("--output", dest="wts_path", default="models/converted/resnet50_imagenette_best_8031_x86.wts")
    p_trt.add_argument("--weight-path", default=None)
    p_trt.add_argument("--engine-path", default="models/quantized/int8/resnet50_int8_x86.engine")
    p_trt.add_argument("--batch-size", type=int, default=1)
    p_trt.add_argument("--input-h", type=int, default=224)
    p_trt.add_argument("--input-w", type=int, default=224)
    p_trt.add_argument("--output-size", type=int, default=10)
    p_trt.add_argument("--input-blob-name", default="data")
    p_trt.add_argument("--output-blob-name", default="prob")
    p_trt.add_argument("--eps", type=float, default=1e-5)
    p_trt.add_argument("--skip-convert", action="store_true")
    p_trt.add_argument("--serialize", action="store_true")
    p_trt.add_argument("--deserialize", action="store_true")
    p_trt.add_argument("--int8", dest="use_int8", action="store_true", default=True)
    p_trt.add_argument("--calib-dir", default="data_set/calib_images")
    p_trt.add_argument("--calib-batch-size", type=int, default=8)
    p_trt.add_argument("--calib-dataset-size", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "calib-images":
        cfg = CalibImagesConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_images=args.num_images,
            target_h=args.target_size[1],
            target_w=args.target_size[0],
            seed=args.seed,
        )
        run_prepare_calib_images(cfg)
        return

    cfg = TensorRTPTQConfig(
        cuda_id=args.cuda_id,
        pth_path=args.pth_path,
        wts_path=args.wts_path,
        weight_path=args.weight_path or args.wts_path,
        engine_path=args.engine_path,
        batch_size=args.batch_size,
        input_h=args.input_h,
        input_w=args.input_w,
        output_size=args.output_size,
        input_blob_name=args.input_blob_name,
        output_blob_name=args.output_blob_name,
        eps=args.eps,
        use_int8=args.use_int8,
        calib_dir=args.calib_dir,
        calib_batch_size=args.calib_batch_size,
        calib_dataset_size=args.calib_dataset_size,
        skip_convert=args.skip_convert,
        serialize=args.serialize,
        deserialize=args.deserialize,
    )
    run_tensorrt_ptq(cfg)


if __name__ == "__main__":
    main()
