import argparse
import os
import sys

# 添加项目根目录到Python导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 从tensorrt_utils.py导入所有必要的函数和类
from utils.tensorrt_quantization_utils import (
    convert_pth_to_wts,
    Int8Calibrator,
    load_weights,
    build_engine,
    serialize_engine,
    test_inference,
    build_resnet50_network
)

import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
import torch
from utils.quantization_utils import create_output_directory


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用tensorrt进行模型int8量化")
    parser.add_argument("--cuda_id", type=int, default=0, help="使用第几块GPU")
    parser.add_argument("--input", default="models/trained/resnet50_imagenette_best_8031.pth", help="输入PyTorch模型文件路径")
    parser.add_argument("--output", default="models/converted/resnet50_imagenette_best_8031_x86_test.wts", help="输出WTS文件路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--input_h", type=int, default=224, help="输入图像高度")
    parser.add_argument("--input_w", type=int, default=224, help="输入图像宽度")
    parser.add_argument("--output_size", type=int, default=10, help="数据集类别数")
    parser.add_argument("--input_blob_name", default="data", help="输入张量名称")    
    parser.add_argument("--output_blob_name", default="prob", help="输出张量名称")        
    parser.add_argument("--eps", type=float, default=1e-5, help="数值稳定常数")     
    parser.add_argument("--model-name", default="resnet50", help="模型名称，用于日志输出")
    parser.add_argument("--skip-convert", action='store_true', help='跳过PyTorch模型到WTS格式的转换')
    parser.add_argument("--serialize", action='store_true', default=True, help='序列化模型到引擎文件')
    parser.add_argument("--deserialize", action='store_true', help='从引擎文件加载并测试推理')
    parser.add_argument('--int8', action='store_true', default=True, help='使用INT8量化 (仅在序列化时有效)')
    parser.add_argument('--calib-dir', type=str, default="data_set/calib_images", help='校准图像目录路径')
    parser.add_argument('--weight-path', type=str, default="models/converted/resnet50_imagenette_best_8031_x86_test.wts", help='权重文件路径')
    parser.add_argument('--engine-path', type=str, default="models/quantized/int8/resnet50_int8_x86_test.engine", help='引擎文件路径')
    parser.add_argument('--calib-batch-size', type=int, default=8, help='校准批次大小')
    parser.add_argument('--calib-dataset-size', type=int, default=2000, help='校准数据集大小')
    return parser.parse_args()



def validate_args(args: argparse.Namespace) -> None:
    """验证并处理命令行参数。"""
    if args.serialize and args.deserialize:
        print("警告: 同时指定了--serialize和--deserialize，默认使用serialize模式")
        args.deserialize = False
    elif not args.serialize and not args.deserialize:
        args.serialize = True



def check_device(args: argparse.Namespace) -> None:
    """检查并设置CUDA设备。"""
    cuda.init()
    if args.cuda_id >= cuda.Device.count():
        print(f"警告: 指定的设备索引 {args.cuda_id} 超出范围，使用默认设备 0")
        args.cuda_id = 0



def run_model_conversion(args: argparse.Namespace) -> None:
    """执行PyTorch模型到WTS格式的转换。"""
    if not args.skip_convert:
        if args.output is None:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"{base_name}.wts"
        
        # 如果用户没有指定weight-path，确保它与output一致，这样转换后的文件能被正确加载
        if not hasattr(args, 'weight_path') or args.weight_path is None:
            args.weight_path = args.output
        
        if not convert_pth_to_wts(args.input, args.output, args.model_name):
            print("模型转换失败，程序退出")
            sys.exit(1)



def execute_main_operation(args: argparse.Namespace) -> None:
    """执行主操作（序列化或推理）。"""
    print("\n" + "="*60)
    print("TensorRT 10.0.1 ResNet50 INT8 量化工具")
    print("="*60)
    
    # 版本信息
    try:
        trt_version = trt.__version__
    except AttributeError:
        trt_version = "无法获取"
    print(f"TensorRT版本: {trt_version}")
    
    # 执行主操作
    if args.serialize:
        serialize_engine(
            max_batch_size=args.batch_size,
            use_int8=args.int8,
            weight_path=args.weight_path,
            input_blob_name=args.input_blob_name,
            input_h=args.input_h,
            input_w=args.input_w,
            output_size=args.output_size,
            output_blob_name=args.output_blob_name,
            eps=args.eps,
            calib_dir=args.calib_dir,
            calib_batch_size=args.calib_batch_size,
            calib_dataset_size=args.calib_dataset_size,
            engine_path=args.engine_path
        )
    else:
        test_inference(engine_path=args.engine_path)
    
    print("\n" + "="*60)
    print("操作成功完成!")
    print("="*60)



def main() -> None:
    """使用TensorRT进行模型INT8量化的主函数。"""
    try:
        # 1. 解析命令行参数
        args = parse_args()
        
        # 2. 验证参数
        validate_args(args)
        
        # 3. 检查设备
        check_device(args)
        
        # 4. PyTorch模型转WTS
        run_model_conversion(args)
        
        # 5. 执行量化或推理
        execute_main_operation(args)
        
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__=='__main__':
    main()
