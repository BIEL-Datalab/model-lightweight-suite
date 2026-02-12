"""基于 NVIDIA ModelOpt 的 ONNX PTQ（INT8）量化脚本。

本脚本提供一个命令行入口，用于：
1) 将 PyTorch 权重导出为 ONNX；
2) 使用 ModelOpt 对 ONNX 模型执行 PTQ 量化；
3) 创建 ONNX Runtime 会话对比评估原始与量化模型；
4) 生成并保存量化结果报告。
"""

import argparse
import os
import sys

# 设置环境变量解决OpenMP重复初始化问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.env_checks import ensure_conda_env
from utils.data_loader import get_data_loaders
from utils.quantization_utils import create_output_directory, compare_model_sizes
from utils.modelopt_quantization_utils import (
    convert_pth_to_onnx,
    create_onnx_sessions,
    evaluate_and_compare_models,
    generate_and_save_results,
    perform_quantization,
    prepare_validation_dataset,
)

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    功能描述：
    定义并解析 ModelOpt PTQ 脚本所需的命令行参数，包括输入权重路径、ONNX 输出路径、量化输出路径、
    校准数据路径与数据集路径等。

    参数说明：
    - 无。

    返回值说明：
    - argparse.Namespace: 解析后的参数对象。

    可能抛出的异常：
    - SystemExit: 当参数解析失败或触发 ``--help`` 时由 argparse 抛出。

    使用示例：
    >>> from quantization.ModelOpt.ModelOpt_PTQ import parse_args
    >>> _ = parse_args()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description="使用modelopt进行ONNX模型PTQ量化")
    parser.add_argument("--input", default="models/trained/resnet50_imagenette_best_8031.pth", help="输入PyTorch模型文件路径 (.pth 或 .pt)")
    parser.add_argument("--output_onnx", default="models/converted/resnet50_imagenette_best_8031_test.onnx", help="输出ONNX模型文件路径")
    parser.add_argument("--output_quant", default="models/quantized/int8/resnet50_imagenette_modelopt_int8_x86_test.onnx", help="输出量化模型文件路径")
    parser.add_argument("--calibration_data", default="data_set/calibration_data_500.npy", help="校准数据路径")
    parser.add_argument("--dataset_path", default="data_set/imagenette", help="数据集路径")
    parser.add_argument("--batch_size", type=int, default=100, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=47, help="数据加载器工作线程数")
    parser.add_argument("--device", default="cuda:1", help="设备类型")
    parser.add_argument("--skip_convert", action='store_true', help='跳过PyTorch模型到ONNX格式的转换')
    parser.add_argument("--model-name", default="resnet50", help="模型名称，用于日志输出")
    return parser.parse_args()


def run_onnx_conversion(args: argparse.Namespace) -> None:
    """执行 PyTorch 模型到 ONNX 的转换步骤。

    功能描述：
    当未设置 ``args.skip_convert`` 时，调用 ``convert_pth_to_onnx`` 将 PyTorch 权重导出为 ONNX；
    若转换失败则直接退出进程。

    参数说明：
    - args (argparse.Namespace): 参数对象，需包含 ``input/output_onnx/dataset_path/batch_size/num_workers/device/model_name/skip_convert`` 等属性。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - SystemExit: 当转换失败时触发（显式 ``sys.exit(1)``）。

    使用示例：
    >>> import argparse
    >>> from quantization.ModelOpt.ModelOpt_PTQ import run_onnx_conversion
    >>> run_onnx_conversion(argparse.Namespace(skip_convert=True))  # doctest: +SKIP
    """
    if not args.skip_convert:
        if not convert_pth_to_onnx(
            pth_path=args.input,
            onnx_path=args.output_onnx,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            model_name=args.model_name
        ):
            print("模型转换失败，程序退出")
            sys.exit(1)
    else:
        print("跳过PyTorch模型到ONNX格式的转换")


def main() -> None:
    """ModelOpt PTQ 脚本主入口。

    功能描述：
    依次完成：
    1) 环境检查与参数解析；
    2) 可选的 PyTorch->ONNX 转换；
    3) 对 ONNX 模型执行 PTQ 量化；
    4) 模型大小对比与验证集评估；
    5) 生成并落盘量化结果报告。

    参数说明：
    - 无。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - Exception: 当量化或评估流程失败时由底层依赖触发。

    使用示例：
    >>> from quantization.ModelOpt.ModelOpt_PTQ import main
    >>> main()  # doctest: +SKIP
    """
    ensure_conda_env("modelopt_env", "ModelOpt/PTQ")
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. PyTorch模型转ONNX
    run_onnx_conversion(args)
    
    # 3. 执行PTQ量化
    perform_quantization(
        output_onnx=args.output_onnx,
        calibration_data=args.calibration_data,
        output_quant=args.output_quant
    )
    
    # 4. 模型大小对比
    original_size, quantized_size, size_reduction = compare_model_sizes(args.output_onnx, args.output_quant)
    
    # 5. 准备验证数据集
    val_loader = prepare_validation_dataset(args)
    
    # 6. 创建ONNX Runtime会话
    original_session, quantized_session = create_onnx_sessions(args)
    
    # 7. 评估模型并对比结果
    original_results, quantized_results, accuracy_diff, speedup, throughput_improvement = evaluate_and_compare_models(
        original_session, quantized_session, val_loader
    )
    
    # 8. 生成并保存结果
    generate_and_save_results(
        args,
        original_size,
        quantized_size,
        size_reduction,
        original_results,
        quantized_results,
        accuracy_diff,
        speedup,
        throughput_improvement
    )
    
    print("\n" + "="*60)
    print("操作成功完成!")
    print("="*60)


if __name__ == '__main__':
    main()
