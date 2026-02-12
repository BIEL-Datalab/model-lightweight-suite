"""基于 PyTorch FX 的 PTQ（INT8）量化脚本。

本脚本提供一个命令行入口，用于对 ResNet50 等模型进行 FX Graph Mode 的后训练量化（PTQ），
并输出量化前后模型大小、精度与推理速度对比结果。
"""

import os
import sys
import argparse

# 设置环境变量解决OpenMP重复初始化问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from utils.env_checks import ensure_conda_env
from utils.data_loader import get_data_loaders
from utils.quantization_utils import compare_model_sizes, create_output_directory
from utils.torch_quantization_utils import (
    calibrate,
    evaluate_model_accuracy,
    load_model,
    save_and_test_loaded_model,
    setup_quantization_config,
    test_and_compare_speed,
)

# 设置警告
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# 随机种子设置
_ = torch.manual_seed(191009)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    功能描述：
    定义并解析 Torch PTQ 量化脚本所需的命令行参数，例如数据集路径、权重路径、保存路径、批次大小等。

    参数说明：
    - 无。

    返回值说明：
    - argparse.Namespace: 解析后的参数对象。

    可能抛出的异常：
    - SystemExit: 当参数解析失败或触发 ``--help`` 时由 argparse 抛出。

    使用示例：
    >>> from quantization.Torch.torch_PTQ import parse_args
    >>> _ = parse_args()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description="使用PyTorch进行模型int8量化")
    parser.add_argument("--data_path", type=str, default="data_set/imagenette", help="数据集路径")
    parser.add_argument("--float_model_file", type=str, default="models/trained/resnet50_imagenette_best_8031.pth", help="输入PyTorch模型文件路径")
    parser.add_argument("--save_quantized_model_path", type=str, default="models/quantized/int8/resnet50_imagenette_quantized_int8.pth", help="输出量化模型文件路径")
    parser.add_argument("--calib_batch_size", type=int, default=50, help="校准批次大小")
    parser.add_argument("--num_workers", type=int, default=47, help="数据加载器工作线程数")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备(cpu或cuda)")
    parser.add_argument("--calib_max_batches", type=int, default=10, help="校准最大批次数量")
    parser.add_argument("--test_num_samples", type=int, default=1000, help="推理速度测试样本数量")
    return parser.parse_args()


def setup_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    """从命令行参数生成配置字典。

    功能描述：
    将 argparse 的参数对象转换为后续流程使用的配置字典，便于函数间传递。

    参数说明：
    - args (argparse.Namespace): 参数对象。

    返回值说明：
    - dict[str, object]: 配置字典。

    可能抛出的异常：
    - AttributeError: 当 ``args`` 缺少预期属性时触发。

    使用示例：
    >>> import argparse
    >>> from quantization.Torch.torch_PTQ import setup_config_from_args
    >>> cfg = setup_config_from_args(argparse.Namespace(data_path="d", float_model_file="m.pth", save_quantized_model_path="q.pt", calib_batch_size=1, num_workers=0, device="cpu", calib_max_batches=1, test_num_samples=10))
    >>> "data_path" in cfg
    True
    """
    return {
        'data_path': args.data_path,
        'float_model_file': args.float_model_file,
        'save_quantized_model_path': args.save_quantized_model_path,
        'calib_batch_size': args.calib_batch_size,
        'num_workers': args.num_workers,
        'device': args.device,
        'calib_max_batches': args.calib_max_batches,
        'test_num_samples': args.test_num_samples
    }


def prepare_models(config: dict[str, object], train_dataset: object, example_inputs: object) -> tuple[nn.Module, nn.Module]:
    """准备 FP32 模型与待量化模型。

    功能描述：
    基于相同权重文件分别加载两个模型实例：
    - FP32 模型：作为对比基线
    - 待量化模型：作为 FX 量化输入
    ``example_inputs`` 参数在当前实现中未使用，仅作为调用侧占位。

    参数说明：
    - config (dict[str, object]): 配置字典，需包含 ``float_model_file`` 与 ``device``。
    - train_dataset (object): 训练集对象，需具备 ``classes`` 属性。
    - example_inputs (object): 示例输入占位参数（当前未使用）。

    返回值说明：
    - tuple[nn.Module, nn.Module]: ``(float_model, model_to_quantize)``。

    可能抛出的异常：
    - KeyError: 当 ``config`` 缺少必要键时触发。
    - RuntimeError: 当权重加载或设备迁移失败时由 PyTorch 触发。

    使用示例：
    >>> from quantization.Torch.torch_PTQ import prepare_models
    >>> _ = prepare_models({"float_model_file": "m.pth", "device": "cpu"}, object(), object())  # doctest: +SKIP
    """
    float_model = load_model(model_file=config['float_model_file'], train_dataset=train_dataset).to(config['device'])
    float_model.eval()
    
    model_to_quantize = load_model(model_file=config['float_model_file'], train_dataset=train_dataset).to(config['device'])
    model_to_quantize.eval()
    
    return float_model, model_to_quantize


def main() -> None:
    """Torch PTQ 量化脚本主入口。

    功能描述：
    依次完成：
    1) 环境检查与参数解析；
    2) 数据加载与示例输入准备；
    3) FX 量化配置、prepare、校准与 convert；
    4) 模型大小/精度/速度对比；
    5) 保存量化模型并重加载验证。

    参数说明：
    - 无。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - Exception: 当数据加载、量化或推理流程失败时由底层依赖触发。

    使用示例：
    >>> from quantization.Torch.torch_PTQ import main
    >>> main()  # doctest: +SKIP
    """
    ensure_conda_env("torch241_cu118_py310", "Torch/PTQ")
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 从参数设置配置信息
    config = setup_config_from_args(args)
    
    # 3. 创建输出目录
    create_output_directory(config['save_quantized_model_path'])
    
    # 4. 加载数据
    train_loader, val_loader, calibration_loader, train_dataset, val_dataset = get_data_loaders(
        data_root=config['data_path'], 
        batch_size=config['calib_batch_size'],
        num_workers=config['num_workers'],
        device=config['device']
    )
    
    # 5. 准备示例输入和损失函数
    example_inputs = next(iter(train_loader))[0]
    criterion = nn.CrossEntropyLoss()
    
    # 6. 准备模型
    float_model, model_to_quantize = prepare_models(config, train_dataset, example_inputs)
    
    # 7. 配置量化参数
    qconfig_mapping = setup_quantization_config()
    
    # 8. 模型准备
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    print(f'\nprepared_model.graph:{prepared_model.graph}')
    
    # 9. 模型校准
    calibrate(prepared_model, calibration_loader, max_batches=config['calib_max_batches'])
    
    # 10. 模型转换
    quantized_model = convert_fx(prepared_model)
    print(f'\nquantized_model:{quantized_model}')
    
    # 11. 模型大小对比
    original_size, quantized_size, size_reduction = compare_model_sizes(float_model, quantized_model)
    
    # 12. 精度评估
    print('='*20 + 'top1, top5' + '='*60)
    evaluate_model_accuracy(float_model, criterion, calibration_loader, "量化前")
    evaluate_model_accuracy(quantized_model, criterion, calibration_loader, "量化后")
    
    # 13. 推理速度测试
    test_and_compare_speed(float_model, quantized_model, val_loader, config['device'])
    
    # 14. 保存和重加载测试
    save_and_test_loaded_model(
        quantized_model, 
        config['save_quantized_model_path'], 
        criterion, 
        calibration_loader, 
        val_loader, 
        config['device']
    )


if __name__ == '__main__':
    main()
