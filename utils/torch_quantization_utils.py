"""基于 PyTorch FX Graph Mode 的 PTQ 量化辅助函数。

本模块封装了 FX Graph Mode 的量化配置、校准、评估与推理速度测试等步骤，
用于将 ResNet50 等模型从 FP32 转换为 INT8 并进行对比评估。
"""

import os
import time
import torch
from torch.ao.quantization import QConfig, MovingAverageMinMaxObserver, PerChannelMinMaxObserver, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import torch.nn as nn


def load_model(model_file: str, train_dataset: object) -> nn.Module:
    """加载 ResNet50 模型并载入权重。

    功能描述：
    构建不带预训练权重的 ResNet50，根据 ``train_dataset`` 的类别数重建最后一层全连接，
    然后从 ``model_file`` 加载权重并返回模型实例。

    参数说明：
    - model_file (str): 权重文件路径（通常为 ``.pth``）。
    - train_dataset (object): 训练集对象，需具备 ``classes`` 属性以确定分类数。

    返回值说明：
    - nn.Module: 已加载权重的模型实例。

    可能抛出的异常：
    - FileNotFoundError: 当 ``model_file`` 不存在时触发（由底层文件读取触发）。
    - RuntimeError: 当权重与模型结构不匹配或反序列化失败时由 PyTorch 触发。
    - AttributeError: 当 ``train_dataset`` 缺少 ``classes`` 属性时触发。

    使用示例：
    >>> from utils.torch_quantization_utils import load_model
    >>> _ = load_model("path/to/model.pth", train_dataset=object())  # doctest: +SKIP
    """
    from torchvision.models.resnet import resnet50
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def evaluate(
    model: nn.Module, criterion: nn.Module, data_loader: "torch.utils.data.DataLoader[object]"
) -> "tuple[object, object]":
    """评估模型在验证集上的精度指标。

    功能描述：
    在 ``data_loader`` 上前向推理并计算 Top-1 与 Top-5 准确率的累计平均值，返回两个 ``AverageMeter`` 对象。

    参数说明：
    - model (nn.Module): 待评估模型。
    - criterion (nn.Module): 损失函数（当前实现计算但不返回 loss）。
    - data_loader (torch.utils.data.DataLoader[object]): 提供 ``(image, target)`` 的数据加载器。

    返回值说明：
    - tuple[object, object]: 二元组 ``(top1, top5)``，通常为 ``AverageMeter`` 实例。

    可能抛出的异常：
    - RuntimeError: 当输入张量设备/维度不匹配导致推理失败时由 PyTorch 触发。

    使用示例：
    >>> from utils.torch_quantization_utils import evaluate
    >>> _ = evaluate(model=object(), criterion=object(), data_loader=object())  # doctest: +SKIP
    """
    from utils.quantization_utils import AverageMeter, accuracy
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')
    return top1, top5


def calibrate(
    model: nn.Module, data_loader: "torch.utils.data.DataLoader[object]", max_batches: int = 10
) -> None:
    """对量化准备阶段的模型执行校准（收集量化统计量）。

    功能描述：
    在 ``max_batches`` 个 batch 上运行前向推理，以触发 observer 收集激活/权重范围等统计量。

    参数说明：
    - model (nn.Module): 处于量化准备（prepare）后的模型。
    - data_loader (torch.utils.data.DataLoader[object]): 校准数据加载器。
    - max_batches (int): 最多使用的 batch 数，默认 10。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - RuntimeError: 当推理执行失败时由 PyTorch 触发。

    使用示例：
    >>> from utils.torch_quantization_utils import calibrate
    >>> calibrate(model=object(), data_loader=object(), max_batches=1)  # doctest: +SKIP
    """
    model.eval()
    print("开始校准模型...")
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            model(image)
            cnt += 1
            print(f"已完成 {cnt} 个批次的校准")
            if cnt >= max_batches:
                break
    print(f"校准完成，共使用 {cnt} 个批次")


def test_inference_speed(
    model: nn.Module,
    data_loader: "torch.utils.data.DataLoader[object]",
    device: str,
    num_samples: int = 1000,
) -> tuple[float, float]:
    """测试模型推理速度并返回吞吐量与延迟。

    功能描述：
    先进行少量预热推理，然后在 ``num_samples`` 的样本数上统计总耗时，计算吞吐量（samples/s）与平均延迟（ms/sample）。

    参数说明：
    - model (nn.Module): 待测速模型。
    - data_loader (torch.utils.data.DataLoader[object]): 提供输入 batch 的数据加载器。
    - device (str): 推理设备字符串（用于 ``Tensor.to``），例如 ``'cpu'`` 或 ``'cuda:0'``。
    - num_samples (int): 最大测试样本数，默认 1000。

    返回值说明：
    - tuple[float, float]: 二元组 ``(throughput, latency)``，分别为吞吐量（samples/s）与延迟（ms/sample）。

    可能抛出的异常：
    - RuntimeError: 当推理执行失败或设备不可用时由 PyTorch 触发。

    使用示例：
    >>> from utils.torch_quantization_utils import test_inference_speed
    >>> _ = test_inference_speed(model=object(), data_loader=object(), device="cpu")  # doctest: +SKIP
    """
    model.eval()
    total_time = 0.0
    total_samples = 0
    
    print("预热阶段...")
    with torch.no_grad():
        for i, (image, _) in enumerate(data_loader):
            if i >= 5:
                break
            image = image.to(device)
            _ = model(image)
    
    print("正式测试推理速度...")
    with torch.no_grad():
        for image, _ in data_loader:
            if total_samples >= num_samples:
                break
            
            image = image.to(device)
            start_time = time.time()
            _ = model(image)
            end_time = time.time()
            
            batch_size = image.size(0)
            total_time += (end_time - start_time)
            total_samples += batch_size
    
    throughput = total_samples / total_time
    latency = total_time / total_samples * 1000
    
    print(f"测试完成: {total_samples} 个样本, 总时间: {total_time:.2f} 秒")
    return throughput, latency


def setup_quantization_config() -> QConfigMapping:
    """构建并返回全局量化配置（QConfigMapping）。

    功能描述：
    使用 MovingAverageMinMaxObserver 作为激活 observer，PerChannelMinMaxObserver 作为权重 observer，
    构建全局量化配置并返回。

    参数说明：
    - 无。

    返回值说明：
    - QConfigMapping: FX Graph Mode 量化配置映射。

    可能抛出的异常：
    - 无。

    使用示例：
    >>> from utils.torch_quantization_utils import setup_quantization_config
    >>> m = setup_quantization_config()
    >>> hasattr(m, "set_global")
    True
    """
    qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8,
            quant_min=0,
            quant_max=255,
            averaging_constant=0.001
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            quant_min=-128,
            quant_max=127,
            ch_axis=0
        )
    )
    
    return QConfigMapping().set_global(qconfig)


def evaluate_model_accuracy(
    model: nn.Module,
    criterion: nn.Module,
    calibration_loader: "torch.utils.data.DataLoader[object]",
    model_name: str,
) -> "tuple[object, object]":
    """评估模型准确率并打印结果。

    功能描述：
    复用本模块的 ``evaluate`` 在给定数据加载器上计算 Top-1/Top-5，并打印平均值。

    参数说明：
    - model (nn.Module): 待评估模型。
    - criterion (nn.Module): 损失函数。
    - calibration_loader (torch.utils.data.DataLoader[object]): 用于评估的数据加载器。
    - model_name (str): 模型名称，用于日志展示。

    返回值说明：
    - tuple[object, object]: ``(top1, top5)``，通常为 ``AverageMeter`` 实例。

    可能抛出的异常：
    - RuntimeError: 当推理执行失败时由 PyTorch 触发。

    使用示例：
    >>> from utils.torch_quantization_utils import evaluate_model_accuracy
    >>> _ = evaluate_model_accuracy(object(), object(), object(), "fp32")  # doctest: +SKIP
    """
    from utils.torch_quantization_utils import evaluate
    top1, top5 = evaluate(model, criterion, calibration_loader)
    print(f"[{model_name}] 在验证集上评估精度: {top1.avg:.2f}, {top5.avg:.2f}")
    return top1, top5


def test_and_compare_speed(
    float_model: nn.Module, 
    quantized_model: nn.Module, 
    val_loader: "torch.utils.data.DataLoader[object]", 
    device: str
) -> None:
    """测试并打印 FP32 与 INT8 模型推理速度对比。

    功能描述：
    分别调用 ``test_inference_speed`` 评估 FP32 与 INT8 的吞吐量与延迟，并打印对比信息。

    参数说明：
    - float_model (nn.Module): FP32 模型。
    - quantized_model (nn.Module): 量化模型。
    - val_loader (torch.utils.data.DataLoader[object]): 用于测速的数据加载器。
    - device (str): 推理设备字符串。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - RuntimeError: 当推理执行失败时由 PyTorch 触发。

    使用示例：
    >>> from utils.torch_quantization_utils import test_and_compare_speed
    >>> test_and_compare_speed(object(), object(), object(), device="cpu")  # doctest: +SKIP
    """
    from utils.torch_quantization_utils import test_inference_speed
    print("\n测试推理速度...")
    fp32_throughput, fp32_latency = test_inference_speed(float_model, val_loader, device=device, num_samples=1000)
    quant_throughput, quant_latency = test_inference_speed(quantized_model, val_loader, device=device, num_samples=1000)
    
    print("\n[int8重加载前] 推理速度对比:")
    print(f"FP32模型: {fp32_throughput:.2f} 样本/秒, 延迟: {fp32_latency:.2f} 毫秒/样本")
    print(f"INT8模型: {quant_throughput:.2f} 样本/秒, 延迟: {quant_latency:.2f} 毫秒/样本")
    print(f"吞吐量提升: {quant_throughput/fp32_throughput:.2f}x")
    print(f"延迟降低: {fp32_latency/quant_latency:.2f}x")


def save_and_test_loaded_model(
    quantized_model: nn.Module, 
    save_path: str, 
    criterion: nn.Module, 
    calibration_loader: "torch.utils.data.DataLoader[object]", 
    val_loader: "torch.utils.data.DataLoader[object]", 
    device: str
) -> None:
    """保存量化模型并重加载后进行精度与速度测试。

    功能描述：
    将 ``quantized_model`` 以 TorchScript 形式保存到 ``save_path``，随后重加载并在给定数据集上测试精度与推理速度。

    参数说明：
    - quantized_model (nn.Module): 量化模型。
    - save_path (str): TorchScript 保存路径。
    - criterion (nn.Module): 损失函数。
    - calibration_loader (torch.utils.data.DataLoader[object]): 评估精度的数据加载器。
    - val_loader (torch.utils.data.DataLoader[object]): 测速的数据加载器。
    - device (str): 推理设备字符串。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - OSError: 保存路径不可写或读写失败时触发。
    - RuntimeError: TorchScript 保存/加载或推理失败时由 PyTorch 触发。

    使用示例：
    >>> from utils.torch_quantization_utils import save_and_test_loaded_model
    >>> save_and_test_loaded_model(object(), "m.pt", object(), object(), object(), device="cpu")  # doctest: +SKIP
    """
    from utils.quantization_utils import create_output_directory
    from utils.torch_quantization_utils import evaluate_model_accuracy, test_inference_speed
    # 创建输出目录
    create_output_directory(save_path)
    # 保存量化模型
    torch.jit.save(torch.jit.script(quantized_model), save_path)
    print(f"\n量化模型已保存到: {save_path}")
    
    # 重加载测试
    loaded_quantized_model = torch.jit.load(save_path)
    evaluate_model_accuracy(loaded_quantized_model, criterion, calibration_loader, "重加载后")
    
    print("\n[int8重加载后] 推理速度:")
    quant_throughput, quant_latency = test_inference_speed(loaded_quantized_model, val_loader, device=device, num_samples=1000)
    print(f"重加载后INT8模型: {quant_throughput:.2f} 样本/秒, 延迟: {quant_latency:.2f} 毫秒/样本")


def compare_model_sizes_torch(float_model: nn.Module, quantized_model: nn.Module) -> tuple[float, float, float]:
    """比较 FP32 与量化模型的大小并打印差异。

    功能描述：
    调用 ``print_size_of_model`` 分别打印并返回两者大小（MB），并计算大小减少百分比。

    参数说明：
    - float_model (nn.Module): FP32 模型。
    - quantized_model (nn.Module): 量化模型。

    返回值说明：
    - tuple: ``(original_size, quantized_size, size_reduction)``，单位分别为 MB、MB、百分比。

    可能抛出的异常：
    - RuntimeError/OSError: 当 TorchScript 保存失败或临时文件操作失败时由底层触发。

    使用示例：
    >>> from utils.torch_quantization_utils import compare_model_sizes_torch
    >>> _ = compare_model_sizes_torch(object(), object())  # doctest: +SKIP
    """
    from utils.quantization_utils import print_size_of_model
    print("\n模型大小对比:")
    original_size = print_size_of_model(float_model)
    quantized_size = print_size_of_model(quantized_model)
    size_reduction = (original_size - quantized_size) / original_size * 100
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"量化后模型大小: {quantized_size:.2f} MB")
    print(f"模型大小减少: {size_reduction:.2f}%")
    return original_size, quantized_size, size_reduction


def prepare_models(config: dict[str, object], train_dataset: object, example_inputs: object) -> tuple[nn.Module, nn.Module]:
    """准备 FP32 模型与待量化模型。

    功能描述：
    使用相同权重文件分别加载两个模型实例，其中一个作为 FP32 基线模型，另一个作为后续 FX 量化流程的输入模型。
    ``example_inputs`` 参数在当前实现中未使用，仅作为调用侧占位。

    参数说明：
    - config (dict[str, object]): 配置字典，需包含键 ``'float_model_file'`` 与 ``'device'``。
    - train_dataset (object): 训练集对象，需具备 ``classes`` 属性。
    - example_inputs (object): 示例输入占位参数（当前未使用）。

    返回值说明：
    - tuple[nn.Module, nn.Module]: 二元组 ``(float_model, model_to_quantize)``。

    可能抛出的异常：
    - KeyError: 当 ``config`` 缺少必需键时触发。
    - RuntimeError: 当权重加载或模型迁移到设备失败时由 PyTorch 触发。

    使用示例：
    >>> from utils.torch_quantization_utils import prepare_models
    >>> _ = prepare_models({"float_model_file": "m.pth", "device": "cpu"}, object(), object())  # doctest: +SKIP
    """
    from utils.torch_quantization_utils import load_model
    float_model = load_model(model_file=config['float_model_file'], train_dataset=train_dataset).to(config['device'])
    float_model.eval()
    
    model_to_quantize = load_model(model_file=config['float_model_file'], train_dataset=train_dataset).to(config['device'])
    model_to_quantize.eval()
    
    return float_model, model_to_quantize
