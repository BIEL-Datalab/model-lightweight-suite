"""基于 ModelOpt 的 ONNX PTQ 量化工具集。

本模块提供一组面向 ResNet50 + ImageFolder 数据集的脚本化工具函数，用于：
1) 加载 PyTorch 权重并构建模型；
2) 导出 ONNX；
3) 使用 modelopt.onnx.quantization 执行 PTQ（Post-Training Quantization）；
4) 使用 onnxruntime 对原始/量化模型进行推理评估并输出对比结果；
5) 生成并落盘量化结果报告。
"""

import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
import time
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import modelopt.onnx.quantization as moq

from utils.data_loader import get_data_loaders
from utils.quantization_utils import create_output_directory, save_quantization_results, print_quantization_report, compare_model_sizes


def load_model(model_file: str, train_dataset: object, device: str) -> nn.Module:
    """加载 PyTorch 模型并载入权重。

    功能描述：
    构建一个不带预训练权重的 ``resnet50``，根据 ``train_dataset`` 的类别数重建最后一层全连接，
    然后从 ``model_file`` 加载权重并返回模型实例。

    参数说明：
    - model_file (str): ``.pth`` 权重文件路径。
    - train_dataset (object): 训练数据集对象，需具备 ``classes`` 属性（用于确定分类数）。
    - device (str): 权重加载的设备映射（传递给 ``torch.load(map_location=...)``），例如 ``'cpu'`` 或 ``'cuda:0'``。

    返回值说明：
    - nn.Module: 已加载权重的模型实例。

    可能抛出的异常：
    - FileNotFoundError: 当 ``model_file`` 不存在时由底层文件读取触发。
    - RuntimeError: 当权重与模型结构不匹配或反序列化失败时由 PyTorch 触发。
    - AttributeError: 当 ``train_dataset`` 缺少 ``classes`` 属性时触发。

    使用示例：
    >>> from utils.modelopt_quantization_utils import load_model
    >>> # 需要真实权重文件与数据集对象；此示例仅展示调用方式
    >>> _ = load_model("path/to/model.pth", train_dataset=object(), device="cpu")  # doctest: +SKIP
    """
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    state_dict = torch.load(model_file, weights_only=False, map_location=device)
    model.load_state_dict(state_dict)
    return model


def convert_pth_to_onnx(
    pth_path: str, 
    onnx_path: str, 
    dataset_path: str, 
    batch_size: int, 
    num_workers: int, 
    device: str, 
    model_name: str = "resnet50"
) -> bool:
    """将 PyTorch 权重导出为 ONNX 模型文件。

    功能描述：
    从 ``pth_path`` 加载权重，构建模型并基于训练集的一个 batch 作为示例输入，
    调用 ``torch.onnx.export`` 导出 ONNX 文件到 ``onnx_path``。

    参数说明：
    - pth_path (str): PyTorch 权重文件路径。
    - onnx_path (str): ONNX 输出文件路径。
    - dataset_path (str): 数据集根目录路径（需包含 train/val 子目录）。
    - batch_size (int): DataLoader 批次大小。
    - num_workers (int): DataLoader 工作线程数。
    - device (str): 权重加载设备映射字符串（例如 ``'cpu'``、``'cuda:0'``）。
    - model_name (str): 模型名称展示用字符串，默认 ``"resnet50"``。

    返回值说明：
    - bool: 成功导出返回 ``True``；输入文件不存在或导出异常返回 ``False``。

    可能抛出的异常：
    - 无。函数内部捕获导出异常并返回 ``False``；其余阶段异常通常由外部依赖触发并中断执行。

    使用示例：
    >>> from utils.modelopt_quantization_utils import convert_pth_to_onnx
    >>> _ = convert_pth_to_onnx(
    ...     pth_path="model.pth",
    ...     onnx_path="model.onnx",
    ...     dataset_path="path/to/dataset",
    ...     batch_size=32,
    ...     num_workers=4,
    ...     device="cpu",
    ... )  # doctest: +SKIP
    """
    print("\n" + "="*60)
    print("PyTorch模型转ONNX格式脚本")
    print("="*60)
    print(f"模型名称: {model_name}")
    print(f"输入文件: {pth_path}")
    print(f"输出文件: {onnx_path}")
    print("="*60)
    
    if not os.path.exists(pth_path):
        print(f"错误: 输入文件 {pth_path} 不存在")
        return False
    
    # 创建输出目录
    create_output_directory(onnx_path)
    
    # 加载数据
    print("正在加载数据集...")
    train_loader, val_loader, calibration_loader, train_dataset, val_dataset = get_data_loaders(
        data_root=dataset_path, 
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )
    
    # 获取示例输入
    example_inputs = next(iter(train_loader))[0]
    print(f"示例输入形状: {example_inputs.shape}")
    
    # 加载模型
    print("正在加载模型...")
    model = load_model(model_file=pth_path, train_dataset=train_dataset, device=device)
    model.eval()
    print("模型加载成功")
    
    # 导出ONNX模型
    print("正在导出ONNX模型...")
    try:
        torch.onnx.export(
            model, 
            example_inputs, 
            onnx_path,
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )
        print("ONNX模型导出成功")
        print(f"ONNX模型已保存到: {onnx_path}")
        print(f"ONNX模型大小: {os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB")
        return True
    except Exception as e:
        print(f"错误: 导出ONNX模型失败: {e}")
        return False


def evaluate_model(
    session: ort.InferenceSession, 
    data_loader: DataLoader[object], 
    model_name: str, 
    max_batches: int = 500
) -> dict[str, float]:
    """评估 ONNX 模型的准确率与推理性能。

    功能描述：
    使用 ``onnxruntime.InferenceSession`` 对 ``data_loader`` 提供的输入批次进行推理，
    统计 Top-1 分类准确率、平均推理耗时（ms）与吞吐量（图像/秒）。

    参数说明：
    - session (ort.InferenceSession): ONNX Runtime 推理会话。
    - data_loader (DataLoader[object]): 提供 ``(images, labels)`` 的数据加载器。
    - model_name (str): 用于进度条与日志展示的模型名称。
    - max_batches (int): 最多评估的 batch 数，默认 500。

    返回值说明：
    - dict[str, float]: 结果字典，包含：
      - ``accuracy``：准确率（百分比）
      - ``avg_inference_time``：平均推理耗时（ms）
      - ``throughput``：吞吐量（图像/秒）

    可能抛出的异常：
    - RuntimeError: 当推理输入输出维度不匹配或执行失败时由 onnxruntime 触发。

    使用示例：
    >>> from utils.modelopt_quantization_utils import evaluate_model
    >>> # 需要有效的 InferenceSession 与数据加载器；此示例仅展示调用方式
    >>> _ = evaluate_model(session=object(), data_loader=object(), model_name="test")  # doctest: +SKIP
    """
    correct = 0
    total = 0
    inference_times = []
    
    print(f"\n评估{model_name}模型 ...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc=f"{model_name}评估")):
        if batch_idx >= max_batches:
            break
            
        images_np = images.numpy()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        start_time = time.perf_counter()
        outputs = session.run([output_name], {input_name: images_np})
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000
        predictions = np.argmax(outputs[0], axis=1)
        correct += np.sum(predictions == labels.numpy())
        total += len(labels)
        inference_times.append(inference_time)
    
    accuracy = correct / total * 100
    avg_inference_time = np.mean(inference_times)
    throughput = 1000 / avg_inference_time * data_loader.batch_size
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'throughput': throughput
    }


def perform_quantization(
    output_onnx: str, 
    calibration_data: str, 
    output_quant: str
) -> None:
    """对 ONNX 模型执行 PTQ 量化并做结构校验。

    功能描述：
    从 ``calibration_data`` 指向的 ``.npy`` 文件加载校准数据，调用 ``moq.quantize`` 生成 INT8 量化模型，
    并使用 ``onnx.checker.check_model`` 对量化结果做基础结构校验。

    参数说明：
    - output_onnx (str): 原始 ONNX 模型路径。
    - calibration_data (str): 校准数据 ``.npy`` 文件路径。
    - output_quant (str): 量化后 ONNX 模型输出路径。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - FileNotFoundError: 当校准数据文件不存在时由 NumPy 触发。
    - Exception: 当量化过程失败时由 modelopt/onnxruntime/onnx 等依赖触发。

    使用示例：
    >>> from utils.modelopt_quantization_utils import perform_quantization
    >>> perform_quantization("model.onnx", "calib.npy", "model_int8.onnx")  # doctest: +SKIP
    """
    # 加载校准数据
    print("\n" + "="*60)
    print("使用modelopt进行ONNX模型PTQ量化")
    print("="*60)
    print(f"校准数据路径: {calibration_data}")
    calibration_data = np.load(calibration_data)
    
    print("开始对ResNet50模型进行PTQ量化...")
    print(f"校准数据形状: {calibration_data.shape}")
    print(f"原始ONNX模型: {output_onnx}")
    print(f"量化后模型: {output_quant}")
    
    # 创建量化模型输出目录
    create_output_directory(output_quant)
    
    # 应用PTQ量化
    moq.quantize(
        onnx_path=output_onnx,
        calibration_data=calibration_data,
        output_path=output_quant,
        quantize_mode="int8",
        per_channel_quantization=True,
        op_types_to_quantize=["Conv", "Gemm", "MatMul", "AveragePool"],
        op_types_to_exclude=[
            "Softmax", "Sigmoid", "Add", "Concat", "BatchNormalization",
            "Relu", "Clip", "GlobalAveragePool", "Flatten", "Identity", "fc"
        ],
        activation_quantization_type="per_tensor",
        weight_quantization_type="per_channel",
        quantize_residuals=False,
        calibration_batch_size=32,
        verbose=True,
        quant_format="qdq",
        activation_symmetric=False,
        weight_symmetric=True,
        enable_distributed_calibration=True
    )
    
    print(f"量化模型已保存到: {output_quant}")
    
    # 验证量化模型
    print("\n验证量化模型结构...")
    quant_model = onnx.load(output_quant)
    onnx.checker.check_model(quant_model)
    print("量化模型结构验证通过")


def prepare_validation_dataset(args: object) -> DataLoader[object]:
    """构建验证集 DataLoader。

    功能描述：
    基于 ``args.dataset_path`` 下的 ``val`` 子目录创建 ImageFolder 数据集，并返回默认 batch_size=32 的 DataLoader。

    参数说明：
    - args (object): 参数对象，需具备 ``dataset_path`` 属性。

    返回值说明：
    - DataLoader[object]: 验证集数据加载器。

    可能抛出的异常：
    - AttributeError: 当 ``args`` 缺少 ``dataset_path`` 属性时触发。
    - FileNotFoundError: 当 ``val`` 目录不存在时由 ImageFolder 触发。

    使用示例：
    >>> from utils.modelopt_quantization_utils import prepare_validation_dataset
    >>> prepare_validation_dataset(args=object())  # doctest: +SKIP
    """
    print("\n准备验证数据集...")
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = datasets.ImageFolder(os.path.join(args.dataset_path, 'val'), transform=val_transform)
    return DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


def create_onnx_sessions(args: object) -> tuple[object, object]:
    """创建原始模型与量化模型的 ONNX Runtime 推理会话。

    功能描述：
    读取 ``args.output_onnx`` 与 ``args.output_quant`` 两个路径，分别创建原始模型与量化模型的推理会话，
    并配置 CUDA/TensorRT/CPU 等 provider 组合。

    参数说明：
    - args (object): 参数对象，需具备 ``output_onnx`` 与 ``output_quant`` 属性。

    返回值说明：
    - tuple[object, object]: 二元组 ``(original_session, quantized_session)``。

    可能抛出的异常：
    - AttributeError: 当 ``args`` 缺少必要属性时触发。
    - Exception: 当 onnxruntime 创建会话失败时由其依赖触发。

    使用示例：
    >>> from utils.modelopt_quantization_utils import create_onnx_sessions
    >>> _ = create_onnx_sessions(args=object())  # doctest: +SKIP
    """
    print("\n创建ONNX Runtime会话...")
    
    # 原始模型会话配置
    original_session = ort.InferenceSession(
        args.output_onnx, 
        providers=[
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
    )
    
    # 量化模型会话配置
    quantized_session = ort.InferenceSession(
        args.output_quant, 
        providers=[
            ('TensorrtExecutionProvider', { 
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache',
                'trt_fp16_enable': True,
            }),
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
                'enable_cuda_graph': False
            }),
            'CPUExecutionProvider'
        ]
    )
    
    return original_session, quantized_session


def evaluate_and_compare_models(
    original_session: ort.InferenceSession,
    quantized_session: ort.InferenceSession,
    val_loader: DataLoader[object]
) -> tuple[dict[str, float], dict[str, float], float, float, float]:
    """评估原始/量化模型并返回对比结果。

    功能描述：
    分别调用 ``evaluate_model`` 评估原始会话与量化会话的准确率与性能指标，
    并计算准确率差异、加速倍数与吞吐量提升。

    参数说明：
    - original_session (ort.InferenceSession): 原始模型推理会话。
    - quantized_session (ort.InferenceSession): 量化模型推理会话。
    - val_loader (DataLoader[object]): 验证集数据加载器。

    返回值说明：
    - tuple[dict[str, float], dict[str, float], float, float, float]:
      依次为原始结果字典、量化结果字典、准确率差异（原始-量化，百分比）、加速倍数、吞吐量提升（百分比）。

    可能抛出的异常：
    - RuntimeError: 当推理执行失败时由 onnxruntime 触发。

    使用示例：
    >>> from utils.modelopt_quantization_utils import evaluate_and_compare_models
    >>> _ = evaluate_and_compare_models(object(), object(), object())  # doctest: +SKIP
    """
    # 评估模型
    original_results = evaluate_model(original_session, val_loader, "原始")
    quantized_results = evaluate_model(quantized_session, val_loader, "量化")
    
    # 打印评估结果
    print(f"原始模型准确率: {original_results['accuracy']:.2f}%")
    print(f"原始模型平均推理时间: {original_results['avg_inference_time']:.2f} ms")
    print(f"原始模型吞吐量: {original_results['throughput']:.2f} 图像/秒")
    
    print(f"量化模型准确率: {quantized_results['accuracy']:.2f}%")
    print(f"量化模型平均推理时间: {quantized_results['avg_inference_time']:.2f} ms")
    print(f"量化模型吞吐量: {quantized_results['throughput']:.2f} 图像/秒")
    
    # 结果对比
    accuracy_diff = original_results['accuracy'] - quantized_results['accuracy']
    speedup = original_results['avg_inference_time'] / quantized_results['avg_inference_time'] if quantized_results['avg_inference_time'] > 0 else 0
    throughput_improvement = quantized_results['throughput'] / original_results['throughput'] * 100 - 100 if original_results['throughput'] > 0 else 0
    
    print("\n对比结果:")
    print(f"准确率差异: {accuracy_diff:.2f}% (原始 - 量化)")
    print(f"加速倍数: {speedup:.2f}x")
    print(f"吞吐量提升: {throughput_improvement:.2f}%")
    
    return original_results, quantized_results, accuracy_diff, speedup, throughput_improvement


def generate_and_save_results(
    args: object,
    original_size: float,
    quantized_size: float,
    size_reduction: float,
    original_results: dict[str, float],
    quantized_results: dict[str, float],
    accuracy_diff: float,
    speedup: float,
    throughput_improvement: float
) -> None:
    """生成量化报告并保存结果到 JSON 文件。

    功能描述：
    先通过 ``print_quantization_report`` 打印汇总报告，然后组织结果字典并调用
    ``save_quantization_results`` 写入 ``models/ONNX/quantization_results.json``。

    参数说明：
    - args (object): 参数对象，需具备 ``calibration_data`` 属性（用于报告展示）。
    - original_size (float): 原始模型大小（MB）。
    - quantized_size (float): 量化模型大小（MB）。
    - size_reduction (float): 大小减少百分比。
    - original_results (dict[str, float]): 原始模型评估结果字典（来自 ``evaluate_model``）。
    - quantized_results (dict[str, float]): 量化模型评估结果字典（来自 ``evaluate_model``）。
    - accuracy_diff (float): 准确率差异（原始-量化，百分比）。
    - speedup (float): 加速倍数。
    - throughput_improvement (float): 吞吐量提升（百分比）。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - AttributeError: 当 ``args`` 缺少 ``calibration_data`` 属性时触发。
    - OSError/TypeError: 当写入 JSON 失败或数据不可序列化时由底层触发。

    使用示例：
    >>> from utils.modelopt_quantization_utils import generate_and_save_results
    >>> generate_and_save_results(  # doctest: +SKIP
    ...     args=object(),
    ...     original_size=1.0,
    ...     quantized_size=1.0,
    ...     size_reduction=0.0,
    ...     original_results={"accuracy": 0.0, "avg_inference_time": 1.0, "throughput": 1.0},
    ...     quantized_results={"accuracy": 0.0, "avg_inference_time": 1.0, "throughput": 1.0},
    ...     accuracy_diff=0.0,
    ...     speedup=1.0,
    ...     throughput_improvement=0.0,
    ... )
    """
    # 生成并保存评估报告
    print_quantization_report(
        model_name="ResNet50 on ImageNette数据集",
        quantization_mode="INT8",
        original_size=original_size,
        quantized_size=quantized_size,
        original_accuracy=original_results['accuracy'],
        quantized_accuracy=quantized_results['accuracy'],
        original_inference_time=original_results['avg_inference_time'],
        quantized_inference_time=quantized_results['avg_inference_time'],
        throughput_improvement=throughput_improvement,
        calibration_data=args.calibration_data
    )
    
    # 保存结果到文件
    results = {
        'original_model': {
            'size_mb': original_size,
            'accuracy': original_results['accuracy'],
            'avg_inference_time_ms': original_results['avg_inference_time'],
            'throughput_images_per_sec': original_results['throughput']
        },
        'quantized_model': {
            'size_mb': quantized_size,
            'accuracy': quantized_results['accuracy'],
            'avg_inference_time_ms': quantized_results['avg_inference_time'],
            'throughput_images_per_sec': quantized_results['throughput']
        },
        'comparison': {
            'size_reduction_percent': size_reduction,
            'accuracy_difference_percent': accuracy_diff,
            'speedup_factor': speedup,
            'throughput_improvement_percent': throughput_improvement
        }
    }
    
    save_quantization_results(
        original_model=results['original_model'],
        quantized_model=results['quantized_model'],
        comparison=results['comparison'],
        output_path='models/ONNX/quantization_results.json'
    )
