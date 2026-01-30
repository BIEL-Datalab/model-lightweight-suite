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


def load_model(model_file: str, train_dataset, device: str) -> nn.Module:
    """加载PyTorch模型并初始化权重。"""
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
    """将PyTorch模型转换为ONNX格式。"""
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
    data_loader: DataLoader, 
    model_name: str, 
    max_batches: int = 500
) -> dict:
    """评估ONNX模型的性能。"""
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
    """应用PTQ量化。"""
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


def prepare_validation_dataset(args: any) -> DataLoader:
    """准备验证数据集。"""
    print("\n准备验证数据集...")
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = datasets.ImageFolder(os.path.join(args.dataset_path, 'val'), transform=val_transform)
    return DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


def create_onnx_sessions(args: any) -> tuple:
    """创建原始模型和量化模型的ONNX Runtime会话。"""
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
    val_loader: DataLoader
) -> tuple:
    """评估模型并对比结果。"""
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
    args: any,
    original_size: float,
    quantized_size: float,
    size_reduction: float,
    original_results: dict,
    quantized_results: dict,
    accuracy_diff: float,
    speedup: float,
    throughput_improvement: float
) -> None:
    """生成并保存量化结果报告。"""
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
