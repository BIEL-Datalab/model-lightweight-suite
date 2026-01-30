import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any


class AverageMeter:
    """用于统计指标的平均值、总和等信息的工具类。"""
    def __init__(self, name: str, fmt: str = ':f') -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """计算模型输出的准确率。"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_output_directory(output_path: str) -> None:
    """创建输出目录（如果不存在）。"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")


def save_quantization_results(
    original_model: Dict[str, Any],
    quantized_model: Dict[str, Any],
    comparison: Dict[str, Any],
    output_path: str = 'models/ONNX/quantization_results.json'
) -> None:
    """保存量化结果到JSON文件。"""
    results = {
        'original_model': original_model,
        'quantized_model': quantized_model,
        'comparison': comparison
    }
    
    create_output_directory(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"量化结果已保存到: {output_path}")


def print_size_of_model(model: nn.Module) -> float:
    """打印并返回模型的大小（以MB为单位）。"""
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    print(f"Size (MB): {size_mb:.2f}")
    os.remove("temp.p")
    return size_mb


def compare_model_sizes(original_model: Any, quantized_model: Any) -> tuple:
    """比较原始模型和量化模型的大小，并返回大小信息。
    
    Args:
        original_model: 原始模型，可以是nn.Module或模型文件路径。
        quantized_model: 量化模型，可以是nn.Module或模型文件路径。
        
    Returns:
        tuple: (原始模型大小MB, 量化模型大小MB, 大小减少百分比)
    """
    def get_model_size(model: Any) -> float:
        """获取模型大小（MB）。"""
        if isinstance(model, nn.Module):
            return print_size_of_model(model)
        elif isinstance(model, str) and os.path.exists(model):
            size_mb = os.path.getsize(model) / (1024 * 1024)
            print(f"Size (MB): {size_mb:.2f}")
            return size_mb
        else:
            raise ValueError(f"不支持的模型类型: {type(model)}")
    
    print("="*80)
    print("量化前模型大小：")
    original_size = get_model_size(original_model)
    print("量化后模型大小：")
    quantized_size = get_model_size(quantized_model)
    print("="*80)
    
    size_reduction = (original_size - quantized_size) / original_size * 100
    return original_size, quantized_size, size_reduction


def print_quantization_report(
    model_name: str,
    quantization_mode: str,
    original_size: float,
    quantized_size: float,
    original_accuracy: float,
    quantized_accuracy: float,
    original_inference_time: float,
    quantized_inference_time: float,
    throughput_improvement: float,
    calibration_data: str = ""
) -> None:
    """打印量化评估报告。"""
    print("\n" + "="*60)
    print("PTQ量化评估报告")
    print("="*60)
    print(f"模型: {model_name}")
    print(f"量化模式: {quantization_mode}")
    if calibration_data:
        print(f"校准数据: {calibration_data}")
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"量化模型大小: {quantized_size:.2f} MB")
    print(f"大小减少: {(original_size - quantized_size) / original_size * 100:.2f}%")
    print(f"原始准确率: {original_accuracy:.2f}%")
    print(f"量化准确率: {quantized_accuracy:.2f}%")
    print(f"准确率下降: {original_accuracy - quantized_accuracy:.2f}%")
    print(f"原始推理时间: {original_inference_time:.2f} ms")
    print(f"量化推理时间: {quantized_inference_time:.2f} ms")
    print(f"加速倍数: {original_inference_time / quantized_inference_time:.2f}x")
    print(f"吞吐量提升: {throughput_improvement:.2f}%")
    print("="*60)
