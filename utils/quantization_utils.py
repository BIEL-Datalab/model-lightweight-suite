"""量化流程的通用辅助工具。

本模块提供若干与 PTQ（Post-Training Quantization）相关的通用能力，包括：
指标统计、Top-K 准确率计算、量化结果落盘、模型体积评估与报告打印等。
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any


class AverageMeter:
    """用于统计标量指标的当前值、累计值与平均值。

    功能描述：
    该类常用于训练/评估循环中对 loss、accuracy 等标量指标进行累积统计；在部分调用场景下，
    ``val`` 也可能直接传入 ``torch.Tensor``（例如 Top-K 准确率的张量结果），当前实现同样支持。

    参数说明：
    - name (str): 指标名称，用于字符串展示。
    - fmt (str): 数值格式化字符串片段（用于 ``format``），默认 ``':f'``。

    返回值说明：
    - AverageMeter: 统计器实例。

    可能抛出的异常：
    - ZeroDivisionError: 当 ``update`` 从未被调用且访问 ``avg`` 时不会触发；但若外部手动将 ``count`` 设为 0 并再次计算平均值，可能触发除零错误。

    使用示例：
    >>> from utils.quantization_utils import AverageMeter
    >>> meter = AverageMeter("loss")
    >>> meter.update(1.0, n=2)
    >>> meter.update(2.0, n=1)
    >>> round(meter.avg, 6)
    1.333333
    """
    def __init__(self, name: str, fmt: str = ':f') -> None:
        self.val: float | torch.Tensor
        self.avg: float | torch.Tensor
        self.sum: float | torch.Tensor
        self.count: int
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float | torch.Tensor, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """计算模型输出的 Top-K 准确率。

    功能描述：
    给定模型输出 ``output`` 与真实标签 ``target``，按 ``topk`` 指定的 k 值计算 Top-K 准确率，
    返回每个 k 对应的百分比（0-100）。

    参数说明：
    - output (torch.Tensor): 模型输出的 logits 或得分张量，形状通常为 ``(N, C)``。
    - target (torch.Tensor): 真实标签张量，形状通常为 ``(N,)``。
    - topk (tuple[int, ...]): 需要计算的 k 值集合，默认仅计算 Top-1。

    返回值说明：
    - list[torch.Tensor]: 与 ``topk`` 等长的列表；每个元素为 1 元素张量，表示对应 Top-K 准确率（百分比）。

    可能抛出的异常：
    - RuntimeError: 当 ``output`` 维度不符合 ``topk`` 所需或张量设备不一致时，由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from utils.quantization_utils import accuracy
    >>> output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    >>> target = torch.tensor([1, 0])
    >>> [round(v.item(), 1) for v in accuracy(output, target, topk=(1, 2))]
    [100.0, 100.0]
    """
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
    """创建输出目录（若不存在则创建）。

    功能描述：
    根据 ``output_path`` 推导其父目录并在不存在时创建；若父目录为空（例如仅文件名），则不做任何操作。

    参数说明：
    - output_path (str): 输出文件路径或包含文件名的相对/绝对路径。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - OSError: 当目录创建失败（权限不足、路径非法等）时，由 ``os.makedirs`` 触发。

    使用示例：
    >>> import os
    >>> import tempfile
    >>> from utils.quantization_utils import create_output_directory
    >>> tmp = tempfile.TemporaryDirectory()
    >>> out_path = os.path.join(tmp.name, "subdir", "a.json")
    >>> create_output_directory(out_path)  # doctest: +ELLIPSIS
    创建输出目录: ...
    >>> os.path.isdir(os.path.dirname(out_path))
    True
    >>> tmp.cleanup()
    """
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
    """将量化结果保存为 JSON 文件。

    功能描述：
    将原始模型信息、量化模型信息与对比信息组织为字典，并以 JSON 形式写入 ``output_path``。

    参数说明：
    - original_model (Dict[str, Any]): 原始模型的统计/元数据字典。
    - quantized_model (Dict[str, Any]): 量化模型的统计/元数据字典。
    - comparison (Dict[str, Any]): 对比结果字典（例如大小、精度、性能差异）。
    - output_path (str): 输出 JSON 文件路径，默认 ``'models/ONNX/quantization_results.json'``。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - OSError: 当无法创建目录或无法写入文件时触发。
    - TypeError: 当输入字典包含不可 JSON 序列化对象时，由 ``json.dump`` 触发。

    使用示例：
    >>> import json
    >>> import os
    >>> import tempfile
    >>> from utils.quantization_utils import save_quantization_results
    >>> tmp = tempfile.TemporaryDirectory()
    >>> out_path = os.path.join(tmp.name, "out", "q.json")
    >>> save_quantization_results({"a": 1}, {"b": 2}, {"c": 3}, output_path=out_path)  # doctest: +ELLIPSIS
    创建输出目录: ...
    量化结果已保存到: ...
    >>> with open(out_path, "r") as f:
    ...     data = json.load(f)
    >>> sorted(data.keys())
    ['comparison', 'original_model', 'quantized_model']
    >>> tmp.cleanup()
    """
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
    """打印并返回模型的大小（单位：MB）。

    功能描述：
    将模型保存为临时 TorchScript 文件 ``temp.p``，计算其文件大小并打印，随后删除临时文件。

    参数说明：
    - model (nn.Module): 需要评估大小的 PyTorch 模型。

    返回值说明：
    - float: 模型大小（MB）。

    可能抛出的异常：
    - RuntimeError: TorchScript 编译或保存失败时触发。
    - OSError: 临时文件读写或删除失败时触发。

    使用示例：
    >>> import torch
    >>> from utils.quantization_utils import print_size_of_model
    >>> _ = print_size_of_model(torch.nn.Linear(2, 3))  # doctest: +SKIP
    """
    if isinstance(model, torch.jit.RecursiveScriptModule):  # type: ignore[attr-defined]
        torch.jit.save(model, "temp.p")  # type: ignore[no-untyped-call]
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")  # type: ignore[no-untyped-call]
    size_mb = os.path.getsize("temp.p") / 1e6
    print(f"Size (MB): {size_mb:.2f}")
    os.remove("temp.p")
    return size_mb


def compare_model_sizes(original_model: Any, quantized_model: Any) -> tuple[float, float, float]:
    """比较原始模型与量化模型的大小并返回差异信息。

    功能描述：
    支持以 ``nn.Module`` 或模型文件路径的形式输入原始模型与量化模型，计算两者大小（MB），并返回缩减百分比。

    参数说明：
    - original_model (Any): 原始模型（``nn.Module``）或模型文件路径（``str``）。
    - quantized_model (Any): 量化模型（``nn.Module``）或模型文件路径（``str``）。

    返回值说明：
    - tuple[float, float, float]: 依次为原始模型大小（MB）、量化模型大小（MB）、大小减少百分比。

    可能抛出的异常：
    - ValueError: 当输入既不是 ``nn.Module`` 也不是存在的文件路径时触发。
    - OSError: 当读取文件大小失败时触发。
    - RuntimeError: 当对 ``nn.Module`` 进行 TorchScript 保存失败时触发（由 ``print_size_of_model`` 触发）。

    使用示例：
    >>> import torch
    >>> from utils.quantization_utils import compare_model_sizes
    >>> _ = compare_model_sizes(torch.nn.Linear(2, 3), torch.nn.Linear(2, 3))  # doctest: +SKIP
    """
    def get_model_size(model: Any) -> float:
        """获取模型大小（MB）。

        功能描述：
        根据 ``model`` 的类型（``nn.Module`` 或文件路径）计算其体积。

        参数说明：
        - model (Any): ``nn.Module`` 或文件路径（``str``）。

        返回值说明：
        - float: 模型大小（MB）。

        可能抛出的异常：
        - ValueError: 当输入类型不受支持时触发。
        - OSError: 当文件路径存在但无法读取大小时触发。
        - RuntimeError: 当 TorchScript 保存失败时触发。
        """
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
    """打印 PTQ 量化评估报告到标准输出。

    功能描述：
    按固定模板打印模型名称、量化模式、模型大小、精度、推理时间与吞吐量提升等指标，便于在命令行快速查看。

    参数说明：
    - model_name (str): 模型名称或标识。
    - quantization_mode (str): 量化模式标识（例如 ``'torch_int8'``、``'tensorrt_int8'``）。
    - original_size (float): 原始模型大小（MB）。
    - quantized_size (float): 量化后模型大小（MB）。
    - original_accuracy (float): 原始模型准确率（百分比）。
    - quantized_accuracy (float): 量化后模型准确率（百分比）。
    - original_inference_time (float): 原始模型推理时间（ms）。
    - quantized_inference_time (float): 量化后模型推理时间（ms）。
    - throughput_improvement (float): 吞吐量提升（百分比）。
    - calibration_data (str): 校准数据说明字符串；为空时不打印该行。

    返回值说明：
    - None: 无返回值。

    可能抛出的异常：
    - ZeroDivisionError: 当 ``quantized_inference_time`` 为 0 导致除零时触发。

    使用示例：
    >>> from utils.quantization_utils import print_quantization_report
    >>> print_quantization_report(
    ...     model_name="resnet50",
    ...     quantization_mode="torch_int8",
    ...     original_size=100.0,
    ...     quantized_size=30.0,
    ...     original_accuracy=76.0,
    ...     quantized_accuracy=75.0,
    ...     original_inference_time=10.0,
    ...     quantized_inference_time=5.0,
    ...     throughput_improvement=100.0,
    ... )  # doctest: +SKIP
    """
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
