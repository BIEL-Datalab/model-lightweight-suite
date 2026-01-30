import os
import time
import torch
from torch.ao.quantization import QConfig, MovingAverageMinMaxObserver, PerChannelMinMaxObserver, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import torch.nn as nn


def load_model(model_file: str, train_dataset) -> nn.Module:
    """加载预训练的ResNet50模型。"""
    from torchvision.models.resnet import resnet50
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def evaluate(model: nn.Module, criterion: nn.Module, data_loader: torch.utils.data.DataLoader) -> tuple:
    """评估模型在验证集上的性能。"""
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


def calibrate(model: nn.Module, data_loader: torch.utils.data.DataLoader, max_batches: int = 10) -> None:
    """校准量化模型。"""
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


def test_inference_speed(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str, num_samples: int = 1000) -> tuple:
    """测试模型的推理速度。"""
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
    """配置量化参数。"""
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


def evaluate_model_accuracy(model: nn.Module, criterion: nn.Module, calibration_loader: torch.utils.data.DataLoader, model_name: str) -> tuple:
    """评估模型准确率。"""
    from utils.torch_quantization_utils import evaluate
    top1, top5 = evaluate(model, criterion, calibration_loader)
    print(f"[{model_name}] 在验证集上评估精度: {top1.avg:.2f}, {top5.avg:.2f}")
    return top1, top5


def test_and_compare_speed(
    float_model: nn.Module, 
    quantized_model: nn.Module, 
    val_loader: torch.utils.data.DataLoader, 
    device: str
) -> None:
    """测试并比较模型推理速度。"""
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
    calibration_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    device: str
) -> None:
    """保存并测试重加载的量化模型。"""
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


def compare_model_sizes_torch(float_model: nn.Module, quantized_model: nn.Module) -> tuple:
    """比较原始模型和量化模型的大小。"""
    from utils.quantization_utils import print_size_of_model
    print("\n模型大小对比:")
    original_size = print_size_of_model(float_model)
    quantized_size = print_size_of_model(quantized_model)
    size_reduction = (original_size - quantized_size) / original_size * 100
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"量化后模型大小: {quantized_size:.2f} MB")
    print(f"模型大小减少: {size_reduction:.2f}%")
    return original_size, quantized_size, size_reduction


def prepare_models(config: dict, train_dataset, example_inputs) -> tuple:
    """准备原始模型和待量化模型。"""
    from utils.torch_quantization_utils import load_model
    float_model = load_model(model_file=config['float_model_file'], train_dataset=train_dataset).to(config['device'])
    float_model.eval()
    
    model_to_quantize = load_model(model_file=config['float_model_file'], train_dataset=train_dataset).to(config['device'])
    model_to_quantize.eval()
    
    return float_model, model_to_quantize
