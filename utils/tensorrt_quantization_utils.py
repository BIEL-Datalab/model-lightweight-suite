import os
import struct
import sys
import glob
from PIL import Image
import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
import torch

# 添加项目根目录到Python导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.quantization_utils import create_output_directory

import tensorrt as trt
import numpy as np


def add_batchnorm_2d(network: trt.INetworkDefinition, weight_map: dict, input_tensor: trt.ITensor, layer_name: str, eps: float = 1e-5) -> trt.IScaleLayer:
    """向TensorRT网络添加BatchNorm2d层。"""
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = np.sqrt(weight_map[layer_name + ".running_var"] + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    
    return network.add_scale(input=input_tensor,
                             mode=trt.ScaleMode.CHANNEL,
                             shift=shift,
                             scale=scale)



def bottleneck(network: trt.INetworkDefinition, weight_map: dict, input_tensor: trt.ITensor, in_channels: int, out_channels: int, stride: int, layer_name: str, eps: float = 1e-5) -> trt.IActivationLayer:
    """向TensorRT网络添加ResNet的Bottleneck块。"""
    # 1x1卷积
    conv1 = network.add_convolution_nd(
        input=input_tensor,
        num_output_maps=out_channels,
        kernel_shape=trt.Dims([1, 1]),
        kernel=weight_map[layer_name + "conv1.weight"],
        bias=trt.Weights()
    )
    assert conv1 is not None, f"创建conv1失败: {layer_name}"
    
    bn1 = add_batchnorm_2d(network, weight_map, conv1.get_output(0), layer_name + "bn1", eps)
    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    
    # 3x3卷积
    conv2 = network.add_convolution_nd(
        input=relu1.get_output(0),
        num_output_maps=out_channels,
        kernel_shape=trt.Dims([3, 3]),
        kernel=weight_map[layer_name + "conv2.weight"],
        bias=trt.Weights()
    )
    assert conv2 is not None, f"创建conv2失败: {layer_name}"
    conv2.stride_nd = trt.Dims([stride, stride])
    conv2.padding_nd = trt.Dims([1, 1])
    
    bn2 = add_batchnorm_2d(network, weight_map, conv2.get_output(0), layer_name + "bn2", eps)
    relu2 = network.add_activation(bn2.get_output(0), type=trt.ActivationType.RELU)
    
    # 1x1卷积
    conv3 = network.add_convolution_nd(
        input=relu2.get_output(0),
        num_output_maps=out_channels * 4,
        kernel_shape=trt.Dims([1, 1]),
        kernel=weight_map[layer_name + "conv3.weight"],
        bias=trt.Weights()
    )
    assert conv3 is not None, f"创建conv3失败: {layer_name}"
    
    bn3 = add_batchnorm_2d(network, weight_map, conv3.get_output(0), layer_name + "bn3", eps)
    
    # 残差连接
    if stride != 1 or in_channels != 4 * out_channels:
        conv4 = network.add_convolution_nd(
            input=input_tensor,
            num_output_maps=out_channels * 4,
            kernel_shape=trt.Dims([1, 1]),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=trt.Weights()
        )
        assert conv4 is not None, f"创建downsample卷积失败: {layer_name}"
        conv4.stride_nd = trt.Dims([stride, stride])
        
        bn4 = add_batchnorm_2d(network, weight_map, conv4.get_output(0), layer_name + "downsample.1", eps)
        residual = bn4.get_output(0)
    else:
        residual = input_tensor
    
    # 残差相加
    ew_sum = network.add_elementwise(
        bn3.get_output(0),
        residual,
        trt.ElementWiseOperation.SUM
    )
    assert ew_sum is not None, f"创建残差连接失败: {layer_name}"
    
    return network.add_activation(ew_sum.get_output(0), type=trt.ActivationType.RELU)



def build_resnet50_network(
    network: trt.INetworkDefinition, 
    weight_map: dict, 
    input_tensor: trt.ITensor, 
    output_size: int, 
    eps: float = 1e-5
) -> trt.ITensor:
    """构建ResNet50网络结构。"""
    print("  - 开始构建ResNet50网络结构...")
    
    # 初始层
    conv1 = network.add_convolution_nd(
        input=input_tensor,
        num_output_maps=64,
        kernel_shape=trt.Dims([7, 7]),
        kernel=weight_map["conv1.weight"],
        bias=trt.Weights()
    )
    conv1.stride_nd = trt.Dims([2, 2])
    conv1.padding_nd = trt.Dims([3, 3])
    
    bn1 = add_batchnorm_2d(network, weight_map, conv1.get_output(0), "bn1", eps)
    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    
    pool1 = network.add_pooling_nd(
        input=relu1.get_output(0),
        window_size=trt.Dims([3, 3]),
        type=trt.PoolingType.MAX
    )
    pool1.stride_nd = trt.Dims([2, 2])
    pool1.padding_nd = trt.Dims([1, 1])
    
    # 构建各个层
    x = bottleneck(network, weight_map, pool1.get_output(0), 64, 64, 1, "layer1.0.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1, "layer1.1.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1, "layer1.2.", eps)
    
    x = bottleneck(network, weight_map, x.get_output(0), 256, 128, 2, "layer2.0.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1, "layer2.1.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1, "layer2.2.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1, "layer2.3.", eps)
    
    x = bottleneck(network, weight_map, x.get_output(0), 512, 256, 2, "layer3.0.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1, "layer3.1.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1, "layer3.2.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1, "layer3.3.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1, "layer3.4.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1, "layer3.5.", eps)
    
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 512, 2, "layer4.0.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1, "layer4.1.", eps)
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1, "layer4.2.", eps)
    
    # 全局平均池化
    pool2 = network.add_pooling_nd(
        input=x.get_output(0),
        window_size=trt.Dims([7, 7]),
        type=trt.PoolingType.AVERAGE
    )
    pool2.stride_nd = trt.Dims([1, 1])
    
    # 重塑
    shuffle = network.add_shuffle(pool2.get_output(0))
    shuffle.reshape_dims = trt.Dims([-1, 2048])
    reshaped_out = shuffle.get_output(0)
    
    # 全连接层
    fc_weight = weight_map['fc.weight'].reshape(output_size, 2048)
    fc_bias = weight_map['fc.bias']
    
    weight_const = network.add_constant(
        trt.Dims([2048, output_size]),
        np.ascontiguousarray(fc_weight.T)
    )
    
    matmul = network.add_matrix_multiply(
        reshaped_out,
        trt.MatrixOperation.NONE,
        weight_const.get_output(0),
        trt.MatrixOperation.NONE
    )
    
    bias_const = network.add_constant(
        trt.Dims([1, output_size]),
        np.ascontiguousarray(fc_bias.reshape(1, output_size))
    )
    
    output_layer = network.add_elementwise(
        matmul.get_output(0),
        bias_const.get_output(0),
        trt.ElementWiseOperation.SUM
    )
    
    return output_layer.get_output(0)


def convert_pth_to_wts(pth_path: str, wts_path: str, model_name: str = "resnet50") -> bool:
    """将PyTorch模型转换为WTS格式。
    
    读取PyTorch模型文件(.pth或.pt)，将权重转换为TensorRT WTS格式并保存。
    
    Args:
        pth_path (str): 输入PyTorch模型文件路径。
        wts_path (str): 输出WTS文件路径。
        model_name (str, optional): 模型名称，用于日志输出，默认"resnet50"。
    
    Returns:
        bool: 转换是否成功。
    """
    print("\n" + "="*60)
    print("PyTorch模型转WTS格式脚本")
    print("="*60)
    print(f"模型名称: {model_name}")
    print(f"输入文件: {pth_path}")
    print(f"输出文件: {wts_path}")
    print("="*60)
    
    if not os.path.exists(pth_path):
        print(f"错误: 输入文件 {pth_path} 不存在")
        return False
    
    print("正在加载模型...")
    try:
        model = torch.load(pth_path, map_location=torch.device('cpu'))
        print("模型加载成功")
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        return False
    
    is_parallel = False
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
        is_parallel = True
        print("已处理并行模型包装")
    
    if isinstance(model, dict):
        state_dict = model
        print(f"模型为权重字典格式，包含 {len(state_dict.keys())} 个权重")
    else:
        state_dict = model.state_dict()
        print(f"从模型中提取权重，包含 {len(state_dict.keys())} 个权重")
    
    print("正在写入WTS文件...")
    try:
        with open(wts_path, 'w', encoding='utf-8') as f:
            f.write(f"{len(state_dict.keys())}\n")
            
            for i, (k, v) in enumerate(state_dict.items()):
                if (i + 1) % 10 == 0 or i + 1 == len(state_dict.keys()):
                    print(f"已处理 {i + 1}/{len(state_dict.keys())} 个权重")
                
                vr = v.reshape(-1).cpu().numpy()
                
                f.write(f"{k} {len(vr)}")
                
                for vv in vr:
                    f.write(" " + struct.pack(">f", float(vv)).hex())
                f.write("\n")
        
        print("\n" + "="*60)
        print("转换完成！")
        print(f"输出文件: {wts_path}")
        print(f"文件大小: {os.path.getsize(wts_path) / 1024 / 1024:.2f} MB")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"错误: 写入文件失败: {e}")
        return False



# INT8校准器实现
class Int8Calibrator(trt.IInt8MinMaxCalibrator):
    """TensorRT INT8校准器类，用于生成校准数据。
    
    实现了TensorRT的IInt8MinMaxCalibrator接口，用于收集校准数据
    以生成INT8量化所需的缩放因子。
    
    Args:
        calib_image_dir (str): 校准图像目录路径。
        batch_size (int): 校准批次大小。
        input_shape (tuple): 模型输入形状，格式为(通道数, 高度, 宽度)。
        cache_file (str, optional): 校准缓存文件路径，默认"calib_cache.bin"。
        input_h (int, optional): 输入图像高度，默认224。
        input_w (int, optional): 输入图像宽度，默认224。
        calib_dataset_size (int, optional): 校准数据集大小，默认2000。
    """
    def __init__(self, calib_image_dir: str, batch_size: int, input_shape: tuple, cache_file: str = "calib_cache.bin", input_h: int = 224, input_w: int = 224, calib_dataset_size: int = 2000):
        trt.IInt8MinMaxCalibrator.__init__(self)
        
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.calib_image_dir = calib_image_dir
        self.cache_file = cache_file
        self.input_h = input_h
        self.input_w = input_w
        self.calib_dataset_size = calib_dataset_size
        
        self.image_list = glob.glob(os.path.join(calib_image_dir, "*.jpg")) + \
                         glob.glob(os.path.join(calib_image_dir, "*.png")) + \
                         glob.glob(os.path.join(calib_image_dir, "*.jpeg"))
        
        if len(self.image_list) < batch_size:
            raise ValueError(f"校准图像数量({len(self.image_list)})不足批次大小({batch_size})")
        
        if len(self.image_list) > self.calib_dataset_size:
            self.image_list = self.image_list[:self.calib_dataset_size]
            print(f"校准数据集大小限制为: {self.calib_dataset_size} 张图片")
        
        np.random.seed(42)
        np.random.shuffle(self.image_list)
        
        self.current_index = 0
        
        memory_size = trt.volume(input_shape) * batch_size * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(memory_size)
        
        self.host_input = np.zeros([batch_size] + list(input_shape), dtype=np.float32)
        
        print(f"INT8校准器初始化完成:")
        print(f"  - 校准图像数量: {len(self.image_list)}")
        print(f"  - 校准批次大小: {batch_size}")
        print(f"  - 输入形状: {input_shape}")
        print(f"  - 校准缓存文件: {cache_file}")
        print(f"  - 已分配校准设备内存: {memory_size / (1024**2):.2f} MB")
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_batch(self, names: list) -> list:
        if self.current_index >= len(self.image_list):
            print("校准完成，无更多批次")
            return None
        
        end_index = self.current_index + self.batch_size
        
        if end_index > len(self.image_list):
            remaining = end_index - len(self.image_list)
            batch_images = self.image_list[self.current_index:] + self.image_list[:remaining]
        else:
            batch_images = self.image_list[self.current_index:end_index]
        
        print(f"校准批次: {self.current_index//self.batch_size + 1}, "              f"图像范围: [{self.current_index}, {min(end_index, len(self.image_list))})")
        
        for i, image_path in enumerate(batch_images):
            try:
                image = Image.open(image_path)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image = image.resize((256, 256), Image.Resampling.LANCZOS)
                left = (256 - self.input_w) // 2
                top = (256 - self.input_h) // 2
                right = left + self.input_w
                bottom = top + self.input_h
                image = image.crop((left, top, right, bottom))
                
                image = np.array(image).astype(np.float32)
                
                image = np.transpose(image, (2, 0, 1))
                
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                image = (image / 255.0 - mean) / std
                
                self.host_input[i] = image
            except Exception as e:
                print(f"警告: 处理图像 {image_path} 时出错: {str(e)}")
                self.host_input[i] = np.zeros(self.input_shape, dtype=np.float32)
        
        cuda.memcpy_htod(self.device_input, self.host_input.ravel())
        
        self.current_index = end_index
        return [int(self.device_input)]
    
    def read_calibration_cache(self) -> bytes:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                cache_data = f.read()
            print(f"加载校准缓存: {self.cache_file}")
            return cache_data
        print(f"校准缓存不存在，将执行完整校准: {self.cache_file}")
        return None
    
    def write_calibration_cache(self, cache: bytes) -> None:
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"校准缓存已保存: {self.cache_file}")
    
    def __del__(self) -> None:
        if hasattr(self, 'device_input') and self.device_input is not None:
            self.device_input.free()
            print("释放校准器设备内存")



def load_weights(file_path: str) -> dict:
    """加载WTS格式的权重文件。
    
    从指定路径加载WTS格式的权重文件，并将其转换为字典格式。
    
    Args:
        file_path (str): WTS权重文件路径。
    
    Returns:
        dict: 包含权重的字典，键为层名称，值为对应的权重数组。
    
    Raises:
        AssertionError: 如果权重文件不存在。
        ValueError: 如果权重文件格式不支持。
    """
    print(f"Loading weights: {file_path}")

    assert os.path.exists(file_path), f'Unable to load weight file: {file_path}'
    
    if not file_path.endswith('.wts'):
        raise ValueError(f"Unsupported weight file format: {file_path}. Only .wts is supported")
    
    weight_map = {}
    
    print("  - 检测到.wts文件，使用TensorRT标准格式加载")
    
    with open(file_path, "r", encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    
    count = int(lines[0])
    print(f"  - Weight file contains {count} layers")
    
    for i in range(1, count + 1):
        splits = lines[i].split()
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits), f"Layer {name} has incorrect weight count"
        
        values = []
        for j in range(2, len(splits)):
            values.append(struct.unpack(">f", bytes.fromhex(splits[j]))[0])
        
        weight_map[name] = np.array(values, dtype=np.float32)
        
        if i % 80 == 0:
            print(f"    Loaded {i}/{count} layers...")
    
    print(f"  - 成功加载 {len(weight_map)} 层权重")
    return weight_map


def build_engine(max_batch_size: int, builder: trt.Builder, config: trt.IBuilderConfig, input_dtype: trt.DataType, use_int8: bool = False,
                 weight_path: str = "resnet50_imagenette_best_8031.wts",
                 input_blob_name: str = "data", input_h: int = 224, input_w: int = 224, output_size: int = 10,
                 output_blob_name: str = "prob", eps: float = 1e-5, calib_dir: str = "calib_images",
                 calib_batch_size: int = 8, calib_input_shape: tuple = (3, 224, 224), calib_dataset_size: int = 2000) -> trt.IHostMemory:
    """构建TensorRT引擎。"""
    print("\n开始构建TensorRT引擎...")
    print("  - 显式批处理模式: 启用")
    print(f"  - INT8量化: {'启用' if use_int8 else '禁用'}")
    print(f"  - 最大批大小: {max_batch_size}")
    
    # 加载权重
    weight_map = load_weights(weight_path)
    
    # 创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 添加输入张量
    input_tensor = network.add_input(
        name=input_blob_name,
        dtype=input_dtype,
        shape=trt.Dims([max_batch_size, 3, input_h, input_w])
    )
    
    # 构建ResNet50网络
    output_tensor = build_resnet50_network(network, weight_map, input_tensor, output_size, eps)
    
    # 标记输出
    output_tensor.name = output_blob_name
    network.mark_output(output_tensor)
    
    # 配置Builder
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    
    # 配置INT8量化
    if use_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        calibrator = Int8Calibrator(calib_dir, calib_batch_size, calib_input_shape, input_h=input_h, input_w=input_w, calib_dataset_size=calib_dataset_size)
        config.int8_calibrator = calibrator
    else:
        config.set_flag(trt.BuilderFlag.FP32)
    
    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)
    
    # 清理资源
    del network
    del weight_map
    
    return serialized_engine


def serialize_engine(max_batch_size: int, use_int8: bool = False,
                   weight_path: str = "resnet50_imagenette_best_8031.wts",
                   input_blob_name: str = "data", input_h: int = 224, input_w: int = 224, output_size: int = 10,
                   output_blob_name: str = "prob", eps: float = 1e-5, calib_dir: str = "calib_images",
                   calib_batch_size: int = 8, engine_path: str = "resnet50_int8_v2.engine", calib_dataset_size: int = 2000) -> None:
    """序列化TensorRT引擎。"""
    print("\n" + "="*50)
    print("TensorRT ResNet50 INT8 量化流程")
    print("="*50)
    
    # 1. 验证校准目录
    if use_int8 and not os.path.exists(calib_dir):
        print(f"校准目录不存在: {calib_dir}")
        # 尝试查找校准目录
        found = False
        for path in [
            os.path.join(os.path.dirname(__file__), calib_dir),
            os.path.join(os.getcwd(), calib_dir),
            os.path.abspath(calib_dir),
        ]:
            if os.path.exists(path) and os.path.isdir(path):
                calib_dir = path
                found = True
                print(f"✓ 找到校准目录: {path}")
                break
        
        if not found:
            # 向上查找
            for _ in range(3):
                parent_dir = os.path.dirname(os.getcwd())
                candidate_path = os.path.join(parent_dir, "calib_images")
                if os.path.exists(candidate_path) and os.path.isdir(candidate_path):
                    calib_dir = candidate_path
                    found = True
                    print(f"✓ 在父目录找到校准目录: {candidate_path}")
                    break
        
        if not found:
            raise FileNotFoundError(f"找不到校准图像目录，请使用 --calib-dir 参数指定正确路径")
    
    # 2. 创建Builder和Config
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    config = builder.create_builder_config()
    
    # 设置工作区大小
    workspace_size = 2 << 30  # 2GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    # 3. 构建序列化引擎
    serialized_engine = build_engine(
        max_batch_size, builder, config, trt.float32, use_int8,
        weight_path=weight_path,
        input_blob_name=input_blob_name, input_h=input_h, input_w=input_w, output_size=output_size,
        output_blob_name=output_blob_name, eps=eps, calib_dir=calib_dir,
        calib_batch_size=calib_batch_size, calib_input_shape=(3, input_h, input_w),
        calib_dataset_size=calib_dataset_size
    )
    
    if serialized_engine is None:
        raise RuntimeError("引擎构建失败: build_serialized_network返回None")
    
    # 4. 保存引擎
    create_output_directory(engine_path)
    
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"\n引擎已保存到: {engine_path}")
    print(f"文件大小: {os.path.getsize(engine_path) / (1024**2):.2f} MB")
    
    # 5. 验证引擎
    print("\n验证引擎完整性...")
    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        raise RuntimeError("引擎验证失败")
    
    print("  - 引擎验证成功!")
    print(f"  - 精度模式: {'INT8' if use_int8 else 'FP32'}")


def test_inference(engine_path: str = "resnet50_int8_v2.engine") -> None:
    """测试TensorRT引擎的推理功能。"""
    print("\n" + "="*50)
    print("测试TensorRT引擎推理")
    print("="*50)
    
    # 加载引擎
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"引擎文件不存在: {engine_path}")
    
    trt_logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        raise RuntimeError("引擎加载失败")
    
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("执行上下文创建失败")
    
    # 获取输入输出形状
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)
    
    print(f"引擎加载成功: 输入形状 {input_shape}, 输出形状 {output_shape}")
    
    # 准备测试数据
    host_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    
    # 生成随机输入
    np.random.seed(42)
    test_data = np.random.randn(*input_shape).astype(np.float32)
    np.copyto(host_input, test_data.ravel())
    
    # 执行推理
    device_input = cuda.mem_alloc(host_input.nbytes)
    device_output = cuda.mem_alloc(host_output.nbytes)
    bindings = [int(device_input), int(device_output)]
    stream = cuda.Stream()
    
    # 异步推理
    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    
    # 显示结果
    output_data = host_output.reshape(output_shape)
    print("\n推理结果:")
    print(f"  - 输出张量形状: {output_data.shape}")
    print(f"  - 输出值范围: [{output_data.min():.4f}, {output_data.max():.4f}]")
    print(f"  - 前5个输出值: {output_data[0, :5]}")
    
    # 清理资源
    device_input.free()
    device_output.free()
    
    print("\n推理测试完成!")
