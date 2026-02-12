"""ResNet50 TensorRT 网络构建函数。

本模块以函数方式构建 ResNet50 的 TensorRT 网络拓扑（显式 batch），并提供 BatchNorm 与 bottleneck 的构建辅助函数。
"""

from __future__ import annotations

import numpy as np


def add_batchnorm_2d(
    network: object, weight_map: dict[str, np.ndarray], input_tensor: object, layer_name: str, eps: float
) -> object:
    """向 TensorRT 网络添加 BatchNorm2d 对应的 Scale 层。

    功能描述：
    从 ``weight_map`` 中读取 BatchNorm 参数并计算 scale/shift，通过 ``network.add_scale`` 注入网络。

    参数说明：
    - network (object): TensorRT 网络定义对象。
    - weight_map (dict[str, np.ndarray]): 权重字典。
    - input_tensor (object): 输入张量对象。
    - layer_name (str): BatchNorm 层名前缀（不含 ``.weight`` 等后缀）。
    - eps (float): 数值稳定项。

    返回值说明：
    - object: TensorRT Scale layer 对象。

    可能抛出的异常：
    - KeyError: 当 ``weight_map`` 缺少必要键时触发。
    - Exception: 当 TensorRT API 调用失败时由底层触发。

    使用示例：
    >>> from resnet.trt.network import add_batchnorm_2d
    >>> _ = add_batchnorm_2d(object(), {}, object(), "bn1", 1e-5)  # doctest: +SKIP
    """
    import tensorrt as trt

    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta

    return network.add_scale(input=input_tensor, mode=trt.ScaleMode.CHANNEL, shift=shift, scale=scale)


def bottleneck(
    network: object,
    weight_map: dict[str, np.ndarray],
    input_tensor: object,
    in_channels: int,
    out_channels: int,
    stride: int,
    layer_name: str,
    eps: float,
) -> object:
    """向 TensorRT 网络添加 ResNet bottleneck 块。

    功能描述：
    构建 1x1、3x3、1x1 卷积与对应 BN+ReLU，并在需要时创建 downsample 分支后做残差相加。

    参数说明：
    - network (object): TensorRT 网络定义对象。
    - weight_map (dict[str, np.ndarray]): 权重字典。
    - input_tensor (object): 输入张量对象。
    - in_channels (int): 输入通道数。
    - out_channels (int): bottleneck 中间通道数。
    - stride (int): 3x3 卷积步幅。
    - layer_name (str): block 层名前缀（例如 ``"layer1.0."``）。
    - eps (float): BatchNorm 数值稳定项。

    返回值说明：
    - object: 最后一层 ReLU 的 TensorRT layer 对象。

    可能抛出的异常：
    - KeyError: 当 ``weight_map`` 缺少必要权重键时触发。
    - Exception: 当 TensorRT API 调用失败时由底层触发。

    使用示例：
    >>> from resnet.trt.network import bottleneck
    >>> _ = bottleneck(object(), {}, object(), 64, 64, 1, "layer1.0.", 1e-5)  # doctest: +SKIP
    """
    import tensorrt as trt

    conv1 = network.add_convolution_nd(
        input=input_tensor,
        num_output_maps=out_channels,
        kernel_shape=trt.Dims([1, 1]),
        kernel=weight_map[layer_name + "conv1.weight"],
        bias=trt.Weights(),
    )
    bn1 = add_batchnorm_2d(network, weight_map, conv1.get_output(0), layer_name + "bn1", eps)
    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)

    conv2 = network.add_convolution_nd(
        input=relu1.get_output(0),
        num_output_maps=out_channels,
        kernel_shape=trt.Dims([3, 3]),
        kernel=weight_map[layer_name + "conv2.weight"],
        bias=trt.Weights(),
    )
    conv2.stride_nd = trt.Dims([stride, stride])
    conv2.padding_nd = trt.Dims([1, 1])
    bn2 = add_batchnorm_2d(network, weight_map, conv2.get_output(0), layer_name + "bn2", eps)
    relu2 = network.add_activation(bn2.get_output(0), type=trt.ActivationType.RELU)

    conv3 = network.add_convolution_nd(
        input=relu2.get_output(0),
        num_output_maps=out_channels * 4,
        kernel_shape=trt.Dims([1, 1]),
        kernel=weight_map[layer_name + "conv3.weight"],
        bias=trt.Weights(),
    )
    bn3 = add_batchnorm_2d(network, weight_map, conv3.get_output(0), layer_name + "bn3", eps)

    if stride != 1 or in_channels != 4 * out_channels:
        conv4 = network.add_convolution_nd(
            input=input_tensor,
            num_output_maps=out_channels * 4,
            kernel_shape=trt.Dims([1, 1]),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=trt.Weights(),
        )
        conv4.stride_nd = trt.Dims([stride, stride])
        bn4 = add_batchnorm_2d(network, weight_map, conv4.get_output(0), layer_name + "downsample.1", eps)
        residual = bn4.get_output(0)
    else:
        residual = input_tensor

    ew_sum = network.add_elementwise(bn3.get_output(0), residual, trt.ElementWiseOperation.SUM)
    relu3 = network.add_activation(ew_sum.get_output(0), type=trt.ActivationType.RELU)
    return relu3


def build_resnet50_network(
    network: object,
    input_tensor: object,
    weight_map: dict[str, np.ndarray],
    input_h: int,
    input_w: int,
    output_size: int,
    output_blob_name: str,
    eps: float,
    use_int8: bool,
) -> None:
    """构建 ResNet50 的 TensorRT 网络并标记输出。

    功能描述：
    在 ``network`` 上构建 ResNet50 拓扑并最终 ``mark_output``；
    当 ``use_int8`` 为 True 时，会将最后的 matmul 与 add 输出强制为 FLOAT 以提升兼容性。

    参数说明：
    - network (object): TensorRT 网络定义对象。
    - input_tensor (object): 输入张量对象。
    - weight_map (dict[str, np.ndarray]): 权重字典。
    - input_h (int): 输入高度。
    - input_w (int): 输入宽度。
    - output_size (int): 输出类别数。
    - output_blob_name (str): 输出张量名称。
    - eps (float): BatchNorm 数值稳定项。
    - use_int8 (bool): 是否启用 INT8 相关精度设置。

    返回值说明：
    - None: 无返回值；函数内部直接 ``mark_output``。

    可能抛出的异常：
    - KeyError: 当 ``weight_map`` 缺少必要权重键时触发。
    - Exception: 当 TensorRT API 调用失败时由底层触发。

    使用示例：
    >>> from resnet.trt.network import build_resnet50_network
    >>> build_resnet50_network(object(), object(), {}, 224, 224, 10, "prob", 1e-5, True)  # doctest: +SKIP
    """
    import tensorrt as trt

    conv1 = network.add_convolution_nd(
        input=input_tensor,
        num_output_maps=64,
        kernel_shape=trt.Dims([7, 7]),
        kernel=weight_map["conv1.weight"],
        bias=trt.Weights(),
    )
    conv1.stride_nd = trt.Dims([2, 2])
    conv1.padding_nd = trt.Dims([3, 3])
    bn1 = add_batchnorm_2d(network, weight_map, conv1.get_output(0), "bn1", eps)
    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)

    pool1 = network.add_pooling_nd(input=relu1.get_output(0), window_size=trt.Dims([3, 3]), type=trt.PoolingType.MAX)
    pool1.stride_nd = trt.Dims([2, 2])
    pool1.padding_nd = trt.Dims([1, 1])

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

    pool2 = network.add_pooling_nd(input=x.get_output(0), window_size=trt.Dims([7, 7]), type=trt.PoolingType.AVERAGE)
    pool2.stride_nd = trt.Dims([1, 1])

    shuffle = network.add_shuffle(pool2.get_output(0))
    shuffle.reshape_dims = trt.Dims([-1, 2048])
    reshaped_out = shuffle.get_output(0)

    fc_weight = weight_map["fc.weight"].reshape(output_size, 2048)
    fc_bias = weight_map["fc.bias"]

    weight_const = network.add_constant(trt.Dims([2048, output_size]), np.ascontiguousarray(fc_weight.T))
    matmul = network.add_matrix_multiply(
        reshaped_out,
        trt.MatrixOperation.NONE,
        weight_const.get_output(0),
        trt.MatrixOperation.NONE,
    )
    if use_int8:
        matmul.precision = trt.DataType.FLOAT
        matmul.set_output_type(0, trt.DataType.FLOAT)

    bias_const = network.add_constant(trt.Dims([1, output_size]), np.ascontiguousarray(fc_bias.reshape(1, output_size)))
    output_layer = network.add_elementwise(matmul.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM)
    if use_int8:
        output_layer.precision = trt.DataType.FLOAT
        output_layer.set_output_type(0, trt.DataType.FLOAT)

    output_tensor = output_layer.get_output(0)
    output_tensor.name = output_blob_name
    network.mark_output(output_tensor)
