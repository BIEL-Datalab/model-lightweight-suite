from __future__ import annotations

import numpy as np


def add_batchnorm_2d(network, weight_map: dict[str, np.ndarray], input_tensor, layer_name: str, eps: float):
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
    network,
    weight_map: dict[str, np.ndarray],
    input_tensor,
    in_channels: int,
    out_channels: int,
    stride: int,
    layer_name: str,
    eps: float,
):
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
    network,
    input_tensor,
    weight_map: dict[str, np.ndarray],
    input_h: int,
    input_w: int,
    output_size: int,
    output_blob_name: str,
    eps: float,
    use_int8: bool,
):
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

