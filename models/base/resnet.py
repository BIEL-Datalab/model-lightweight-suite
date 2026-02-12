"""ResNet/ResNeXt/Wide-ResNet 模型定义（TorchVision 风格实现）。

本模块为 TorchVision 的 ResNet 家族实现风格的本地拷贝版本，提供：
- ResNet 基类与 BasicBlock/Bottleneck 结构；
- resnet18/resnet34/resnet50 等工厂函数；
- 对应的权重枚举（WeightsEnum）与元数据。
"""

from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._api import register_model as _tv_register_model
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


def register_model(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """注册模型工厂函数到 TorchVision 的模型注册表。

    功能描述：
    包装 ``torchvision.models._api.register_model``，并在遇到重复注册（同名已存在）时返回原函数以保持幂等，
    避免本地拷贝与 TorchVision 内置注册发生冲突。

    参数说明：
    - *args (Any): 透传给 TorchVision ``register_model`` 的位置参数。
    - **kwargs (Any): 透传给 TorchVision ``register_model`` 的关键字参数。

    返回值说明：
    - Callable[[Callable[..., Any]], Callable[..., Any]]: 装饰器，用于装饰模型工厂函数。

    可能抛出的异常：
    - ValueError: 当发生非“重复注册”类错误时由底层抛出。

    使用示例：
    >>> from models.base.resnet import register_model
    >>> deco = register_model()
    >>> callable(deco)
    True
    """
    decorator = _tv_register_model(*args, **kwargs)

    def _inner(fn: Callable[..., Any]) -> Callable[..., Any]:
        """执行实际注册逻辑（内部函数）。"""
        try:
            return decorator(fn)
        except ValueError as e:
            if "already registered under the name" in str(e):
                return fn
            raise

    return _inner


__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """创建一个 3x3 卷积层（带 padding）。

    功能描述：
    创建 ``kernel_size=3`` 且 ``padding=dilation`` 的 ``nn.Conv2d``，默认不带 bias，
    用于 ResNet 结构中的常见卷积配置。

    参数说明：
    - in_planes (int): 输入通道数。
    - out_planes (int): 输出通道数。
    - stride (int): 步幅，默认 1。
    - groups (int): 分组卷积组数，默认 1。
    - dilation (int): 空洞卷积膨胀系数，默认 1。

    返回值说明：
    - nn.Conv2d: 构建好的卷积层。

    可能抛出的异常：
    - ValueError: 当输入参数不合法导致 ``nn.Conv2d`` 构造失败时触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import conv3x3
    >>> m = conv3x3(3, 8)
    >>> m(torch.randn(1, 3, 224, 224)).shape[1]
    8
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """创建一个 1x1 卷积层。

    功能描述：
    创建 ``kernel_size=1`` 的 ``nn.Conv2d``，默认不带 bias，常用于通道变换与 downsample 分支。

    参数说明：
    - in_planes (int): 输入通道数。
    - out_planes (int): 输出通道数。
    - stride (int): 步幅，默认 1。

    返回值说明：
    - nn.Conv2d: 构建好的卷积层。

    可能抛出的异常：
    - ValueError: 当输入参数不合法导致 ``nn.Conv2d`` 构造失败时触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import conv1x1
    >>> m = conv1x1(3, 8)
    >>> m(torch.randn(1, 3, 224, 224)).shape[1]
    8
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet 的 BasicBlock 残差块实现。

    功能描述：
    该 block 由两层 3x3 卷积组成，并在必要时通过 ``downsample`` 分支匹配残差的空间尺寸/通道数。

    参数说明：
    - inplanes (int): 输入通道数。
    - planes (int): block 的输出通道数（不含 expansion）。
    - stride (int): 第一层卷积步幅。
    - downsample (nn.Module | None): 下采样分支模块（用于匹配残差），可为 ``None``。
    - groups (int): 分组卷积组数（BasicBlock 仅支持 1）。
    - base_width (int): 基础宽度（BasicBlock 仅支持 64）。
    - dilation (int): 空洞系数（BasicBlock 不支持 >1）。
    - norm_layer (Callable[..., nn.Module] | None): 归一化层工厂；为 ``None`` 时使用 ``nn.BatchNorm2d``。

    返回值说明：
    - BasicBlock: block 实例。

    可能抛出的异常：
    - ValueError: 当 ``groups != 1`` 或 ``base_width != 64`` 时触发。
    - NotImplementedError: 当 ``dilation > 1`` 时触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import BasicBlock
    >>> b = BasicBlock(inplanes=64, planes=64)
    >>> y = b(torch.randn(1, 64, 56, 56))
    >>> y.shape
    torch.Size([1, 64, 56, 56])
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """初始化 BasicBlock。

        功能描述：
        构建两层 3x3 卷积及对应的归一化与激活，并保存残差分支与步幅配置。

        参数说明：
        - inplanes (int): 输入通道数。
        - planes (int): 输出通道数。
        - stride (int): 第一层卷积步幅。
        - downsample (nn.Module | None): 残差下采样分支。
        - groups (int): 分组卷积组数（仅支持 1）。
        - base_width (int): 基础宽度（仅支持 64）。
        - dilation (int): 空洞系数（不支持 >1）。
        - norm_layer (Callable[..., nn.Module] | None): 归一化层工厂。

        返回值说明：
        - None: 无返回值。

        可能抛出的异常：
        - ValueError: 当参数不满足 BasicBlock 约束时触发。
        - NotImplementedError: 当 dilation 不被支持时触发。
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """前向传播。

        功能描述：
        对输入执行两次卷积+归一化+ReLU，并与（可选的）残差分支相加后再 ReLU。

        参数说明：
        - x (Tensor): 输入张量，形状通常为 ``(N, C, H, W)``。

        返回值说明：
        - Tensor: 输出张量，形状与主分支输出一致。

        可能抛出的异常：
        - RuntimeError: 当张量形状/设备不匹配导致算子执行失败时由 PyTorch 触发。
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet 的 Bottleneck 残差块实现（V1.5 变体）。

    功能描述：
    该 block 由 1x1、3x3、1x1 三层卷积组成，其中下采样 stride 放置在第二层 3x3 卷积（TorchVision V1.5 风格），
    并在必要时通过 ``downsample`` 分支匹配残差的空间尺寸/通道数。

    参数说明：
    - inplanes (int): 输入通道数。
    - planes (int): bottleneck 的基准通道数（最终输出通道为 ``planes * expansion``）。
    - stride (int): 3x3 卷积步幅。
    - downsample (nn.Module | None): 下采样分支模块（用于匹配残差），可为 ``None``。
    - groups (int): 分组卷积组数。
    - base_width (int): 每组的宽度基准。
    - dilation (int): 空洞系数。
    - norm_layer (Callable[..., nn.Module] | None): 归一化层工厂；为 ``None`` 时使用 ``nn.BatchNorm2d``。

    返回值说明：
    - Bottleneck: block 实例。

    可能抛出的异常：
    - RuntimeError: 当张量形状/设备不匹配导致算子执行失败时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import Bottleneck
    >>> b = Bottleneck(inplanes=256, planes=64)
    >>> y = b(torch.randn(1, 256, 56, 56))
    >>> y.shape[1]
    256
    """
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """初始化 Bottleneck。"""
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """前向传播。

        功能描述：
        对输入依次执行 1x1、3x3、1x1 卷积与归一化/激活，并与（可选的）残差分支相加后再 ReLU。

        参数说明：
        - x (Tensor): 输入张量，形状通常为 ``(N, C, H, W)``。

        返回值说明：
        - Tensor: 输出张量。

        可能抛出的异常：
        - RuntimeError: 当张量形状/设备不匹配导致算子执行失败时由 PyTorch 触发。
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet 主干网络实现。

    功能描述：
    基于给定 block 类型与各 stage 的层数配置，构建 ResNet 系列网络，并在 forward 中输出分类 logits。

    参数说明：
    - block (type[BasicBlock | Bottleneck]): 残差块类型。
    - layers (list[int]): 四个 stage 的 block 数量列表。
    - num_classes (int): 分类类别数，默认 1000。
    - zero_init_residual (bool): 是否对残差分支的最后 BN 做零初始化，默认 False。
    - groups (int): 分组卷积组数，默认 1。
    - width_per_group (int): 每组通道宽度，默认 64。
    - replace_stride_with_dilation (list[bool] | None): 是否用 dilation 替换 stride 的配置列表。
    - norm_layer (Callable[..., nn.Module] | None): 归一化层工厂。

    返回值说明：
    - ResNet: 网络实例。

    可能抛出的异常：
    - ValueError: 当 ``replace_stride_with_dilation`` 长度不为 3 时触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnet50
    >>> m = resnet50(weights=None, num_classes=10)
    >>> y = m(torch.randn(1, 3, 224, 224))
    >>> y.shape
    torch.Size([1, 10])
    """
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """初始化 ResNet 网络。"""
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """构建一个 ResNet stage（由多个残差块组成）。

        功能描述：
        根据 ``block`` 类型与 ``blocks`` 数量，创建一个由若干残差块组成的 ``nn.Sequential``；
        当需要改变空间尺寸或通道数时，会自动创建 downsample 分支。

        参数说明：
        - block (type[BasicBlock | Bottleneck]): 残差块类型。
        - planes (int): stage 的基准通道数。
        - blocks (int): 残差块数量。
        - stride (int): stage 首个 block 的步幅。
        - dilate (bool): 是否使用 dilation 替换 stride（会累积到 ``self.dilation``）。

        返回值说明：
        - nn.Sequential: stage 模块。

        可能抛出的异常：
        - RuntimeError: 当模块构造失败或参数不一致时由 PyTorch 触发。
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """ResNet 前向计算的内部实现。

        功能描述：
        执行 stem（conv/bn/relu/maxpool）+ 4 个 stage + 全局平均池化 + 全连接分类头。

        参数说明：
        - x (Tensor): 输入张量，形状通常为 ``(N, 3, H, W)``。

        返回值说明：
        - Tensor: 分类 logits，形状为 ``(N, num_classes)``。

        可能抛出的异常：
        - RuntimeError: 当张量形状/设备不匹配导致算子执行失败时由 PyTorch 触发。
        """
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """前向传播。

        功能描述：
        调用内部实现 ``_forward_impl`` 执行完整前向计算。

        参数说明：
        - x (Tensor): 输入张量。

        返回值说明：
        - Tensor: 分类 logits。

        可能抛出的异常：
        - RuntimeError: 当算子执行失败时由 PyTorch 触发。
        """
        return self._forward_impl(x)


def _resnet(
    block: type[Union[BasicBlock, Bottleneck]],
    layers: list[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    """ResNet 工厂函数的内部实现（私有）。

    功能描述：
    根据 ``layers`` 配置创建 ``ResNet``，并在 ``weights`` 不为空时自动覆盖 ``num_classes`` 并加载预训练权重。

    参数说明：
    - block (type[BasicBlock | Bottleneck]): 残差块类型。
    - layers (list[int]): 四个 stage 的 block 数量列表。
    - weights (WeightsEnum | None): 预训练权重枚举；为 None 表示不加载权重。
    - progress (bool): 是否显示权重下载进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或不匹配时由 PyTorch 触发。
    """
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    """ResNet-18 的预训练权重枚举。

    功能描述：
    封装 ResNet-18 的可用预训练权重与其元数据（类别、指标、训练配方等），并提供权重校验与 state_dict 获取接口。

    参数说明：
    - 无。

    返回值说明：
    - ResNet18_Weights: 枚举成员（例如 ``IMAGENET1K_V1``）。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发（在调用 ``get_state_dict`` 时发生）。

    使用示例：
    >>> from models.base.resnet import ResNet18_Weights
    >>> ResNet18_Weights.IMAGENET1K_V1 is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_ops": 1.814,
            "_file_size": 44.661,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class ResNet34_Weights(WeightsEnum):
    """ResNet-34 的预训练权重枚举。

    功能描述：
    封装 ResNet-34 的可用预训练权重与元数据。

    参数说明：
    - 无。

    返回值说明：
    - ResNet34_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import ResNet34_Weights
    >>> ResNet34_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet34-b627a593.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 21797672,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.314,
                    "acc@5": 91.420,
                }
            },
            "_ops": 3.664,
            "_file_size": 83.275,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class ResNet50_Weights(WeightsEnum):
    """ResNet-50 的预训练权重枚举。

    功能描述：
    封装 ResNet-50 的可用预训练权重与元数据；TorchVision 的 ResNet-50 通常采用 V1.5 变体。

    参数说明：
    - 无。

    返回值说明：
    - ResNet50_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import ResNet50_Weights
    >>> ResNet50_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_ops": 4.089,
            "_file_size": 97.781,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_ops": 4.089,
            "_file_size": 97.79,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNet101_Weights(WeightsEnum):
    """ResNet-101 的预训练权重枚举。

    功能描述：
    封装 ResNet-101 的可用预训练权重与元数据。

    参数说明：
    - 无。

    返回值说明：
    - ResNet101_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import ResNet101_Weights
    >>> ResNet101_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 44549160,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.374,
                    "acc@5": 93.546,
                }
            },
            "_ops": 7.801,
            "_file_size": 170.511,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 44549160,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 95.780,
                }
            },
            "_ops": 7.801,
            "_file_size": 170.53,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNet152_Weights(WeightsEnum):
    """ResNet-152 的预训练权重枚举。

    功能描述：
    封装 ResNet-152 的可用预训练权重与元数据。

    参数说明：
    - 无。

    返回值说明：
    - ResNet152_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import ResNet152_Weights
    >>> ResNet152_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 60192808,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.312,
                    "acc@5": 94.046,
                }
            },
            "_ops": 11.514,
            "_file_size": 230.434,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 60192808,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.284,
                    "acc@5": 96.002,
                }
            },
            "_ops": 11.514,
            "_file_size": 230.474,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt50_32X4D_Weights(WeightsEnum):
    """ResNeXt50_32x4d 的预训练权重枚举。

    功能描述：
    封装 ResNeXt50_32x4d 的可用预训练权重与元数据。

    参数说明：
    - 无。

    返回值说明：
    - ResNeXt50_32X4D_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import ResNeXt50_32X4D_Weights
    >>> ResNeXt50_32X4D_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.618,
                    "acc@5": 93.698,
                }
            },
            "_ops": 4.23,
            "_file_size": 95.789,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.198,
                    "acc@5": 95.340,
                }
            },
            "_ops": 4.23,
            "_file_size": 95.833,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_32X8D_Weights(WeightsEnum):
    """ResNeXt101_32x8d 的预训练权重枚举。

    功能描述：
    封装 ResNeXt101_32x8d 的可用预训练权重与元数据。

    参数说明：
    - 无。

    返回值说明：
    - ResNeXt101_32X8D_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import ResNeXt101_32X8D_Weights
    >>> ResNeXt101_32X8D_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.312,
                    "acc@5": 94.526,
                }
            },
            "_ops": 16.414,
            "_file_size": 339.586,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.834,
                    "acc@5": 96.228,
                }
            },
            "_ops": 16.414,
            "_file_size": 339.673,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_64X4D_Weights(WeightsEnum):
    """ResNeXt101_64x4d 的预训练权重枚举。

    功能描述：
    封装 ResNeXt101_64x4d 的可用预训练权重与元数据。

    参数说明：
    - 无。

    返回值说明：
    - ResNeXt101_64X4D_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import ResNeXt101_64X4D_Weights
    >>> ResNeXt101_64X4D_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 83455272,
            "recipe": "https://github.com/pytorch/vision/pull/5935",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.246,
                    "acc@5": 96.454,
                }
            },
            "_ops": 15.46,
            "_file_size": 319.318,
            "_docs": """
                These weights were trained from scratch by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class Wide_ResNet50_2_Weights(WeightsEnum):
    """Wide_ResNet50_2 的预训练权重枚举。

    功能描述：
    封装 Wide_ResNet50_2 的可用预训练权重与元数据。

    参数说明：
    - 无。

    返回值说明：
    - Wide_ResNet50_2_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import Wide_ResNet50_2_Weights
    >>> Wide_ResNet50_2_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.468,
                    "acc@5": 94.086,
                }
            },
            "_ops": 11.398,
            "_file_size": 131.82,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.602,
                    "acc@5": 95.758,
                }
            },
            "_ops": 11.398,
            "_file_size": 263.124,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class Wide_ResNet101_2_Weights(WeightsEnum):
    """Wide_ResNet101_2 的预训练权重枚举。

    功能描述：
    封装 Wide_ResNet101_2 的可用预训练权重与元数据。

    参数说明：
    - 无。

    返回值说明：
    - Wide_ResNet101_2_Weights: 枚举成员。

    可能抛出的异常：
    - Exception: 当权重下载或校验失败时由底层依赖触发。

    使用示例：
    >>> from models.base.resnet import Wide_ResNet101_2_Weights
    >>> Wide_ResNet101_2_Weights.DEFAULT is not None
    True
    """
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 126886696,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.848,
                    "acc@5": 94.284,
                }
            },
            "_ops": 22.753,
            "_file_size": 242.896,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 126886696,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.510,
                    "acc@5": 96.020,
                }
            },
            "_ops": 22.753,
            "_file_size": 484.747,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """构建 ResNet-18 模型。

    功能描述：
    根据论文 “Deep Residual Learning for Image Recognition” 的结构配置构建 ResNet-18，
    可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (ResNet18_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数（例如 ``num_classes`` 等）。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnet18
    >>> m = resnet18(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """构建 ResNet-34 模型。

    功能描述：
    按 ResNet-34 的层数配置构建模型，并可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (ResNet34_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnet34
    >>> m = resnet34(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """构建 ResNet-50 模型（TorchVision V1.5 变体）。

    功能描述：
    按 ResNet-50 的层数配置构建模型。当前实现采用 TorchVision 常见的 V1.5 变体：
    下采样 stride 放置在 bottleneck 的第二个 3x3 卷积上。

    参数说明：
    - weights (ResNet50_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnet50
    >>> m = resnet50(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet101_Weights.IMAGENET1K_V1))
def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """构建 ResNet-101 模型（TorchVision V1.5 变体）。

    功能描述：
    按 ResNet-101 的层数配置构建模型，并可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (ResNet101_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnet101
    >>> m = resnet101(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = ResNet101_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet152_Weights.IMAGENET1K_V1))
def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """构建 ResNet-152 模型（TorchVision V1.5 变体）。

    功能描述：
    按 ResNet-152 的层数配置构建模型，并可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (ResNet152_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnet152
    >>> m = resnet152(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = ResNet152_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V1))
def resnext50_32x4d(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """构建 ResNeXt-50 32x4d 模型。

    功能描述：
    按 ResNeXt-50 32x4d 的配置构建模型，并自动设置 ``groups=32`` 与 ``width_per_group=4``；
    可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (ResNeXt50_32X4D_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnext50_32x4d
    >>> m = resnext50_32x4d(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt101_32X8D_Weights.IMAGENET1K_V1))
def resnext101_32x8d(
    *, weights: Optional[ResNeXt101_32X8D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """构建 ResNeXt-101 32x8d 模型。

    功能描述：
    按 ResNeXt-101 32x8d 的配置构建模型，并自动设置 ``groups=32`` 与 ``width_per_group=8``；
    可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (ResNeXt101_32X8D_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnext101_32x8d
    >>> m = resnext101_32x8d(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = ResNeXt101_32X8D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt101_64X4D_Weights.IMAGENET1K_V1))
def resnext101_64x4d(
    *, weights: Optional[ResNeXt101_64X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """构建 ResNeXt-101 64x4d 模型。

    功能描述：
    按 ResNeXt-101 64x4d 的配置构建模型，并自动设置 ``groups=64`` 与 ``width_per_group=4``；
    可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (ResNeXt101_64X4D_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import resnext101_64x4d
    >>> m = resnext101_64x4d(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = ResNeXt101_64X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.IMAGENET1K_V1))
def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """构建 Wide ResNet-50-2 模型。

    功能描述：
    该模型与 ResNet 类似，但 bottleneck 中间通道数加倍（通过 ``width_per_group=64*2`` 实现），
    从而提高模型容量；可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (Wide_ResNet50_2_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import wide_resnet50_2
    >>> m = wide_resnet50_2(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", Wide_ResNet101_2_Weights.IMAGENET1K_V1))
def wide_resnet101_2(
    *, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """构建 Wide ResNet-101-2 模型。

    功能描述：
    该模型与 ResNet 类似，但 bottleneck 中间通道数加倍（通过 ``width_per_group=64*2`` 实现）；
    可选加载 ``weights`` 指定的预训练权重。

    参数说明：
    - weights (Wide_ResNet101_2_Weights | None): 预训练权重枚举；为 ``None`` 表示不加载权重。
    - progress (bool): 下载权重时是否显示进度条。
    - **kwargs (Any): 透传给 ``ResNet`` 构造函数的参数。

    返回值说明：
    - ResNet: 构建得到的模型实例。

    可能抛出的异常：
    - RuntimeError: 当权重加载失败或与模型结构不匹配时由 PyTorch 触发。

    使用示例：
    >>> import torch
    >>> from models.base.resnet import wide_resnet101_2
    >>> m = wide_resnet101_2(weights=None, num_classes=10)
    >>> m(torch.randn(1, 3, 224, 224)).shape
    torch.Size([1, 10])
    """
    weights = Wide_ResNet101_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
