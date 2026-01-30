# 模型目录

本目录仅包含“模型定义代码”和“量化产物目录说明”（不包含训练/导出产物本身）。目前仓库实际提交的模型定义为 ResNet 系列，位于 [base/resnet.py](file:///d:/3_CODES/model-tiny-suite/models/base/resnet.py)。

## 项目概述与主要功能
- 提供 ResNet 家族模型定义与工厂函数：`resnet18/34/50/101/152`、`resnext*`、`wide_resnet*`
- 提供 TorchVision 风格的 `WeightsEnum`（可选预训练权重下载与默认预处理 transforms）
- 作为本仓库量化/评估脚本的模型源代码依赖（脚本入口位于 quantization/ 与 evaluation/）

## 目录结构（与仓库实际内容同步）

```
models/
├── README.md
├── base/
│   └── resnet.py
└── quantized/
    └── README.md
```

### 文件/模块说明
- [base/resnet.py](file:///d:/3_CODES/model-tiny-suite/models/base/resnet.py)：ResNet/ResNeXt/Wide-ResNet 定义（TorchVision 实现风格）
- [quantized/README.md](file:///d:/3_CODES/model-tiny-suite/models/quantized/README.md)：量化模型目录与命名规范说明（目录说明文件本身随仓库提交）

## 安装与部署

本目录不是独立可部署应用，它属于仓库代码的一部分。安装依赖请在仓库根目录进行：

- Torch/TensorRT 工具链（推荐）：`pip install -r requirements/torch241_cu118_py310.txt`
- ModelOpt 工具链：`pip install -r requirements/modelopt_env.txt`

## 主要依赖项与版本要求

本目录中的 [base/resnet.py](file:///d:/3_CODES/model-tiny-suite/models/base/resnet.py) 直接依赖：
- `torch`（模型/张量/层定义）
- `torchvision`（Weights/Transforms/注册接口）

仓库推荐版本以根目录依赖清单为准（torch 2.4.1 + torchvision 0.19.1 组合）。

## 使用方法与示例

### 1) 构建一个自定义类别数的 ResNet50（不加载预训练）

```python
import torch
from models.base.resnet import resnet50

model = resnet50(weights=None, num_classes=10)
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = model(x)
print(y.shape)  # torch.Size([1, 10])
```

### 2) 加载 TorchVision 预训练权重（需要联网下载）

```python
from models.base.resnet import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()
```

## API 文档与接口说明

### 核心类
- `ResNet`：主网络类，支持 `num_classes/groups/width_per_group/replace_stride_with_dilation` 等参数  
  - 定义位置：[resnet.py](file:///d:/3_CODES/model-tiny-suite/models/base/resnet.py#L166-L286)
- `BasicBlock` / `Bottleneck`：残差块实现  
  - 定义位置：[resnet.py](file:///d:/3_CODES/model-tiny-suite/models/base/resnet.py#L59-L164)

### 工厂函数（返回 `ResNet` 实例）
- `resnet18/resnet34/resnet50/resnet101/resnet152`
- `resnext50_32x4d/resnext101_32x8d/resnext101_64x4d`
- `wide_resnet50_2/wide_resnet101_2`

这些函数的参数约定与 TorchVision 接口保持一致：`weights`（可选）、`progress`（下载进度）、以及透传到 `ResNet` 的 `**kwargs`。

### Weights 枚举
- `ResNet18_Weights`、`ResNet34_Weights`、`ResNet50_Weights`、`ResNet101_Weights`、`ResNet152_Weights` 等  
  - 典型用法：`ResNet50_Weights.DEFAULT`

## 配置指南与注意事项
- **类别数**：通过工厂函数的 `num_classes` 指定；若传入 `weights!=None`，会自动覆盖 `num_classes` 为权重元数据中的类别数（ImageNet-1K 为 1000）。
- **输入尺寸**：Weights 默认 transforms 以 224 crop 为主（见 WeightsEnum 元数据），自定义输入需自行准备预处理。
- **网络下载**：使用 `WeightsEnum` 会触发从 `pytorch.org` 下载权重文件；离线环境请提前缓存或使用本地权重加载逻辑（本目录未提供）。

## 测试方法与质量保证

建议在仓库根目录执行以下冒烟检查：

```bash
python -m py_compile models/base/resnet.py
python -c "from models.base.resnet import resnet50; import torch; m=resnet50(weights=None, num_classes=10); y=m(torch.randn(1,3,224,224)); print(y.shape)"
```

## 贡献指南与开发规范
- 保持与 TorchVision API 的兼容性：新增模型或改动参数时，请维持工厂函数签名风格（`weights/progress/**kwargs`）。
- 如需新增其他模型族，请在 `models/base/` 下新增独立文件，并在本 README 的“目录结构/API”中同步更新。
