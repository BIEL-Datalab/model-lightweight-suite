# 模型量化项目

## 项目简介

本项目是一个基于模型的PTQ（Post-Training Quantization）INT8量化项目，支持多种量化工具链，包括TensorRT、Torch.FX和ModelOpt等。

## 项目基本工作流程
1. 获取训练好的模型（从pytorch平台或自行训练）
2. 根据需求选择量化工具链（quantization目录），并进行量化
3. 使用evaluation中对应的评估脚本对量化后的模型进行评估

## 环境隔离
本项目按工具链拆分为两套独立环境，避免依赖交叉引用：

- torch241_cu118_py310：用于 Torch.FX / TensorRT 量化与评估
- modelopt_env：用于 ModelOpt(ONNX) 量化与评估

对应依赖清单在 requirements/ 目录：

- requirements/torch241_cu118_py310.txt
- requirements/modelopt_env.txt

### 环境要求（建议）
- 操作系统：Windows / Linux（Jetson 见 quantization/jetson）
- Python：3.10（环境名中的 py310 为默认推荐）
- NVIDIA GPU：TensorRT / ONNXRuntime GPU / CUDAExecutionProvider 相关功能需要

注意：仓库根目录仍保留 requirements.txt，但它不再代表“单一可用环境”，请以 requirements/ 下的分环境清单为准。

## 快速开始

### 1) 安装依赖

Torch/TensorRT 工具链（torch241_cu118_py310）：

```bash
pip install -r requirements/torch241_cu118_py310.txt
```

ModelOpt 工具链（modelopt_env）：

```bash
pip install -r requirements/modelopt_env.txt
```

### 2) 量化
- Torch.FX：运行 quantization/Torch/torch_PTQ.py
- TensorRT：运行 quantization/TensorRT/tensorrt_PTQ_pth2trt_int8.py 或 resnet/run.py（ResNet50）
- ModelOpt：运行 quantization/ModelOpt/ModelOpt_PTQ.py

### 3) 评估
- Torch INT8：运行 evaluation/comprehensive_evaluation_torch_int8.py
- TensorRT INT8：运行 evaluation/comprehensive_evaluation_tensorrt_int8.py
- ModelOpt INT8：运行 evaluation/comprehensive_evaluation_modelopt_int8.py

## 项目结构

```
model-tiny-suite/
├── docs/                           # 文档目录
├── configs/                        # 配置示例（当前脚本以命令行参数为主）
├── data_set/                       # 数据集说明（数据集不随仓库分发）
├── evaluation/                     # 评估入口（薄封装）+ 模块化实现
│   ├── pipelines/                  # Torch/TensorRT 评估流水线实现
│   └── modelopt/                   # ModelOpt 评估实现（拆分模块）
├── quantization/                   # 量化入口脚本（按工具链分目录）
├── resnet/                         # ResNet50 TensorRT PTQ 自包含工具包（python -m resnet.run）
├── models/                         # 模型目录（运行时会自动创建更多子目录）
├── results/                        # 评估结果输出目录（按工具链写入）
├── requirements/                   # 分环境依赖清单
├── utils/                          # 通用工具函数（避免隐式跨工具链依赖）
└── README.md
```

## 已支持工具链教程
- [安装依赖与工具链使用](./quantization/README.md)
- [嵌入式设备部署](./quantization/jetson/README.md)

## 已支持模型测试结果

| 模型 | 平台 | 设备 | 工具链 | 模式 | 输入尺寸(HxW) | 精度变化(%) | 推理提升(倍) |
|-|-|:-:|:-:|:-:|:-:|:-:|:-:|
| ResNet50 | 服务器 | Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz | Torch | INT8 | 224x224 | -0.67 | 5.58 |
| ResNet50 | 服务器 | GTX3090 | ModelOpt | INT8 | 224x224 | +0.3 | 2.31 |
| ResNet50 | 服务器 | GTX3090 | TensorRT | INT8 | 224x224 | -0.53 | 8.89 |
| ResNet50 | Jetson orin NX |  Cortex-A78AE  | Torch | INT8 | 224x224 | -0.16 | 1.08 |
| ResNet50 | Jetson orin NX | Orin | TensorRT | INT8 | 224x224 | -0.9 | 7.81 |

## 已支持模型使用方法
- [resnet](./resnet/README.md)

## 代码结构说明

### 量化工具链

本项目支持三种主要的量化工具链，每种工具链都有其独立的实现：

1. **Torch.FX PTQ**：使用PyTorch内置的FX框架进行INT8量化，适合PyTorch生态系统
2. **ModelOpt PTQ**：使用ModelOpt工具进行INT8量化，支持多种后端
3. **TensorRT PTQ**：使用NVIDIA TensorRT进行INT8量化，支持GPU加速，适合生产环境部署

### 代码组织

- **utils/quantization_utils.py**：包含所有量化工具链共享的通用工具函数，如模型评估、结果保存等
- **utils/torch_quantization_utils.py**：PyTorch量化专用工具函数
- **utils/tensorrt_quantization_utils.py**：TensorRT量化专用工具函数
- **utils/modelopt_quantization_utils.py**：ModelOpt量化专用工具函数
- **quantization/Torch/torch_PTQ.py**：Torch.FX PTQ量化的主脚本，包含完整的量化流程
- **quantization/TensorRT/tensorrt_PTQ_pth2trt_int8.py**：TensorRT PTQ量化的主脚本
- **quantization/ModelOpt/ModelOpt_PTQ.py**：ModelOpt PTQ量化的主脚本

### 核心功能模块

1. **模型加载与初始化**：负责加载预训练模型并进行必要的初始化
2. **数据准备**：准备训练、验证和校准数据
3. **量化配置**：配置量化参数，如量化方式、校准方法等
4. **模型校准**：使用校准数据对模型进行校准，生成量化缩放因子
5. **模型转换**：将原始模型转换为量化模型
6. **模型评估**：评估量化前后模型的精度和性能
7. **结果保存**：保存量化结果和评估报告

## 参考资料

- [PyTorch量化文档](https://docs.pytorch.org/docs/2.4/quantization.html#)
- [TensorRT量化指南](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8_calib)
- [ModelOpt量化文档](https://github.com/NVIDIA/Model-Optimizer/tree/main)

## 联系方式

如有任何问题或建议，请通过以下方式联系我：

- 邮箱：863392184@qq.com
- Issues：[GitHub Issues](https://github.com/BIEL-Datalab/model-lightweight-suite/issues)

## 贡献指南
- 建议先阅读 docs/developer_guide/ 与各子目录 README（quantization/evaluation/resnet）
- 贡献代码时请保持模块边界：resnet 必须自包含；utils 不应引入 modelopt 的隐式依赖

## 许可证

