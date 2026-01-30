# 配置文件目录

## 目录说明

本目录包含配置示例（YAML）。当前仓库的主要入口以脚本命令行参数为主（quantization/、evaluation/、resnet/），这些 YAML 不会被默认脚本自动加载，但可作为二次封装时的配置来源。

## 文件结构

```
configs/
├── base_config.yaml                # 基础配置
└── environment.yaml                # 环境配置（如 CUDA 版本、依赖库等）
```

## 文件说明

### base_config.yaml
包含模型、数据集、量化和评估的基础配置，如模型类型、输入尺寸、批次大小、量化方法等。

主要配置项：
- **model**：模型相关配置，如模型名称、输入尺寸等
- **dataset**：数据集相关配置，如数据集路径、批次大小等
- **quantization**：量化相关配置，如量化方法、校准数据路径等
- **evaluation**：评估相关配置，如评估指标、批次大小等

### environment.yaml
包含环境配置，如CUDA版本、Python版本、依赖库版本和设备配置等。

主要配置项：
- **cuda**：CUDA相关配置，如CUDA版本、设备ID等
- **python**：Python相关配置，如Python版本等
- **dependencies**：依赖库版本配置
- **device**：设备相关配置，如设备类型、内存大小等

## 使用说明

### 配置文件加载（示例）

在代码中加载配置文件：

```python
import yaml

# 加载基础配置
with open('configs/base_config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# 加载环境配置
with open('configs/environment.yaml', 'r') as f:
    env_config = yaml.safe_load(f)
```

### 命令行指定配置文件

仓库当前未提供统一的 main.py 入口；如需以配置文件驱动运行，建议在你自己的入口脚本中读取 YAML 后调用对应模块：

- Torch.FX：quantization/Torch/torch_PTQ.py
- TensorRT：quantization/TensorRT/tensorrt_PTQ_pth2trt_int8.py 或 resnet/run.py
- ModelOpt：quantization/ModelOpt/ModelOpt_PTQ.py
- 评估：evaluation/ 下三个 comprehensive_evaluation_*.py

## 配置文件格式

配置文件使用YAML格式，具有以下特点：
- 层级结构清晰，易于阅读和维护
- 支持注释，方便添加配置说明
- 支持多种数据类型，如字符串、数字、列表、字典等
- 支持环境变量引用

## 配置示例

### base_config.yaml示例

```yaml
model:
  name: resnet50
  input_size: 224
  num_classes: 10

 dataset:
  root: data_set/imagenette
  batch_size: 50
  num_workers: 4

quantization:
  method: torch_fx_ptq
  calibration_data_path: models/ONNX/calibration_data_500.npy
  output_path: models/optimized/resnet50_quantized.onnx

 evaluation:
  batch_size: 32
  metrics: [accuracy, precision, recall, f1_score]
```

### environment.yaml示例

```yaml
cuda:
  version: 11.8
  device_id: 0
  memory_limit: 2GB

python:
  version: 3.10

 dependencies:
  pytorch: 2.4.1
  torchvision: 0.19.1
  tensorrt: 10.x
  onnx: 1.19.x
  onnxruntime-gpu: 1.23.x

 device:
  type: gpu
  name: NVIDIA GeForce RTX 3090
  memory: 24GB
```

## 注意事项

- 配置文件应保持简洁，只包含必要的配置项
- 每个配置项应包含详细的注释，说明其含义和取值范围
- 敏感配置项（如API密钥、密码等）不应直接硬编码到配置文件中，应使用环境变量
- 配置文件应进行版本控制，确保不同环境下的配置一致性
- 大型配置文件可拆分为多个子文件，便于维护和管理
