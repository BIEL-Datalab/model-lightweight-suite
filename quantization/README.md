# 模型量化目录

## 目录说明

本目录包含与模型量化相关的所有文件和子目录，支持多种量化工具链和部署平台。

## 子目录结构

```
quantization/
├── jetson/                           # Jetson设备部署说明文件
├── ModelOpt/                         # ModelOpt量化方法
│   └── ModelOpt_PTQ.py               # ModelOpt PTQ量化主脚本
├── PC/                               # PC设备部署说明文件
├── TensorRT/                         # TensorRT量化方法
│   └── tensorrt_PTQ_pth2trt_int8.py  # TensorRT PTQ量化主脚本
├── Torch/                            # PyTorch量化方法
│   └── torch_PTQ.py                  # Torch.FX PTQ量化主脚本
└── README.md                         # 目录说明文档
```

## 子目录说明

### jetson/
包含Jetson设备部署说明文件。

### PC/
包含PC平台上部署说明文件。

### ModelOpt/
包含使用ModelOpt进行模型量化的相关脚本，如PyTorch到ONNX转换、ModelOpt PTQ量化等。

### TensorRT/
包含使用TensorRT进行模型量化的相关脚本，如PyTorch模型到WTS格式的转换、TensorRT PTQ量化等。
- **tensorrt_PTQ_pth2trt_int8.py**：TensorRT PTQ量化主脚本

### Torch/
包含使用PyTorch内置量化功能进行模型量化的相关脚本，如Torch.FX PTQ量化等。
- **torch_PTQ.py**：Torch.FX PTQ量化主脚本

## 公共方法库

所有量化工具链共用的公共方法已迁移至项目的`utils/quantization_utils.py`文件中，主要包含：
- `AverageMeter`：用于统计指标的平均值
- `accuracy`：计算模型准确率
- `create_output_directory`：创建输出目录
- `save_quantization_results`：保存量化结果到JSON文件
- `print_quantization_report`：打印量化评估报告

## 支持的量化方法

1. **Torch.FX PTQ**：使用PyTorch内置的FX框架进行INT8量化，适合PyTorch生态系统
2. **ModelOpt PTQ**：使用ModelOpt工具进行INT8量化，支持多种后端
3. **TensorRT PTQ**：使用NVIDIA TensorRT进行INT8量化，支持GPU加速，适合生产环境部署

## 使用说明

### 环境准备

在开始使用本项目之前，请确保您已安装所有必要的依赖项：

#### 1.Torch和TensorRT工具链环境
```bash
pip install -r requirements/torch241_cu118_py310.txt
```

#### 2.ModelOpt工具链环境
```bash
pip install -r requirements/modelopt_env.txt
```

### 模型准备
#### 方法一
下载预训练模型权重放在以下目录中：
```
models/pretrained
```
#### 方法二
将训练好的模型放在一下目录中：
```
models/trained
```

### 数据准备

准备与模型训练相同的数据集，并确保数据集包含训练集和验证集
- **训练数据**：用于模型训练和校准
- **验证数据**：用于模型评估和测试
- **校准数据**：用于INT8量化校准

### PyTorch.FX量化

使用PyTorch FX进行INT8量化

#### 步骤1：准备校准数据+量化

```bash
python quantization/Torch/torch_PTQ.py
```

该脚本将：
- 加载训练后的FP32模型
- 使用FX Graph模式进行静态量化
- 准备校准数据集
- 执行INT8量化
- 保存量化后的模型
- 初步评估量化前后的模型精度和性能

##### 参数说明：

| 参数名 | 类型 | 默认值 | 说明 |
|-|:-:|:-:|:-:|
| `data_path` | str | `data_set/imagenette` | 数据集路径 |
| `float_model_file` | str | `models/trained/resnet50_imagenette_best_8031.pth` | 预训练/训练后的模型路径 |
| `saved_model_dir` | str | `models/quantized/int8/` | 量化模型保存目录 |
| `save_quantized_model_path` | str | `saved_model_dir + f"resnet50_imagenette_quantized_int8.pth"` | 量化模型保存路径 |
| `calib_batch_size` | int | 50 | 校准批次大小 |
| `num_workers` | int | 47 | 数据加载时启用的子进程数量 |
| `device` | str | `cpu` | 使用的设备名称 |

#### 步骤2：基于Torch量化模型评估测试

```bash
python evaluation/comprehensive_evaluation_torch_int8.py
python evaluation/comprehensive_evaluation_torch_int8.py --visualization  # 同时保存可视化结果
```

说明：当前评估入口脚本为“薄封装”，默认的模型/数据/结果路径写在脚本内部；如需自定义路径，建议复制入口脚本并在内部调整默认值，或在 evaluation/pipelines 中扩展参数化能力。

### TensorRT量化

使用TensorRT进行INT8量化：

#### 步骤1：准备校准数据集

```bash
python utils/prepare_calib_images.py
python -m resnet.run calib-images  # ResNet50 专用的自包含工具入口
```

##### 命令行参数:

| 参数名 | 类型 | 默认值 | 说明 |
|-|:-:|:-:|:-:|
| `--input-dir` | str | `data_set/imagenette/train` | 源图像目录（支持ImageNet格式，包含类别子目录） |
| `--output-dir` | str | `data_set/calib_images` | 校准图像保存目录 |
| `--num-images` | int | 2000 | 校准图像数量 |
| `--target-size` | int | `[224, 224]` | 目标图像尺寸 |
| `--seed` | int | 42 | 随机种子，用于可复现的结果 |

#### 步骤2：模型转换+量化

```bash
python quantization/TensorRT/tensorrt_PTQ_pth2trt_int8.py
```

##### 命令行参数:

| 参数名 | 类型 | 默认值 | 说明 |
|-|:-:|:-:|:-:|
| `--cuda_id` | int | 0 | 使用第几块GPU |
| `--input` | str | `models/trained/resnet50_imagenette_best_8031.pth` | 输入PyTorch模型文件路径 (.pth 或 .pt) |
| `--output` | str | `models/converted/resnet50_imagenette_best_8031_x86.wts` | 输出WTS文件路径 |
| `--batch_size` | int | 1 | 批次大小 |
| `--input_h` | int | 224 | 输入图像高度 |
| `--input_w` | int | 224 | 输入图像宽度 |
| `--output_size` | int | 10 | 数据集类别数 |
| `--input_blob_name` | str | `data` | 输入张量名称 |
| `--output_blob_name` | str | `prob` | 输出张量名称 |
| `--eps` | float | 1e-5 | 数值稳定常数 |
| `--model-name` | str | `resnet50` | 模型名称，用于日志输出 |
| `--skip-convert` | bool | False | 跳过PyTorch模型到WTS格式的转换，直接使用现有的WTS文件 |
| `--serialize` | bool | True | 序列化模型到引擎文件 |
| `--deserialize` | bool | False | 从引擎文件加载并测试推理 |
| `--int8` | bool | True | 使用INT8量化 (仅在序列化时有效) |
| `--calib-dir` | str | `data_set/calib_images` | 校准图像目录路径 |
| `--weight-path` | str | `models/converted/resnet50_imagenette_best_8031.wts` | 权重文件路径 |
| `--engine-path` | str | `models/quantized/int8/resnet50_int8_x86.engine` | 引擎文件路径 |
| `--calib-batch-size` | int | 8 | 校准批次大小 |
| `--calib-dataset-size` | int | 2000 | 校准数据集大小 |

#### 步骤3：基于TensorRT量化模型评估测试

```bash
python evaluation/comprehensive_evaluation_tensorrt_int8.py
```

说明：评估入口脚本默认写入 results/TensorRT，并输出 evaluation_results.json / benchmark_report.md。

### ModelOpt量化

使用ModelOpt进行INT8量化

#### 步骤1：准备校准数据

#### 步骤2：模型转换+量化
```bash
python quantization/ModelOpt/ModelOpt_PTQ.py
python quantization/ModelOpt/ModelOpt_PTQ.py -h  # 查看全部参数
```

##### 命令行参数:

| 参数名 | 类型 | 默认值 | 说明 |
|-|:-:|:-:|:-:|
| `--input` | str | `models/trained/resnet50_imagenette_best_8031.pth` | PyTorch模型文件路径 |
| `--output_onnx` | str | `models/converted/resnet50_imagenette_best_8031.onnx` | 输出ONNX模型文件路径 |
| `--output_quant` | str | `models/quantized/int8/resnet50_imagenette_modelopt_int8_x86.onnx` | 输出量化模型文件路径 |
| `--calibration_data` | str | `data_set/calibration_data_500.npy` | 校准数据路径 |
| `--dataset_path` | str | `data_set/imagenette` | 数据集路径 |
| `--batch_size` | int | 50 | 批次大小 |
| `--num_workers` | int | 47 | 数据加载器工作线程数 |
| `--device` | str | `cuda` | 设备类型 |
| `--skip-convert` | bool | False | 跳过PyTorch模型到ONNX格式的转换，直接使用现有的ONNX文件 |
| `--model-name` | str | `resnet50` | 模型名称，用于日志输出 |

#### 步骤3：TensorRT编译
```bash
bash quantization/ModelOpt/convert_onnx_to_tensorrt.sh
bash quantization/ModelOpt/convert_onnx_to_tensorrt.sh -h  # 显示帮助信息
```

#### 步骤4：基于ModelOpt量化模型评估测试
```bash
python evaluation/comprehensive_evaluation_modelopt_int8.py --device cuda:0
python evaluation/comprehensive_evaluation_modelopt_int8.py --device cuda:0 --visualization  # 同时保存可视化结果
```

说明：ModelOpt 评估入口脚本默认写入 results/ModelOpt，并对 PyTorch/ONNX/TensorRT 四种后端做对比评估；如需修改路径请编辑 evaluation/comprehensive_evaluation_modelopt_int8.py 中的默认值。

##### 注意事项

- 校准数据应具有代表性，覆盖数据集的主要特征，不同类别的样本量尽可能呈均匀分布
- 不同量化方法的效果可能不同，应根据实际需求选择合适的方法
- 量化后的模型应进行全面的评估，包括精度、速度和内存占用等指标
