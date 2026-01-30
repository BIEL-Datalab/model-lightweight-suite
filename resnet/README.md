# ResNet基于TensorRT实现INT8量化使用说明

## 步骤1：准备校准数据集

```bash
python -m resnet.run calib-images
```

### 命令行参数:

| 参数名 | 类型 | 默认值 | 说明 |
|-|:-:|:-:|:-:|
| `--input-dir` | str | `data_set/imagenette/train` | 源图像目录（支持ImageNet格式，包含类别子目录） |
| `--output-dir` | str | `data_set/calib_images` | 校准图像保存目录 |
| `--num-images` | int | 2000 | 校准图像数量 |
| `--target-size` | int | `[224, 224]` | 目标图像尺寸 |
| `--seed` | int | 42 | 随机种子，用于可复现的结果 |

## 步骤2：模型转换+量化


```bash
python -m resnet.run trt-ptq --serialize --int8
```

### 命令行参数:
建议以 `python -m resnet.run trt-ptq -h` 的输出为准。

| 参数名 | 类型 | 默认值 | 说明 |
|-|:-:|:-:|:-:|
| `--cuda-id` | int | 0 | 使用第几块GPU |
| `--input` | str | `models/trained/resnet50_imagenette_best_8031.pth` | 输入PyTorch模型文件路径 (.pth 或 .pt) |
| `--output` | str | `models/converted/resnet50_imagenette_best_8031.wts` | 输出WTS文件路径 |
| `--batch-size` | int | 1 | 批次大小 |
| `--input-h` | int | 224 | 输入图像高度 |
| `--input-w` | int | 224 | 输入图像宽度 |
| `--output-size` | int | 10 | 数据集类别数 |
| `--input-blob-name` | str | `data` | 输入张量名称 |
| `--output-blob-name` | str | `prob` | 输出张量名称 |
| `--eps` | float | 1e-5 | 数值稳定常数 |
| `--skip-convert` | bool | False | 跳过PyTorch模型到WTS格式的转换，直接使用现有的WTS文件 |
| `--serialize` | bool | False | 序列化模型到引擎文件（需要显式指定） |
| `--deserialize` | bool | False | 从引擎文件加载并测试推理（需要显式指定） |
| `--int8` | bool | True | 使用INT8量化 (仅在序列化时有效) |
| `--calib-dir` | str | `data_set/calib_images` | 校准图像目录路径 |
| `--weight-path` | str | （默认与 `--output` 一致） | 权重文件路径 |
| `--engine-path` | str | `models/quantized/int8/resnet50_int8_x86.engine` | 引擎文件路径 |
| `--calib-batch-size` | int | 8 | 校准批次大小 |
| `--calib-dataset-size` | int | 2000 | 校准数据集大小 |


## 步骤3：基于TensorRT量化模型评估测试

```bash
python evaluation/comprehensive_evaluation_tensorrt_int8.py
```

### 参数说明：

| 参数名 | 类型 | 默认值 | 说明 |
|-|:-:|:-:|:-:|
| `fp32_model_path` | str | `models/trained/resnet50_imagenette_best_8031.pth` | 基准模型路径 |
| `tensorrt_model_path` | str | `models/quantized/int8/resnet50_int8_x86.engine` | INT8量化后模型的路径 |
| `dataset_path` | str | `data_set/imagenette` | 数据集路径 |
| `results_root` | str | `results/TensorRT` | 结果保存目录 |
| `save_dir_type` | str | `evaluation_result_int8` | 结果保存子目录 |
| `batch_size` | int | 8 | 模型批次大小 |
| `test_sample_count` | int | 3000 | 测试样本数量 |
| `num_classes` | int | 10 | 模型类别数量 |
| `num_workers` | int | 47 | 数据加载时启用的子进程数量 |
