# 模型评估目录

## 目录说明

本目录包含模型评估测试脚本，用于评估模型的性能和精度，支持多种评估指标和评估方式。

## 文件结构

```
evaluation/
├── comprehensive_evaluation_modelopt_int8.py        # ModelOpt INT8评估入口（薄封装）
├── comprehensive_evaluation_tensorrt_int8.py        # TensorRT INT8评估入口（薄封装）
├── comprehensive_evaluation_torch_int8.py           # Torch INT8评估入口（薄封装）
├── modelopt/                                        # ModelOpt评估实现（模块化拆分）
├── pipelines/                                       # Torch/TensorRT评估流程实现
└── README.md                                        # 目录说明文档
```

## 文件说明

### comprehensive_evaluation_torch_int8.py
Torch INT8综合评估脚本，用于评估使用PyTorch量化的INT8模型性能和精度。

主要功能：
- 支持PyTorch INT8模型的评估
- 计算多种评估指标
- 评估推理时间和吞吐量
- 生成详细的评估报告

使用示例：

```bash
python evaluation/comprehensive_evaluation_torch_int8.py --visualization
```

### comprehensive_evaluation_modelopt_int8.py
ModelOpt INT8综合评估脚本，用于评估使用ModelOpt量化的INT8模型性能和精度。

主要功能：
- 支持ModelOpt INT8模型的评估
- 计算准确率、精确率、召回率和F1分数
- 评估推理时间和吞吐量
- 生成详细的评估报告和可视化图表

使用示例：

```bash
python evaluation/comprehensive_evaluation_modelopt_int8.py --device cuda:0 --visualization
```

### comprehensive_evaluation_tensorrt_int8.py
TensorRT INT8综合评估脚本，用于评估使用TensorRT量化的INT8模型性能和精度。

主要功能：
- 支持TensorRT INT8引擎模型的评估
- 计算多种评估指标
- 评估推理时间和吞吐量
- 生成详细的评估报告

使用示例：

```bash
python evaluation/comprehensive_evaluation_tensorrt_int8.py
```



## 评估指标

### 精度指标
- **准确率（Accuracy）**：预测正确的样本数占总样本数的比例
- **精确率（Precision）**：预测为正类的样本中实际为正类的比例
- **召回率（Recall）**：实际为正类的样本中被预测为正类的比例
- **F1分数**：精确率和召回率的调和平均值
- **混淆矩阵**：展示模型在不同类别上的预测结果

### 性能指标
- **推理时间（Inference Time）**：模型处理一个样本所需的时间
- **吞吐量（Throughput）**：模型每秒可以处理的样本数
- **CPU内存占用**：模型推理过程中占用的CPU内存
- **GPU内存占用**：模型推理过程中占用的GPU内存

## 评估结果

评估结果将保存在`results/`目录中，包含以下内容：
- 评估报告（evaluation_results.json 与 benchmark_report.md）
- 可视化图表（可选，accuracy_metrics_comparison.png / performance_comparison.png / gpu_memory_comparison.png / cpu_memory_comparison.png / confusion_matrices.png）
