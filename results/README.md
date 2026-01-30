# 模型评估结果目录

## 目录说明

本目录用于保存模型评估结果，包括评估报告、可视化图表和性能分析报告等。

## 结果文件结构

```
results/
├── Torch/                                    # Torch 工具链评估结果
│   └── evaluation_result_int8_YYYYMMDD_HHMMSS/
│       ├── evaluation_results.json
│       ├── benchmark_report.md
│       ├── accuracy_metrics_comparison.png   # 可视化（可选）
│       ├── performance_comparison.png        # 可视化（可选）
│       ├── gpu_memory_comparison.png         # 可视化（可选）
│       ├── cpu_memory_comparison.png         # 可视化（可选）
│       └── confusion_matrices.png            # 可视化（可选）
├── TensorRT/                                 # TensorRT 工具链评估结果
│   └── evaluation_result_int8_YYYYMMDD_HHMMSS/
│       ├── evaluation_results.json
│       └── benchmark_report.md
├── ModelOpt/                                 # ModelOpt 工具链评估结果
│   └── evaluation_result_int8_YYYYMMDD_HHMMSS/
│       ├── evaluation_results.json
│       ├── benchmark_report.md
│       └── trt_cache/                        # ORT TensorRT EP 缓存（可选）
└── ...
```

## 结果文件说明

### evaluation_results.json
评估结果（结构化 JSON），包含测试配置、模型路径与各模型/后端的指标汇总。

### benchmark_report.md
Markdown 格式的评估报告，便于直接阅读与分享，包含关键指标表格与对比结论。

### 可视化图表（可选）
当运行评估入口脚本时传入 `--visualization`，会额外生成 PNG 图表（文件名以实际脚本输出为准）。

## 结果分析

### 精度分析
- 比较INT8模型和FP32模型的精度差异
- 分析量化对不同类别精度的影响
- 评估模型在不同批次大小下的精度表现

### 性能分析
- 比较INT8模型和FP32模型的推理时间和吞吐量
- 分析量化对模型性能的提升效果
- 评估模型在不同硬件环境下的表现

### 综合分析
- 评估量化带来的精度损失和性能提升之间的权衡
- 分析模型的适用场景
- 提出模型优化建议
