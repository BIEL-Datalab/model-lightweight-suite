# 用户指南目录

## 目录说明

本目录提供面向使用者的“安装/运行/评估”指引与常见问题说明。详细入口与参数以各目录 README 为准。

## 快速开始

1. 安装依赖：见根目录 README 的“环境隔离/快速开始”，或直接参考 requirements/ 下的分环境依赖清单
2. 量化：见 quantization/README.md（Torch / TensorRT / ModelOpt）
3. 评估：见 evaluation/README.md（Torch / TensorRT / ModelOpt）
4. ResNet TensorRT PTQ：见 resnet/readme.md（python -m resnet.run）

## 常见问题

- 数据集不随仓库分发：请参考 data_set/README.md 获取下载与目录结构说明
- 结果输出在哪里：默认写入 results/ 下对应工具链子目录（Torch/TensorRT/ModelOpt）
- 环境不匹配：部分入口脚本会检查 Conda 环境名；请切换到对应环境后再运行

## 注意事项

- 用户指南应定期更新，保持与项目功能同步
- 新增功能时应同时更新相关用户指南
- 文档中的命令示例以 “可复制粘贴运行” 为标准维护
