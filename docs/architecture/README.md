# 架构文档目录

## 目录说明

本目录记录项目的关键架构约束与模块边界，便于在重构与扩展时保持一致性。

## 文档内容

- **工具链分层**：quantization/（量化入口）→ models/（模型产物）→ evaluation/（评估入口与实现）
- **环境隔离**：Torch/TensorRT 与 ModelOpt 分环境安装，避免隐式依赖交叉
- **模块边界**：
  - evaluation/ 入口脚本保持薄封装；实现位于 evaluation/pipelines 与 evaluation/modelopt
  - utils/ 仅放通用工具；禁止通过 utils/__init__.py 聚合导入工具链特定模块
  - resnet/ 为自包含工具包，只引用 resnet/ 内部与第三方库

## 文档格式

- 文档使用Markdown格式编写
- 包含清晰的标题和结构
- 适当使用图表说明复杂概念
- 代码示例使用代码块包裹

## 注意事项

- 架构文档应定期更新，保持与代码同步
- 新增功能时应同时更新相关架构文档
- 架构变更应记录在架构决策记录中
