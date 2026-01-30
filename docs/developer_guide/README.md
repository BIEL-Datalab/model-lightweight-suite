# 开发者指南目录

## 目录说明

本目录提供面向开发者的项目维护说明，包括环境搭建、模块边界约束、验证方式与贡献流程。

## 开发环境

- Python：建议 3.10
- 环境隔离：按工具链拆分两个 Conda 环境（避免依赖交叉）
  - torch241_cu118_py310：Torch.FX / TensorRT 量化与评估
  - modelopt_env：ModelOpt(ONNX) 量化与评估
- 依赖清单：位于 requirements/（不要再把 requirements.txt 当作单环境依赖）

## 模块边界（重要）

- resnet/：必须自包含（只允许引用 resnet/ 内部模块与第三方库）
- utils/：只能放“通用且无隐式工具链依赖”的代码；禁止在 utils/__init__.py 聚合导入 modelopt 相关模块
- evaluation/：入口脚本保持薄封装，核心实现位于 evaluation/pipelines 与 evaluation/modelopt

## 验证与测试

- 仓库当前以脚本为主，建议至少执行语法编译冒烟检查：
  - `python -m py_compile <file...>`
- 文档更新需同步验证 README 中的命令可以执行（至少 `-h/--help` 不报错）。

## 贡献流程
- 修改代码同时更新对应目录 README（quantization/evaluation/resnet/utils 等）
- 新增脚本时优先做模块化拆分，保持入口文件短小
- 不要提交模型权重与大型数据集到仓库（README 中提供下载/生成说明即可）

## 许可证

