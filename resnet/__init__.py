"""ResNet 与 TensorRT PTQ 工具子包。

本包提供围绕 ResNet50 在 ImageFolder 风格数据集上的 TensorRT PTQ（可选 INT8）流程，
包括校准图片准备、WTS 权重转换、TensorRT 引擎构建与推理验证等能力。
"""
