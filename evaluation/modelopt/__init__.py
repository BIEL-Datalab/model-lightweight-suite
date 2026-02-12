"""基于 ModelOpt/ONNX/TensorRT 的 INT8 评估子包。

本子包提供统一的加载器、评估器、报告生成与流水线封装，用于在相同数据与配置下对比：
- PyTorch FP32
- ONNX 原始
- ONNX 量化
- TensorRT 引擎
"""
