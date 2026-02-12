"""评估报告构建工具。

本模块将评估得到的结果字典整理为：
- JSON 结构化报告（便于程序读取）
- Markdown 人类可读报告（便于直接查看/分享）
"""

from __future__ import annotations

from datetime import datetime

import torch


def build_json_report(
    results: dict[str, dict[str, object]],
    model_paths: dict[str, str],
    test_config: dict[str, object],
) -> dict[str, object]:
    """构建 JSON 格式的评估报告数据结构。

    功能描述：
    将 ``results`` 中不同后端的评估结果提取为可序列化字段（float/str/list 等），
    并附带评估时间、测试配置与模型路径信息。

    参数说明：
    - results (dict[str, dict[str, object]]): 评估结果字典，外层键为模型后端标识。
    - model_paths (dict[str, str]): 模型路径字典。
    - test_config (dict[str, object]): 测试配置字典（批次大小、样本数、设备等）。

    返回值说明：
    - dict[str, object]: JSON 报告字典。

    可能抛出的异常：
    - KeyError: 当 ``results`` 缺少必需键时触发。

    使用示例：
    >>> from evaluation.modelopt.reporting import build_json_report
    >>> _ = build_json_report(results={"pytorch": {}, "onnx_original": {}, "onnx_quantized": {}, "tensorrt": {}}, model_paths={}, test_config={})  # doctest: +SKIP
    """
    report_data = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_config": test_config,
        "model_paths": model_paths,
        "results": {},
    }
    for model_key, model_name in [
        ("pytorch", "PyTorch"),
        ("onnx_original", "ONNX原始"),
        ("onnx_quantized", "ONNX量化"),
        ("tensorrt", "TensorRT"),
    ]:
        res = results[model_key]
        report_data["results"][model_key] = {
            "model_type": model_name,
            "accuracy": float(res["accuracy"]),
            "precision": float(res["precision"]),
            "recall": float(res["recall"]),
            "f1_score": float(res["f1_score"]),
            "avg_inference_time_s": float(res["avg_inference_time"]),
            "avg_throughput_fps": float(res["throughput"]),
            "avg_gpu_memory_mb": float(res.get("avg_gpu_memory_mb", 0.0)),
        }
    return report_data


def build_markdown_report(
    results: dict[str, dict[str, object]], model_paths: dict[str, str], device_display: str, batch_size: int
) -> str:
    """构建 Markdown 格式的评估报告文本。

    功能描述：
    将评估结果以 Markdown 表格与结论小节的形式输出，便于直接查看与分享。

    参数说明：
    - results (dict[str, dict[str, object]]): 评估结果字典。
    - model_paths (dict[str, str]): 模型路径字典。
    - device_display (str): 设备展示字符串。
    - batch_size (int): 批次大小。

    返回值说明：
    - str: Markdown 报告内容。

    可能抛出的异常：
    - KeyError: 当 ``results`` 缺少必需键时触发。

    使用示例：
    >>> from evaluation.modelopt.reporting import build_markdown_report
    >>> _ = build_markdown_report(results={"pytorch": {"labels": []}, "onnx_original": {}, "onnx_quantized": {}, "tensorrt": {}}, model_paths={}, device_display="cpu", batch_size=1)  # doctest: +SKIP
    """
    def row(key: str) -> str:
        """构建结果表格的一行字符串（内部工具函数）。"""
        r = results[key]
        return (
            f"| {key} | {device_display} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1_score']:.4f} | {r['avg_inference_time']:.4f} | {r['throughput']:.2f} | "
            f"{r.get('avg_gpu_memory_mb', 0):.2f} |"
        )

    accuracy_diff = results["tensorrt"]["accuracy"] - results["pytorch"]["accuracy"]
    throughput_improvement = (
        (results["tensorrt"]["throughput"] / results["pytorch"]["throughput"] - 1) * 100
        if results["pytorch"]["throughput"] > 0
        else 0.0
    )

    return f"""# ModelOpt INT8 模型评估报告

## 评估基本信息

- **评估时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **测试样本数量**: {len(results['pytorch']['labels'])}
- **批次大小**: {batch_size}
- **设备**: {device_display}
- **PyTorch 版本**: {torch.__version__}

## 模型路径

- **PyTorch 模型**: {model_paths['pytorch']}
- **ONNX 原始模型**: {model_paths['onnx_original']}
- **ONNX 量化模型**: {model_paths['onnx_quantized']}
- **TensorRT 模型**: {model_paths['tensorrt']}

## 详细结果

| 模型 | 设备 | 准确率 | 精确率 | 召回率 | F1分数 | 平均推理时间(s) | 吞吐量(FPS) | 平均GPU内存(MB) |
|------|------|--------|--------|--------|--------|------------------|-------------|-------------------|
| PyTorch | {device_display} | {results['pytorch']['accuracy']:.4f} | {results['pytorch']['precision']:.4f} | {results['pytorch']['recall']:.4f} | {results['pytorch']['f1_score']:.4f} | {results['pytorch']['avg_inference_time']:.4f} | {results['pytorch']['throughput']:.2f} | {results['pytorch'].get('avg_gpu_memory_mb', 0):.2f} |
| ONNX原始 | {device_display} | {results['onnx_original']['accuracy']:.4f} | {results['onnx_original']['precision']:.4f} | {results['onnx_original']['recall']:.4f} | {results['onnx_original']['f1_score']:.4f} | {results['onnx_original']['avg_inference_time']:.4f} | {results['onnx_original']['throughput']:.2f} | {results['onnx_original'].get('avg_gpu_memory_mb', 0):.2f} |
| ONNX量化 | {device_display} | {results['onnx_quantized']['accuracy']:.4f} | {results['onnx_quantized']['precision']:.4f} | {results['onnx_quantized']['recall']:.4f} | {results['onnx_quantized']['f1_score']:.4f} | {results['onnx_quantized']['avg_inference_time']:.4f} | {results['onnx_quantized']['throughput']:.2f} | {results['onnx_quantized'].get('avg_gpu_memory_mb', 0):.2f} |
| TensorRT | {device_display} | {results['tensorrt']['accuracy']:.4f} | {results['tensorrt']['precision']:.4f} | {results['tensorrt']['recall']:.4f} | {results['tensorrt']['f1_score']:.4f} | {results['tensorrt']['avg_inference_time']:.4f} | {results['tensorrt']['throughput']:.2f} | {results['tensorrt'].get('avg_gpu_memory_mb', 0):.2f} |

## 综合评估结论

- **准确率差异 (TensorRT - PyTorch)**: {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)
- **吞吐量提升 (TensorRT vs PyTorch)**: {throughput_improvement:.1f}%
"""
