from __future__ import annotations

from datetime import datetime

import torch


def build_json_report(
    results: dict,
    model_paths: dict,
    test_config: dict,
) -> dict:
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


def build_markdown_report(results: dict, model_paths: dict, device_display: str, batch_size: int) -> str:
    def row(key: str) -> str:
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

