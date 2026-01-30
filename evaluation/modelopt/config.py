from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOptEvalPaths:
    pytorch_model_path: str
    onnx_original_path: str
    onnx_quantized_path: str
    tensorrt_engine_path: str
    dataset_path: str
    results_root: str


@dataclass(frozen=True)
class ModelOptEvalConfig:
    batch_size: int
    num_classes: int
    num_workers: int
    test_sample_count: int
    random_seed: int
    num_warmup_batches: int
    num_warmup: int
    num_iterations: int

