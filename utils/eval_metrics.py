"""评估指标计算工具。

本模块面向分类任务，提供常用指标（accuracy/precision/recall/f1/confusion_matrix）的统一计算与结构化返回。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    """分类任务指标集合。

    功能描述：
    以不可变数据结构的形式承载分类任务的若干常用指标，便于在评估流程中传递、序列化与报告输出。

    参数说明：
    - accuracy (float): 准确率（0-1）。
    - precision (float): 宏平均精确率（0-1）。
    - recall (float): 宏平均召回率（0-1）。
    - f1_score (float): 宏平均 F1（0-1）。
    - confusion_matrix (list[list[int]]): 混淆矩阵（行是真实类别，列是预测类别）。

    返回值说明：
    - ClassificationMetrics: 指标对象。

    可能抛出的异常：
    - 无。该类本身不执行计算逻辑。

    使用示例：
    >>> from utils.eval_metrics import ClassificationMetrics
    >>> m = ClassificationMetrics(accuracy=1.0, precision=1.0, recall=1.0, f1_score=1.0, confusion_matrix=[[1]])
    >>> m.as_dict()["accuracy"]
    1.0
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict[str, float | list[list[int]]]:
        """将指标对象转换为可序列化字典。

        功能描述：
        以固定键集合返回当前指标，便于 JSON 落盘或报告模块消费。

        参数说明：
        - 无。

        返回值说明：
        - dict[str, float | list[list[int]]]: 指标字典。

        可能抛出的异常：
        - 无。
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_classification_metrics(labels: list[int], preds: list[int]) -> ClassificationMetrics:
    """计算分类任务评估指标。

    功能描述：
    基于 scikit-learn 的指标函数计算 accuracy、宏平均 precision/recall/f1，并生成混淆矩阵。

    参数说明：
    - labels (list[int]): 真实标签序列。
    - preds (list[int]): 预测标签序列。

    返回值说明：
    - ClassificationMetrics: 指标对象。

    可能抛出的异常：
    - ValueError: 当输入长度不一致或标签取值不合法时，由 scikit-learn 触发。

    使用示例：
    >>> from utils.eval_metrics import compute_classification_metrics
    >>> metrics = compute_classification_metrics([0, 1, 1], [0, 1, 0])
    >>> round(metrics.accuracy, 6)
    0.666667
    """
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels_np, preds_np)),
        precision=float(precision_score(labels_np, preds_np, average="macro", zero_division=0)),
        recall=float(recall_score(labels_np, preds_np, average="macro", zero_division=0)),
        f1_score=float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(labels_np, preds_np).tolist(),
    )
