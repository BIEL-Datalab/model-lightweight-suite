from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_classification_metrics(labels: list[int], preds: list[int]) -> ClassificationMetrics:
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels_np, preds_np)),
        precision=float(precision_score(labels_np, preds_np, average="macro", zero_division=0)),
        recall=float(recall_score(labels_np, preds_np, average="macro", zero_division=0)),
        f1_score=float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(labels_np, preds_np).tolist(),
    )

