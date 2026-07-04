from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(slots=True)
class ClassificationReport:
    confusion_matrix: list[list[int]]
    per_class: list[dict[str, float | int | str]]
    accuracy: float


def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> ClassificationReport:
    model.eval()
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            for true_label, predicted_label in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                matrix[int(true_label), int(predicted_label)] += 1

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_class = []
    for index, class_name in enumerate(class_names):
        support = int(matrix[index].sum())
        true_positive = int(matrix[index, index])
        predicted_total = int(matrix[:, index].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_total if predicted_total else 0.0
        per_class.append(
            {
                "class_name": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "accuracy": recall,
            }
        )

    return ClassificationReport(
        confusion_matrix=matrix.tolist(),
        per_class=per_class,
        accuracy=correct / total if total else 0.0,
    )


def save_confusion_matrix_csv(
    path: str | Path,
    matrix: list[list[int]],
    class_names: list[str],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        file.write("actual\\predicted," + ",".join(class_names) + "\n")
        for class_name, row in zip(class_names, matrix, strict=True):
            file.write(class_name + "," + ",".join(str(value) for value in row) + "\n")
