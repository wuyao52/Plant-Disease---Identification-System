from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .metrics import evaluate_classification, save_confusion_matrix_csv


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total_samples += batch_size

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
    )


def fit(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: ProjectConfig,
    device: torch.device,
    class_names: list[str],
) -> dict[str, object]:
    training_cfg = config.section("training")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_state = None
    epochs_without_improvement = 0
    started_at = time.time()

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            amp=bool(training_cfg["amp"]),
        )
        val_metrics = run_epoch(model, dataloaders["val"], criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
        )

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(training_cfg["patience"]):
            print("early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_epoch(model, dataloaders["test"], criterion, device)
    report = evaluate_classification(model, dataloaders["test"], device, class_names)
    checkpoint_path = config.checkpoint_path
    save_checkpoint(
        checkpoint_path,
        model,
        config,
        class_names,
        metrics={
            "best_val_accuracy": best_val_accuracy,
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
            "elapsed_seconds": time.time() - started_at,
        },
    )
    save_confusion_matrix_csv(
        config.output_dir / "confusion_matrix.csv",
        report.confusion_matrix,
        class_names,
    )

    return {
        "history": history,
        "test": asdict(test_metrics),
        "classification_report": {
            "accuracy": report.accuracy,
            "confusion_matrix": report.confusion_matrix,
            "per_class": report.per_class,
        },
        "best_val_accuracy": best_val_accuracy,
        "checkpoint": str(checkpoint_path),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: ProjectConfig,
    class_names: list[str],
    metrics: dict[str, float],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "config": config.values,
            "metrics": metrics,
        },
        output,
    )


def write_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
