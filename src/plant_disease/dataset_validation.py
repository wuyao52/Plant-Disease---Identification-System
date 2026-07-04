from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .data import SUPPORTED_IMAGE_EXTENSIONS, is_saved_split_dataset


def validate_dataset(
    root: str | Path,
    min_images_per_class: int = 5,
    check_images: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    report: dict[str, Any] = {
        "root": str(root_path.resolve()),
        "exists": root_path.exists(),
        "format": "missing",
        "valid": False,
        "errors": [],
        "warnings": [],
        "num_classes": 0,
        "total_images": 0,
        "class_counts": {},
        "split_counts": {},
        "invalid_images": [],
    }
    if not root_path.exists():
        report["errors"].append("Dataset directory does not exist.")
        return report

    if is_saved_split_dataset(root_path):
        report["format"] = "saved_split"
        validate_saved_split(root_path, report, min_images_per_class, check_images)
    else:
        report["format"] = "imagefolder"
        validate_imagefolder(root_path, report, min_images_per_class, check_images)

    report["valid"] = not report["errors"]
    return report


def validate_imagefolder(
    root: Path,
    report: dict[str, Any],
    min_images_per_class: int,
    check_images: bool,
) -> None:
    class_counts: dict[str, int] = {}
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = list_image_files(class_dir)
        if images:
            class_counts[class_dir.name] = len(images)
            if len(images) < min_images_per_class:
                report["warnings"].append(
                    f"Class '{class_dir.name}' has only {len(images)} images; "
                    f"recommended minimum is {min_images_per_class}."
                )
            if check_images:
                check_image_files(images, report)

    report["class_counts"] = class_counts
    report["num_classes"] = len(class_counts)
    report["total_images"] = sum(class_counts.values())
    if len(class_counts) < 2:
        report["errors"].append("Dataset must contain at least two non-empty class folders.")
    if report["invalid_images"]:
        report["errors"].append("Dataset contains unreadable or corrupted image files.")


def validate_saved_split(
    root: Path,
    report: dict[str, Any],
    min_images_per_class: int,
    check_images: bool,
) -> None:
    split_counts: dict[str, dict[str, int]] = {}
    split_class_sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            report["errors"].append(f"Missing split directory: {split}")
            continue
        split_counts[split] = {}
        split_class_sets[split] = set()
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            images = list_image_files(class_dir)
            if not images:
                continue
            split_counts[split][class_dir.name] = len(images)
            split_class_sets[split].add(class_dir.name)
            if len(images) < min_images_per_class and split == "train":
                report["warnings"].append(
                    f"Training class '{class_dir.name}' has only {len(images)} images; "
                    f"recommended minimum is {min_images_per_class}."
                )
            if check_images:
                check_image_files(images, report)

    class_sets = list(split_class_sets.values())
    if class_sets and any(class_set != class_sets[0] for class_set in class_sets):
        report["errors"].append("train/val/test must contain the same class folders.")
    classes = sorted(set().union(*class_sets)) if class_sets else []
    report["num_classes"] = len(classes)
    report["split_counts"] = split_counts
    report["total_images"] = sum(count for split in split_counts.values() for count in split.values())
    if len(classes) < 2:
        report["errors"].append("Dataset must contain at least two classes.")
    if report["invalid_images"]:
        report["errors"].append("Dataset contains unreadable or corrupted image files.")


def list_image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def check_image_files(paths: list[Path], report: dict[str, Any]) -> None:
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            report["invalid_images"].append(
                {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
